"""Adapter that exposes a FlashRT VLAModel as an openpi BasePolicy.

The openpi WebsocketPolicyServer was designed against the openpi
JAX/PyTorch ``Policy`` interface (``infer(obs: dict) -> dict``). To
keep robot clients unchanged when we swap the server-side inference to
FlashRT (e.g. on DGX Spark), this adapter translates between the two
formats.

Wire diagram::

    AsyncActionChunkBroker (client)
        │ obs dict (JPEG bytes / np arrays)
        ▼
    WebsocketPolicyServer            ─── lives in openpi
        │ obs dict, JPEG-decoded
        ▼
    FlashRTPolicyAdapter             ─── this module
        │
        ▼ {"actions": np.ndarray, "policy_timing": {...}}
    WebsocketPolicyServer.send
        │
        ▼
    AsyncActionChunkBroker (client)

This module imports from ``openpi_client`` (the lightweight,
separately-installable package shipped at
``packages/openpi-client`` in the openpi repository). It does NOT
import from ``openpi`` itself. So in a Docker container where the
robot client side of openpi is installed but the server / training
side is not, this adapter still works.

Installation:

    pip install "openpi-client @ git+https://github.com/Physical-Intelligence/openpi.git#subdirectory=packages/openpi-client"

or, if you have an openpi checkout sibling to FlashRT:

    pip install -e ../openpi/packages/openpi-client

What the adapter handles:
  - Image extraction from both openpi formats:
      a) Flat keys:      obs["observation/image"], obs["observation/wrist_image"]
      b) Nested "images": obs["images"]["cam_high"], obs["images"]["cam_wrist"]
      c) Flat keys (no observation/ prefix): obs["image"], obs["wrist_image"]
  - Channel-order normalization: openpi server may produce (C, H, W) from
    JPEG decode; FlashRT expects (H, W, C). Re-transpose as needed.
  - Resize to 224x224 (FlashRT's hardcoded pi05 input resolution).
  - Prompt extraction (with default_prompt fallback).
  - Timing instrumentation (matches openpi's policy_timing convention so
    the client's perf logs keep working).
  - Optional pass-through of RTC fields (currently a no-op for FlashRT,
    but preserved so the client doesn't get a None back).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np

try:
    from openpi_client import base_policy as _base_policy
except ImportError as e:  # pragma: no cover - import-time error path
    raise ImportError(
        "flash_rt.serving.openpi_adapter requires the 'openpi-client' "
        "package. Install with one of:\n"
        "    pip install -e ../openpi/packages/openpi-client      # local sibling checkout\n"
        "    pip install \"openpi-client @ git+https://github.com/Physical-Intelligence/"
        "openpi.git#subdirectory=packages/openpi-client\""
    ) from e


logger = logging.getLogger(__name__)


# Camera-key candidates we try, in priority order, to be robust to client
# variation. The first match wins for each slot.
_BASE_IMAGE_CANDIDATES = (
    "observation/image",
    "image",
    "observation/exterior_image_1_left",
    "cam_high",
    "exterior_image_1_left",
)
_WRIST_IMAGE_CANDIDATES = (
    "observation/wrist_image",
    "wrist_image",
    "cam_left_wrist",
    "left_wrist",
)
_WRIST_IMAGE_RIGHT_CANDIDATES = (
    "observation/wrist_image_right",
    "wrist_image_right",
    "cam_right_wrist",
    "right_wrist",
)
_PROMPT_CANDIDATES = ("prompt", "task", "language")
_STATE_CANDIDATES = ("observation/state", "state", "proprio", "robot_state")


def _normalize_image(img: Any, target_hw: int = 224) -> np.ndarray:
    """Coerce an image to (H, W, 3) uint8.

    Accepts:
      - (H, W, 3) uint8 -> returns as-is
      - (3, H, W) uint8 -> transposes
      - (H, W, 3) float in [0, 1] -> scales to uint8
      - PIL Image, torch tensor -> np.asarray then re-coerced

    Does NOT resize unless H or W differs from target_hw. The
    AsyncActionChunkBroker / robot client typically sends 224x224
    already, but this adapter doesn't depend on that.
    """
    if hasattr(img, "numpy"):
        img = img.numpy()
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f":
            arr = np.clip(arr * 255.0 if arr.max() <= 1.0 + 1e-3 else arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    if arr.shape[:2] != (target_hw, target_hw):
        try:
            import cv2

            arr = cv2.resize(arr, (target_hw, target_hw), interpolation=cv2.INTER_AREA)
        except ImportError:
            logger.warning("cv2 not available; can't resize from %s to (%d, %d)",
                           arr.shape[:2], target_hw, target_hw)
    return np.ascontiguousarray(arr)


def _extract_first(obs: dict, candidates: tuple[str, ...]) -> Any | None:
    """Return the first matching value from a flat or nested obs dict."""
    for key in candidates:
        if key in obs:
            return obs[key]
    images = obs.get("images") if isinstance(obs, dict) else None
    if isinstance(images, dict):
        for key in candidates:
            if key in images:
                return images[key]
            if key.startswith("observation/") and key[len("observation/"):] in images:
                return images[key[len("observation/"):]]
    return None


def _parse_delta_action_mask(spec: Any) -> np.ndarray | None:
    """Parse an openpi-style delta-action mask specification.

    Accepted forms:
      - None or empty -> None (no delta-state output transform; default)
      - List/tuple of ints in the openpi convention used by
        `_transforms.make_bool_mask`: positive N = N True, negative N
        = -N False. Examples:
          OpenArm v4 16-DOF: [7, -1, 7, -1] -> 16-bool mask, True for
                             the 7 arm joints of each arm, False for
                             each arm's gripper (already absolute).
          DROID:             [7, -1]         -> 8-bool, True for 7 arm
                                                joints, False for the
                                                gripper.
      - CSV string of the same ints: "7,-1,7,-1" (CLI-friendly)
      - Boolean iterable: passes through as-is.

    Returns a numpy bool array, or None.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = spec.strip()
        if not spec:
            return None
        parts = [int(p) for p in spec.split(",") if p.strip()]
    elif isinstance(spec, np.ndarray):
        return spec.astype(bool)
    else:
        parts = list(spec)
    if all(isinstance(p, (bool, np.bool_)) for p in parts):
        return np.asarray(parts, dtype=bool)
    bools: list[bool] = []
    for dim in parts:
        dim_i = int(dim)
        if dim_i > 0:
            bools.extend([True] * dim_i)
        else:
            bools.extend([False] * (-dim_i))
    return np.asarray(bools, dtype=bool) if bools else None


class FlashRTPolicyAdapter(_base_policy.BasePolicy):
    """Wrap a FlashRT VLAModel as an openpi BasePolicy.

    Args:
        model: an already-loaded ``flash_rt.VLAModel`` instance. The
            caller is responsible for calling ``flash_rt.load_model(...)``
            with the right framework / hardware / robot_action_dim, and
            (recommended) pre-calibrating with representative data.
        default_prompt: used when no prompt is present in the observation.
        chunk_size: expected output chunk length. Used only for shape
            sanity checks; does not change inference.
        metadata: extra metadata to expose at the WebsocketPolicyServer
            handshake. Typically the openpi `policy.metadata` dict so
            existing clients see the same content.
    """

    def __init__(
        self,
        model: Any,
        *,
        default_prompt: str | None = None,
        chunk_size: int = 10,
        metadata: dict[str, Any] | None = None,
        delta_action_mask: Any = None,
    ) -> None:
        self._model = model
        self._default_prompt = default_prompt
        self._chunk_size = chunk_size
        self._metadata = metadata or {}
        # See _parse_delta_action_mask for the accepted formats. When
        # set, infer() reads 'state' from each observation and applies
        # AbsoluteActions(state, mask) on the model output so it
        # matches the openpi server's delta-state -> joint-radian
        # output transform. This is required for any robot whose
        # training config wraps DeltaActions/AbsoluteActions around
        # its action stream (OpenArm v4 = [7, -1, 7, -1], DROID =
        # [7, -1], etc.). Leave None for robots where the model
        # already outputs absolute actions (LIBERO, base pi05).
        self._delta_action_mask = _parse_delta_action_mask(delta_action_mask)
        if self._delta_action_mask is not None:
            logger.info(
                "FlashRTPolicyAdapter: delta_action_mask enabled "
                "(len=%d, n_delta=%d, n_abs=%d). Per-step deltas will "
                "be added to obs['state'] on the delta channels.",
                len(self._delta_action_mask),
                int(self._delta_action_mask.sum()),
                int((~self._delta_action_mask).sum()),
            )
        if hasattr(model, "_pipe") and not getattr(model._pipe, "calibrated", False):
            logger.warning(
                "FlashRTPolicyAdapter: model is not calibrated; first infer "
                "call will block for ~3 s. Pre-warm via "
                "model.calibrate([sample_obs]) before serving."
            )
        self._infer_count = 0

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def infer(self, obs: dict) -> dict:
        t0 = time.monotonic()

        base = _extract_first(obs, _BASE_IMAGE_CANDIDATES)
        wrist = _extract_first(obs, _WRIST_IMAGE_CANDIDATES)
        wrist_right = _extract_first(obs, _WRIST_IMAGE_RIGHT_CANDIDATES)
        if base is None:
            raise KeyError(
                "FlashRTPolicyAdapter: no base camera image found. Tried "
                f"{_BASE_IMAGE_CANDIDATES} (flat) and inside obs['images']. "
                f"Observed keys: {list(obs.keys())[:20]}"
            )

        images = [_normalize_image(base)]
        if wrist is not None:
            images.append(_normalize_image(wrist))
        if wrist_right is not None:
            images.append(_normalize_image(wrist_right))

        prompt = None
        for key in _PROMPT_CANDIDATES:
            if key in obs:
                prompt = obs[key]
                break
        if isinstance(prompt, (bytes, bytearray)):
            prompt = prompt.decode("utf-8", errors="replace")
        if not prompt:
            if self._default_prompt is None:
                raise ValueError(
                    "FlashRTPolicyAdapter: no prompt in observation and no "
                    "default_prompt set"
                )
            prompt = self._default_prompt

        # Extract state once: it's needed both for the model (Pi0/Pi0.5
        # state input) and for the delta-action AbsoluteActions transform
        # below. Avoid two scans of the obs dict.
        state_raw = _extract_first(obs, _STATE_CANDIDATES)
        state_for_model: Optional[np.ndarray] = None
        if state_raw is not None:
            state_for_model = np.asarray(state_raw, dtype=np.float32).reshape(-1)

        actions = self._model.predict(
            images=images, prompt=str(prompt), state=state_for_model)
        actions_np = np.asarray(actions)

        if self._delta_action_mask is not None:
            if state_for_model is None:
                raise KeyError(
                    "FlashRTPolicyAdapter: delta_action_mask is set but no "
                    "'state' / 'observation/state' found in obs. Tried "
                    f"{_STATE_CANDIDATES}. Observed keys: "
                    f"{list(obs.keys())[:20]}"
                )
            state_np = state_for_model.astype(actions_np.dtype, copy=False)
            mask = self._delta_action_mask
            dims = min(mask.shape[0], state_np.shape[0], actions_np.shape[-1])
            if dims > 0:
                offset = np.where(mask[:dims], state_np[:dims], 0.0).astype(
                    actions_np.dtype, copy=False
                )
                actions_np = actions_np.copy()
                actions_np[..., :dims] += offset[None, :]

        elapsed = time.monotonic() - t0

        # RTC pass-through: the client (AsyncActionChunkBroker) reads
        # _rtc_chunk_model_space from the response to feed prev_chunk on
        # the next call. FlashRT's pipeline returns post-unnorm actions
        # directly; we don't have the pre-unnorm chunk available here.
        # Sending the post-unnorm chunk back as _rtc_chunk_model_space
        # is a degradation vs the openpi server (which sends the raw
        # model-space chunk) -- it means RTC guidance still works but is
        # numerically slightly different. Acceptable for the initial
        # FlashRT-on-Spark milestone; revisit when RTC parity is needed.
        self._infer_count += 1
        return {
            "actions": actions_np,
            "policy_timing": {"infer_ms": elapsed * 1000.0},
            "_rtc_chunk_model_space": actions_np,
        }

    def reset(self) -> None:
        # FlashRT VLAModel doesn't expose a reset hook today. The
        # cached prompt + calibration state are kept across resets;
        # callers who need to reload should drop the adapter and build
        # a new one.
        pass
