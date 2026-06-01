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
        # Cached action quantile scale (2 / (q99 - q01)) for relative-action
        # prefix re-anchoring. Populated lazily on first use from the
        # frontend's norm_stats; None if unavailable (re-anchoring then
        # no-ops). See _reanchor_rtc_prefix.
        self._action_qscale: Optional[np.ndarray] = None
        self._action_qscale_loaded = False
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

    def _action_quantile_scale(self) -> Optional[np.ndarray]:
        """Return ``2 / (q99 - q01)`` for the action channels, or None.

        This is the derivative of the quantile normalization
        (``norm = (x - q01) / (q99 - q01) * 2 - 1``, see
        ``core/utils/actions.unnormalize_actions``) w.r.t. the physical
        action ``x``. It converts a physical delta on a channel into the
        equivalent shift in normalized ``[-1, 1]`` space, which is what
        relative-action prefix re-anchoring needs. Cached after the first
        successful read from the frontend's ``norm_stats``.
        """
        if self._action_qscale_loaded:
            return self._action_qscale
        self._action_qscale_loaded = True
        ns = getattr(getattr(self._model, "_pipe", None), "norm_stats", None)
        if not ns or "actions" not in ns:
            logger.warning(
                "FlashRTPolicyAdapter: norm_stats['actions'] unavailable; "
                "RTC relative-prefix re-anchoring disabled (guidance will "
                "use stale-frame deltas for delta-action policies).")
            return None
        a = ns["actions"]
        if "q01" not in a or "q99" not in a:
            return None
        q01 = np.asarray(a["q01"], dtype=np.float32).reshape(-1)
        q99 = np.asarray(a["q99"], dtype=np.float32).reshape(-1)
        self._action_qscale = 2.0 / (q99 - q01 + 1e-6)
        return self._action_qscale

    def _reanchor_rtc_prefix(
        self,
        prev: np.ndarray,
        ref_state: Any,
        cur_state: Optional[np.ndarray],
    ) -> np.ndarray:
        """Re-express a delta-action prefix relative to the current state.

        ``prev`` is the previous chunk's NORMALIZED model-space actions
        (the diffusion variable). For a delta-action policy each row is a
        per-step delta relative to ``ref_state`` (the state at the
        inference that produced it). The new inference's deltas are
        relative to ``cur_state``. For the guidance continuity target to
        live in the same frame the model is about to predict in, shift the
        prefix's delta channels by the physical state drift, mapped into
        normalized space:

            norm(delta + (ref - cur)) = norm(delta) + (ref - cur) * 2/(q99-q01)

        Only delta channels (``self._delta_action_mask``) are shifted;
        absolute channels (e.g. grippers) are left untouched. This mirrors
        lerobot's ``_reanchor_relative_rtc_prefix`` (to_relative_actions +
        re-normalize) done as a closed-form additive correction so it costs
        one broadcast-add instead of an un-norm / re-norm round trip.

        No-ops (returns ``prev`` unchanged) when this isn't a delta-action
        model, when the ref/current state is missing, or when norm_stats
        are unavailable.
        """
        if (self._delta_action_mask is None or ref_state is None
                or cur_state is None):
            return prev
        scale = self._action_quantile_scale()
        if scale is None:
            return prev
        prev = np.asarray(prev, dtype=np.float32)
        ref = np.asarray(ref_state, dtype=np.float32).reshape(-1)
        cur = np.asarray(cur_state, dtype=np.float32).reshape(-1)
        mask = self._delta_action_mask
        a_dim = prev.shape[-1]
        dims = min(mask.shape[0], ref.shape[0], cur.shape[0],
                   scale.shape[0], a_dim)
        if dims <= 0:
            return prev
        # Physical drift between the two anchor frames, normalized, applied
        # only on the delta channels (absolute channels get zero shift).
        shift = (ref[:dims] - cur[:dims]) * scale[:dims]
        corr = np.zeros(a_dim, dtype=np.float32)
        corr[:dims] = np.where(mask[:dims], shift, 0.0)
        return prev + corr[None, :]

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

        # RTC soft-guidance passthrough (Phase 6 / G11): when the client
        # (or AsyncChunkRunner) attaches the ``_rtc_*`` fields, forward
        # them to the pipeline so the diffusion decoder nudges its
        # velocity field toward continuity with the inflight prefix.
        # See ``Pi05Pipeline._rtc_apply_guidance`` for the algorithm
        # and ``Pi05TorchFrontendRtx._stage_rtc_inputs`` for the upload.
        #
        # Fields:
        #   _rtc_prev_chunk         (np.ndarray, (L, action_dim))
        #       Previous chunk's model-space actions starting at the
        #       splice position. L should be ≥ d_pred (anchor length)
        #       and ideally extend to cover the merge window past
        #       d_pred for soft guidance to apply meaningfully (G11+).
        #   _rtc_inference_delay    (int, d_pred)
        #       Number of leading positions hard-anchored to the prefix.
        #   _rtc_execution_horizon  (int, optional)
        #       End of merge window (positions [d_pred, end) ramp 1→0).
        #       Defaults to model frontend's _rtc_execution_horizon.
        #   _rtc_schedule           (str, optional)
        #       "linear" / "exp" / "ones" / "zeros". Defaults to model
        #       frontend's _rtc_schedule.
        extra_obs: Optional[dict[str, Any]] = None
        rtc_prev = obs.get("_rtc_prev_chunk")
        rtc_d = obs.get("_rtc_inference_delay")
        if rtc_prev is not None and rtc_d is not None:
            prev_chunk = np.asarray(rtc_prev, dtype=np.float32)
            # Relative-action re-anchoring: re-express the (delta) prefix
            # relative to THIS observation's state so the guidance
            # continuity target matches the frame the new chunk is
            # predicted in. No-op for absolute-action models or when the
            # client did not send a ref state. See _reanchor_rtc_prefix.
            rtc_ref_state = obs.get("_rtc_ref_state")
            prev_chunk_reanchored = self._reanchor_rtc_prefix(
                prev_chunk, rtc_ref_state, state_for_model)
            extra_obs = {
                "_rtc_prev_chunk": prev_chunk_reanchored,
                "_rtc_inference_delay": int(rtc_d),
            }
            # Optional per-call config: pass through only when set so
            # the frontend's defaults apply otherwise.
            for opt_key in ("_rtc_execution_horizon", "_rtc_schedule"):
                if opt_key in obs and obs[opt_key] is not None:
                    extra_obs[opt_key] = obs[opt_key]
            # Diagnostic log — fires on the first 5 inferences (so we
            # can confirm the soft-guidance wiring is alive without
            # needing a long run that may safety-stop before throttle
            # hits), then once per 50 thereafter. If mode 5 in SparkJAX
            # is producing safety-tripping chunk boundaries, the first
            # thing to check is whether this log is firing at all —
            # silence here means the client is not populating the
            # fields (server-side guidance is a no-op).
            if self._infer_count < 5 or self._infer_count % 50 == 0:
                eh = extra_obs.get("_rtc_execution_horizon", "frontend-default")
                sch = extra_obs.get("_rtc_schedule", "frontend-default")
                reanchor = (
                    "on"
                    if (rtc_ref_state is not None
                        and self._delta_action_mask is not None
                        and self._action_qscale is not None)
                    else "off")
                # Max normalized shift the re-anchoring applied (0 when off
                # or when the robot didn't move during the inference window).
                dmax = float(np.max(np.abs(prev_chunk_reanchored - prev_chunk))) \
                    if reanchor == "on" else 0.0
                logger.info(
                    "[RTC] _rtc_prev_chunk shape=%s d=%d exec_horizon=%s "
                    "sched=%s reanchor=%s max|Δnorm|=%.4f "
                    "(prev[0,:4]=%s prev[-1,:4]=%s)",
                    prev_chunk_reanchored.shape, int(rtc_d), eh, sch,
                    reanchor, dmax,
                    np.round(prev_chunk_reanchored[0, :4], 3).tolist(),
                    np.round(prev_chunk_reanchored[-1, :4], 3).tolist())

        result = self._model.predict(
            images=images, prompt=str(prompt), state=state_for_model,
            extra_obs=extra_obs, return_dict=True)
        actions_np = np.asarray(result["actions"])
        chunk_model_space = result.get("_rtc_chunk_model_space")

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

        # RTC pass-through: ``_rtc_chunk_model_space`` is the true
        # normalized model-space chunk (32 dims, ``[-1, 1]``) returned
        # by ``Pi05TorchFrontendRtx.infer``. Clients should cache this
        # and feed it back as ``_rtc_prev_chunk`` on the next inference
        # to activate the server-side prefix-freeze. If the underlying
        # frontend does not return that field (older Pi0/Thor paths
        # that haven't been updated), fall back to the post-unnorm
        # chunk — RTC guidance still works but is numerically less
        # faithful (mirrors the openpi-server behaviour).
        # Mirror diagnostic on the OUTBOUND side so we can tell whether
        # the model frontend is actually populating model-space chunks
        # (required for the prefix-freeze loop to close). If
        # chunk_model_space is None here we fell back to the
        # post-unnorm actions, which RTC can still use but is
        # numerically less faithful. Same throttle as the inbound log.
        if self._infer_count < 5 or self._infer_count % 50 == 0:
            cms_src = "model" if chunk_model_space is not None else "fallback(actions)"
            logger.info("[RTC] returning _rtc_chunk_model_space (source=%s)", cms_src)
        self._infer_count += 1
        if chunk_model_space is None:
            chunk_model_space = actions_np
        return {
            "actions": actions_np,
            "policy_timing": {"infer_ms": elapsed * 1000.0},
            "_rtc_chunk_model_space": np.asarray(chunk_model_space),
        }

    def reset(self) -> None:
        # FlashRT VLAModel doesn't expose a reset hook today. The
        # cached prompt + calibration state are kept across resets;
        # callers who need to reload should drop the adapter and build
        # a new one.
        pass
