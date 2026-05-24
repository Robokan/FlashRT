"""Client-side chunk consumption with selectable blending modes.

Wraps a ``WebsocketClientPolicy`` (or any ``BasePolicy``-compatible
policy) with FlashRT's :class:`~flash_rt.runtime.AsyncChunkRunner`,
offering four named blending modes that match the convention in
``examples/libero_playground.py`` keys 1..4. Backend-agnostic by
construction: the same wrapper works against openpi-JAX (port 8000)
and FlashRT (port 8002) — the blending is on the *consumer* side,
so a Phase 6 backend comparison can swap the server URL without
changing any blending characteristics.

Modes
-----
Mode 1 is a sync baseline for A/B comparison. Modes 2–4 implement the
RTC-paper-faithful async path (Black et al. 2025, arXiv:2506.07339):
fire the next inference as soon as the previous one completes, splice
the freshly-arrived chunk at index ``d`` (= measured inference latency
in control ticks), and serve seam-blend on the first ``blend_steps``
actions of each new chunk to absorb the discontinuity at the swap.

::

    1: sync truncate-replan k=25, seam blend = 5
                                  - block per chunk, but play only the
                                    first 25 actions (= 0.5 s at 50 Hz)
                                    of each chunk before re-inferring;
                                    remainder of the old chunk is
                                    discarded. Matches the historical
                                    openpi ``ActionChunkBroker`` (ALOHA
                                    default) that is known to produce
                                    good task performance on pi0.5
                                    chocolate-bars. The first 5 actions
                                    of each new chunk are linearly
                                    blended from the last action served
                                    out of the previous chunk; without
                                    this the gripper (absolute action
                                    space) can step-jump > 1 rad at
                                    chunk boundaries and trip SparkJAX
                                    safety. ``prefix_freeze`` (default
                                    False at SparkJAX) is available for
                                    A/B comparison.
    2: async, fire ASAP, splice at d, no seam blend (raw)
                                  - background inference; freshest plan
                                    always served. Hard step at each
                                    seam. Default for debugging.
    3: async, fire ASAP, splice at d, seam blend = 3        DEFAULT.
                                  - linearly ramps the first 3 actions
                                    of each new chunk from
                                    ``last_served_action`` toward the
                                    raw new-chunk action. Smooth seam,
                                    minimal latency cost.
    4: async, fire ASAP, splice at d, seam blend = 5
                                  - 5-step seam ramp. Smoother but
                                    delays convergence to the new
                                    chunk by ~100 ms at 50 Hz.
    5: async fire-ASAP, splice at d, server-side RTC soft-guidance
                                  - the model's diffusion decoder nudges
                                    its velocity field per Euler step
                                    toward making the predicted denoised
                                    endpoint match the inflight prefix,
                                    weighted by a time-decay schedule
                                    (Black et al. 2025 §3.3 "soft
                                    inpainting"; lerobot pi05_base ships
                                    this as ``RTCProcessor``). The
                                    constraint is INTEGRATED into the
                                    trajectory rather than clobbered on
                                    top of it, so the chunk boundary is
                                    smooth by construction — no client-
                                    side seam blend needed. REQUIRES a
                                    backend that honours
                                    ``_rtc_prev_chunk`` +
                                    ``_rtc_inference_delay`` in the obs
                                    dict — FlashRT Pi0.5 RTX (G11+) does;
                                    older openpi-JAX and FlashRT Thor
                                    silently fall back to mode-2
                                    behaviour. Replaced the G10 hard-
                                    freeze inpainting which clobbered
                                    the noise tensor at prefix positions
                                    and zeroed the per-step velocity
                                    update there (still produced jerky
                                    splices on outlier-latency chunks).

``d`` is computed PER PROMOTION from the latency of the inference
just completed: ``d = ceil(this_call_latency_s * target_hz)``. Per-
call (not EMA-smoothed) because variable latency — pipeline
rebuilds on FlashRT Pi0.5 + state-in-prompt, network jitter,
contention — makes an EMA underestimate a fresh long outlier and
lag a fresh short recovery, both of which produce a time-skip jump
at the splice. At 50 Hz with FlashRT's ~140 ms steady-state on
Spark this resolves to ``d = 7``; at 25 Hz it's ``d = 4``. The
``expected_latency_ms`` constructor arg only seeds the very first
promotion before any latency has been measured; from the second
promotion onward, per-call wins.

The chunk length ``H`` is auto-detected from the server's metadata
(``chunk_size`` field) on first call; override via constructor for
servers that don't publish it (vanilla openpi-JAX < 2026-05 returns
empty metadata; we fall back to 50, the Pi0.5 default).

Usage::

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from flash_rt.serving.chunked_websocket_client import ChunkedWebsocketClient

    policy = WebsocketClientPolicy(host="localhost", port=8002)
    client = ChunkedWebsocketClient(policy, blending_mode=3, target_hz=50.0)

    for step in range(N):
        obs = build_obs_from_robot()       # dict with state, images, prompt
        action = client.next_action(obs)   # shape (action_dim,)
        send_to_robot(action)

    # Runtime mode change (e.g. from a ROS service):
    client.set_blending_mode(4)
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from flash_rt.runtime import (
    AsyncChunkRunner,
    CallablePolicyAdapter,
    RTCConfig,
    RTCStats,
)


logger = logging.getLogger(__name__)


_PI05_FALLBACK_CHUNK_LEN = 50

# Mode 1's "execute horizon": how many actions of each freshly-received
# chunk we actually serve before blocking on the next inference. This
# replicates the historical openpi ``ActionChunkBroker`` default
# (``action_horizon=25``, commented "ALOHA default") that SparkJAX used
# successfully on the pi0.5 chocolate-bars checkpoint before we switched
# to the FlashRT-side client wrapper. Empirically that 25-tick (= 0.5 s
# at 50 Hz) re-plan cadence is the sweet spot for this checkpoint:
# long enough that intra-chunk motion is coherent, short enough that
# the model never commits the robot to a stale full-chunk plan whose
# tail (grasp / lift / place) was forecast from out-of-date pixels.
# Playing the full 50-action chunk back-to-back, even with prefix-
# freeze, regressed chocolate-bar grasping noticeably; this constant
# restores the baseline. Clamped per-call to the actual chunk length
# so smaller-chunk servers (e.g. chunk_size=10) still work sensibly.
_MODE1_EXECUTE_HORIZON = 25

# Mode 1's "seam blend window": how many actions of each freshly-
# received chunk are linearly blended from the last-served previous
# action toward the raw new-chunk action. Without seam blending the
# boundary between chunk A and chunk B is a hard step whose magnitude
# is whatever the model commanded in the inference gap — which on the
# OpenArm chocolate_bars checkpoint can be ~2 rad on the absolute-
# space gripper joints when the model transitions phase (approach
# → grasp), tripping per-step joint safety limits. Five ticks at
# 50 Hz (= 100 ms) splits a 2 rad command into 0.4 rad/step, well
# under SparkJAX's 1.0 rad/step limit, while still letting the
# gripper close fast enough to grasp. Clamped to ``execute_horizon``
# so smaller-chunk servers degrade safely.
_MODE1_BLEND_STEPS = 5

# Mode 5 RTC soft-guidance execution horizon (Phase 6 / G11). Number
# of new-chunk positions across which the server's velocity field is
# nudged toward continuity with the inflight prefix. lerobot's default
# is 10. Larger values give the model a longer merge window (smoother
# transitions) at the cost of constraining more of the trajectory.
#
# The shape of the merge: positions [0, d_pred) are weight-1.0 anchor
# (model is strongly guided to match the inflight prefix exactly),
# positions [d_pred, execution_horizon) ramp the weight 1→0, positions
# [execution_horizon, chunk_size) are weight-0 free continuation.
# Tuning lever: if motion is jerky at chunk swaps, increase to ~15–20.
# If task performance regresses (model committing too hard to the past
# plan), decrease toward d_pred + 1.
_MODE5_EXECUTION_HORIZON = 10

# Mode 5 RTC soft-guidance schedule (Phase 6 / G11). Controls the ramp
# shape in the merge window [d_pred, execution_horizon):
#   - "linear": straight line 1→0 (lerobot default, less aggressive
#     falloff so the model fights the prefix more in the middle of the
#     window).
#   - "exp":    e^x-shaped, sharper falloff so the merge weight drops
#     fast once past the anchor (the public lerobot docs example).
# We default to "linear" to match lerobot's RTCConfig and the kinetix
# canonical evaluation. Switch to "exp" if linear blending visibly
# steers the trajectory off task in the middle of the merge window.
_MODE5_SCHEDULE = "linear"

VALID_MODES = (1, 2, 3, 4, 5)

DEFAULT_MODE = 3

MODE_DESCRIPTIONS = {
    1: "sync truncate-replan k=25, seam blend = 5 (ALOHA-default; freeze optional)",
    2: "async fire-ASAP, splice at d, no seam blend",
    3: "async fire-ASAP, splice at d, seam blend = 3 (default)",
    4: "async fire-ASAP, splice at d, seam blend = 5",
    5: "async fire-ASAP, splice at d, server-side RTC soft-guidance "
       "(execution_horizon=10, linear schedule), miss=block",
}


def _resolve_chunk_len(
    policy: Any, override: Optional[int], logger_obj: logging.Logger
) -> int:
    """Decide the chunk length H to use.

    Resolution order: explicit override > server metadata > 50 fallback.
    Mirrors ``sparkjax.teleop.openpi_runner_node._resolve_chunk_len`` so
    a future SparkJAX-side integration sees identical behaviour.
    """
    if override is not None and int(override) > 0:
        cl = int(override)
        logger_obj.info("chunk_len=%d (explicit override)", cl)
        return cl
    if hasattr(policy, "get_server_metadata"):
        try:
            meta = policy.get_server_metadata() or {}
        except Exception as e:  # noqa: BLE001
            logger_obj.warning(
                "get_server_metadata() failed (%r); falling back to %d",
                e, _PI05_FALLBACK_CHUNK_LEN)
            return _PI05_FALLBACK_CHUNK_LEN
        if isinstance(meta, dict) and meta.get("chunk_size"):
            cl = int(meta["chunk_size"])
            logger_obj.info("chunk_len=%d (auto-detected from server metadata %r)",
                            cl, meta)
            return cl
    logger_obj.warning(
        "server metadata has no chunk_size; falling back to %d",
        _PI05_FALLBACK_CHUNK_LEN)
    return _PI05_FALLBACK_CHUNK_LEN


def _initial_inference_delay_steps(
    target_hz: float, expected_latency_ms: float
) -> int:
    """Convert an a-priori latency estimate to control ticks.

    Used to seed the ``inference_delay_steps`` config field as a
    one-shot fallback for the very first promotion, before any real
    latency has been observed. From the second promotion onward,
    ``auto_inference_delay=True`` causes the runner to use the
    per-call measured latency for ``d``; this seed never fires again.

    With Spark's ~140 ms FlashRT round-trip and 50 Hz control this
    gives ``d=7``. At 25 Hz it's ``d=4``.
    """
    return max(0, int(np.ceil((expected_latency_ms / 1000.0) * target_hz)))


def _build_config_for_mode(
    mode: int,
    chunk_len: int,
    target_hz: float,
    *,
    expected_latency_ms: float,
    prefix_freeze: bool = True,
) -> RTCConfig:
    """Translate a blending mode (1..4) into an RTCConfig.

    See module docstring for the mode catalogue. The mapping is the
    single source of truth — mode 5 (server-side RTC inpainting) lands
    here when its server protocol is implemented.

    Async modes (2/3/4) use:
      * ``start_next_at=0`` — fire the next inference as soon as the
        previous one completes (the RTC paper's intent).
      * ``auto_inference_delay=True`` — splice index ``d`` is recomputed
        per promotion from the latency of THAT specific inference, so
        the first ``d`` actions of each freshly-arrived chunk (which
        correspond to control ticks that already elapsed serving the
        OLD chunk) are skipped. Per-call (not EMA) so a single long
        outlier (FlashRT pipeline rebuild, network jitter) is spliced
        at the right d for that one chunk, not at a stale EMA value.
      * ``inference_delay_steps`` is seeded from ``expected_latency_ms``
        as a one-shot fallback for the very first promotion only.
    """
    if mode == 1:
        # Sync truncate-replan baseline. Play the first
        # ``_MODE1_EXECUTE_HORIZON`` (=25) actions of each chunk, then
        # synchronously block on a fresh inference; remaining actions
        # of the old chunk are discarded. This matches the historical
        # openpi ``ActionChunkBroker`` (action_horizon=25, "ALOHA
        # default") that SparkJAX used successfully on pi0.5
        # chocolate-bars before we switched the client to this
        # wrapper, and which the operator confirms produces noticeably
        # better task performance than playing the full 50-action
        # chunk back-to-back (the latter commits the robot to a stale
        # forecast of the manipulation phase before any fresh visual
        # observation can correct it).
        #
        # ``blend_steps=_MODE1_BLEND_STEPS`` (=5) ramps the first 5
        # actions of each new chunk linearly from the last action
        # served out of the previous chunk toward the raw new-chunk
        # action. Without it the seam between A[execute_horizon-1] and
        # B[0] is a hard step whose magnitude is whatever the model
        # decided to do between the two inferences. For the
        # pi05_openarm_ngc_lora_v4 / chocolate_bars checkpoint the
        # gripper is in absolute (not delta) action space and the
        # model legitimately commands large step-changes there when
        # it transitions from "approach" to "grasp" (observed ~2 rad
        # gripper command jump from -2.205 → -0.217 at the very first
        # chunk boundary, tripping SparkJAX's per-step 1.0 rad safety
        # limit). Splitting that 2 rad command across 5 ticks (= 0.4
        # rad/tick at 50 Hz = 20 rad/s peak) keeps the seam under the
        # safety limit while still letting the gripper close in 100
        # ms, which is fast enough to grasp.
        #
        # ``prefix_freeze`` is honoured but defaults False at the
        # SparkJAX caller for this mode — the seam blend above already
        # smooths the boundary; the prefix-freeze inpainting path is
        # tested at mode 5. Set to True to A/B compare; when on, the
        # server constrains the new chunk's first d positions to match
        # the last d actions of the previous chunk and ``_idx`` is set
        # to d after the splice so we skip the constrained prefix.
        #
        # The execute horizon is clamped to the negotiated chunk
        # length so smaller-chunk servers (e.g. chunk_size=10) degrade
        # to "play the whole chunk" rather than running off the end.
        # Blend steps are also clamped so they can't exceed the
        # execute horizon (defensive — a 25-tick window with a 5-tick
        # blend is comfortable; if execute_horizon ever shrinks below
        # blend_steps the blend would overrun the chunk).
        execute_horizon = min(_MODE1_EXECUTE_HORIZON, chunk_len)
        blend_steps = min(_MODE1_BLEND_STEPS, execute_horizon)
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=execute_horizon,
            start_next_at=execute_horizon,
            miss_policy="block",
            blend_steps=blend_steps,
            auto_inference_delay=prefix_freeze,
            enable_prefix_freeze=prefix_freeze)
    d_seed = _initial_inference_delay_steps(target_hz, expected_latency_ms)
    common = dict(
        target_hz=target_hz,
        action_horizon=chunk_len,
        start_next_at=0,
        miss_policy="hold_last",
        inference_delay_steps=d_seed,
        auto_inference_delay=True,
    )
    if mode == 2:
        return RTCConfig(**common, blend_steps=0)
    if mode == 3:
        return RTCConfig(**common, blend_steps=3)
    if mode == 4:
        return RTCConfig(**common, blend_steps=5)
    if mode == 5:
        # Mode 5 (Phase 6 / G11): **soft-guidance RTC**. The server's
        # diffusion decoder nudges the velocity field per Euler step
        # toward making the predicted denoised endpoint match the
        # inflight prefix, weighted by a time-decay schedule
        # (``_MODE5_EXECUTION_HORIZON`` + ``_MODE5_SCHEDULE``). The
        # model integrates the constraint into a coherent trajectory
        # rather than fighting fixed positions — so chunk boundaries
        # come out smooth even when the splice lands past the anchor
        # region.
        #
        # G11 replaces the G10 hard-freeze inpainting (which clobbered
        # the noise tensor at prefix positions and zeroed the per-step
        # velocity update there). Hard-freeze stopped the SparkJAX
        # safety stops but motion was still jerky at chunk boundaries
        # because the free continuation past the frozen region had no
        # continuity guarantee against the anchor — empirically a
        # 1.4 rad joint jump on outlier-latency chunks. Soft guidance
        # subsumes the seam blend and the cap workaround we layered
        # on top of hard-freeze:
        #
        #   - ``blend_steps=0``: no client-side seam interpolation
        #     needed. The merge is done IN THE MODEL by the soft-
        #     guidance kernel, which produces a trajectory that's
        #     smooth at the splice by construction rather than
        #     smooth-by-interpolation after the fact.
        #
        #   - ``prefix_freeze_max_steps=None``: the G10 cap=horizon/4
        #     was a workaround for hard-freeze's EMA-poisoned d_pred
        #     pinning under pipeline-rebuild storms. With soft
        #     guidance, oversizing d_pred is harmless — the merge-
        #     window weights ramp to 0 past ``execution_horizon`` and
        #     the unused tail of d_pred is just a wider free region.
        #     The bottleneck shifts from "splice cliff" to "is there
        #     a fresh chunk ready" (which ``miss_policy=block``
        #     handles).
        #
        # ``miss_policy="block"`` is retained from G10. On Spark with
        # Pi0.5 state-in-prompt, each new ``prompt_len`` triggers a
        # ~750 ms lazy graph capture on first touch; if a capture
        # outlier exceeds the play window between chunks, ``block``
        # waits for the inflight inference instead of repeating the
        # last action and tripping SparkJAX's 25-consecutive-holds
        # safety abort.
        mode5_kwargs = dict(common)
        mode5_kwargs["miss_policy"] = "block"
        return RTCConfig(
            **mode5_kwargs,
            blend_steps=0,
            enable_prefix_freeze=True,
            prefix_freeze_max_steps=None,
            execution_horizon=_MODE5_EXECUTION_HORIZON,
            rtc_schedule=_MODE5_SCHEDULE)
    raise ValueError(
        f"blending_mode must be in {VALID_MODES}, got {mode}")


class ChunkedWebsocketClient:
    """Wrap a base policy with a mode-selectable chunk consumer.

    The wrapper owns the AsyncChunkRunner and forwards observations
    through it. ``next_action`` is the only hot-path call; everything
    else is config plumbing.

    Args:
        policy: A ``BasePolicy``-compatible object whose ``infer(obs)``
            returns a dict containing ``actions`` of shape
            ``(H, action_dim)``. Typically a
            ``openpi_client.WebsocketClientPolicy``.
        blending_mode: One of {1, 2, 3, 4}; see module docstring.
            Default ``3`` (async fire-ASAP, splice at ``d``, seam blend = 3).
            Production-safe.
        target_hz: Controller rate the robot loop runs at. Used by
            ``AsyncChunkRunner`` to convert measured latency seconds
            to splice ticks ``d`` for the async modes. The consumer
            drives actual timing.
        chunk_len_override: If set, skip server metadata and use this
            value as H. Useful when talking to a server that doesn't
            publish ``chunk_size`` and the default 50 is wrong.
        expected_latency_ms: Expected per-call round-trip inference
            latency in ms. Used only to seed ``inference_delay_steps``
            for the FIRST swap (before the runner's latency EMA has
            any samples). After the first completed inference, the
            EMA takes over via ``auto_inference_delay=True``. The
            default 200 ms is conservative and works across both
            FlashRT (~165 ms steady-state on Spark) and openpi-JAX
            (~175 ms steady). Mode 1 (sync) ignores this entirely.
        action_output_key: Key in the server response that holds the
            action chunk. Defaults to ``"actions"`` (openpi / FlashRT
            convention).
        prefix_freeze: Enable server-side RTC prefix-freeze inpainting
            on mode 1 (sync full-chunk). Default True. When enabled
            the runner sends the last ``d`` actions of the just-
            completed chunk as ``_rtc_prev_chunk`` and the server
            constrains the new chunk's first ``d`` positions to match,
            closing the inter-chunk boundary discontinuity. Set to
            False to A/B compare against the baseline behaviour.
            Mode 5 always enables prefix-freeze (it's the defining
            feature of that mode); modes 2/3/4 ignore this flag.
    """

    def __init__(
        self,
        policy: Any,
        *,
        blending_mode: int = DEFAULT_MODE,
        target_hz: float = 25.0,
        chunk_len_override: Optional[int] = None,
        expected_latency_ms: float = 200.0,
        action_output_key: str = "actions",
        prefix_freeze: bool = True,
    ) -> None:
        if blending_mode not in VALID_MODES:
            raise ValueError(
                f"blending_mode must be in {VALID_MODES}, got {blending_mode}")
        self._policy = policy
        self._action_output_key = action_output_key
        self._target_hz = float(target_hz)
        self._expected_latency_ms = float(expected_latency_ms)
        self._prefix_freeze = bool(prefix_freeze)
        self._chunk_len = _resolve_chunk_len(policy, chunk_len_override, logger)
        self._blending_mode = int(blending_mode)
        # ``meta_keys`` lifts the server's ``_rtc_chunk_model_space``
        # (the true normalised model-space chunk) into ChunkResult so
        # that mode 5 / ``enable_prefix_freeze`` can feed it back via
        # ``_rtc_prev_chunk`` on the next inference. Cheap to ask for
        # in all modes — if the server doesn't supply it, the lift is
        # a no-op and the field stays None.
        self._adapter = CallablePolicyAdapter(
            fn=policy.infer,
            output_key=action_output_key,
            meta_keys=("_rtc_chunk_model_space",))
        self._runner = self._build_runner(self._blending_mode)
        cfg = self._runner.config
        logger.info(
            "ChunkedWebsocketClient ready: mode=%d (%s), chunk_len=%d, "
            "target_hz=%.1f, expected_latency_ms=%.0f, "
            "start_next_at=%s, splice_d_seed=%s, blend_steps=%d, "
            "prefix_freeze=%s (effective=%s)",
            self._blending_mode, MODE_DESCRIPTIONS[self._blending_mode],
            self._chunk_len, self._target_hz, self._expected_latency_ms,
            cfg.start_next_at, cfg.inference_delay_steps, cfg.blend_steps,
            self._prefix_freeze, cfg.enable_prefix_freeze)

    def _build_runner(self, mode: int) -> AsyncChunkRunner:
        cfg = _build_config_for_mode(
            mode, self._chunk_len, self._target_hz,
            expected_latency_ms=self._expected_latency_ms,
            prefix_freeze=self._prefix_freeze)
        return AsyncChunkRunner(self._adapter, cfg)

    @property
    def blending_mode(self) -> int:
        return self._blending_mode

    @property
    def chunk_len(self) -> int:
        return self._chunk_len

    @property
    def stats(self) -> RTCStats:
        return self._runner.stats

    def next_action(self, obs: dict, *, block_if_empty: bool = True) -> np.ndarray:
        """Get the next action for the current observation.

        First call blocks for an initial chunk (synchronous inference).
        Later calls return immediately from the current chunk; the
        runner kicks off background inference at the configured
        ``start_next_at`` boundary.
        """
        return self._runner.next_action(obs, block_if_empty=block_if_empty)

    def set_blending_mode(self, mode: int) -> None:
        """Switch blending modes at runtime.

        Tears down the current AsyncChunkRunner and builds a new one
        with the new config. Any chunk currently in flight is dropped;
        the next ``next_action`` call will block on a fresh inference
        (~one chunk's worth of latency). Safe to call mid-loop but the
        caller should expect a hiccup.

        Idempotent: switching to the current mode is a no-op.
        """
        if mode not in VALID_MODES:
            raise ValueError(
                f"blending_mode must be in {VALID_MODES}, got {mode}")
        if mode == self._blending_mode:
            logger.debug("set_blending_mode(%d): no-op (already active)", mode)
            return
        logger.info(
            "Switching blending_mode: %d (%s) -> %d (%s)",
            self._blending_mode, MODE_DESCRIPTIONS[self._blending_mode],
            mode, MODE_DESCRIPTIONS[mode])
        old_runner = self._runner
        try:
            # wait=True so any in-flight background inference completes
            # and releases the shared websocket before we hand the
            # connection to a freshly-built runner. Without this, the
            # new runner's first ``infer`` collides with the old
            # runner's pending ``recv`` and websockets raises
            # ConcurrencyError ("cannot call recv while another thread
            # is already running recv").
            old_runner.close(wait=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to close old runner cleanly: %r", e)
        self._blending_mode = int(mode)
        self._runner = self._build_runner(self._blending_mode)

    def reset(self, obs: dict) -> None:
        """Re-initialize the runner with a fresh chunk for ``obs``.

        Use when the prompt or task changes and the cached chunk
        should not be served any further AND you have a fresh
        observation to seed the next chunk with. Blocks on one
        inference.
        """
        self._runner.reset(obs)

    def clear(self) -> None:
        """Discard the current chunk + cancel any in-flight inference,
        without firing a new one.

        The next ``next_action`` call will synchronously block on a
        fresh inference using the obs it's called with. Use when you
        need to invalidate cached chunks (e.g. prompt changed) but
        don't have a current observation handy at the clear-point.

        Implemented as a runner tear-down + rebuild (same code path as
        ``set_blending_mode``); cheaper than ``reset`` because it
        skips the immediate inference.
        """
        old_runner = self._runner
        try:
            old_runner.close(wait=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("clear: failed to close old runner cleanly: %r", e)
        self._runner = self._build_runner(self._blending_mode)

    def close(self) -> None:
        """Shut down the background executor. Idempotent.

        Waits for any in-flight background inference to complete so
        the underlying websocket is in a quiesced state by the time
        this returns (callers that re-use the same policy for a new
        client otherwise hit websocket ConcurrencyError).
        """
        try:
            self._runner.close(wait=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to close runner cleanly: %r", e)

    def __enter__(self) -> "ChunkedWebsocketClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
