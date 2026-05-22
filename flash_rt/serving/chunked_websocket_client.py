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

    1: sync truncate-replan k=5  - block per chunk, replan every 5 actions.
                                    Robot visibly hitches at each replan.
                                    Useful as an A/B baseline; NOT
                                    production.
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
    5: async fire-ASAP, splice at d, server-side RTC prefix-freeze
                                  - the model's diffusion decoder is
                                    constrained to keep the new chunk's
                                    first ``d`` model-space actions equal
                                    to the inflight prefix (Black et al.
                                    2025 §3.2 "hard inpainting"). The
                                    remaining suffix is denoised under
                                    that constraint, so it stays
                                    continuous with the prefix instead
                                    of being a plan-from-scratch.
                                    Client-side seam blend is therefore
                                    set to 0 (the server already does
                                    the smoothing). REQUIRES a backend
                                    that honours ``_rtc_prev_chunk`` +
                                    ``_rtc_inference_delay`` in the obs
                                    dict — FlashRT Pi0.5 RTX does;
                                    older openpi-JAX and FlashRT Thor
                                    silently fall back to mode-2
                                    behaviour.

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

VALID_MODES = (1, 2, 3, 4, 5)

DEFAULT_MODE = 3

MODE_DESCRIPTIONS = {
    1: "sync truncate-replan k=5 (baseline)",
    2: "async fire-ASAP, splice at d, no seam blend",
    3: "async fire-ASAP, splice at d, seam blend = 3 (default)",
    4: "async fire-ASAP, splice at d, seam blend = 5",
    5: "async fire-ASAP, splice at d, server-side prefix-freeze",
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
        # Sync truncate-replan k=5: action_horizon=5 forces a swap every
        # 5 steps; start_next_at=5 means "submit the next chunk request
        # at chunk exhaustion" — combined with miss_policy="block" this
        # is effectively synchronous. Kept as the A/B baseline for
        # smoothness comparisons against the async modes; matches the
        # canonical LIBERO eval pattern in examples/thor/eval_libero.py.
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=5,
            start_next_at=5,
            miss_policy="block",
            blend_steps=0)
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
        # Server-side prefix-freeze handles the smoothness contract.
        # blend_steps=0 because client-side blending now hides the
        # very feature we'd otherwise observe in motion smoothness
        # numbers (the prefix-freeze is the smoothing).
        return RTCConfig(
            **common,
            blend_steps=0,
            enable_prefix_freeze=True)
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
    ) -> None:
        if blending_mode not in VALID_MODES:
            raise ValueError(
                f"blending_mode must be in {VALID_MODES}, got {blending_mode}")
        self._policy = policy
        self._action_output_key = action_output_key
        self._target_hz = float(target_hz)
        self._expected_latency_ms = float(expected_latency_ms)
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
            "start_next_at=%s, splice_d_seed=%s, blend_steps=%d",
            self._blending_mode, MODE_DESCRIPTIONS[self._blending_mode],
            self._chunk_len, self._target_hz, self._expected_latency_ms,
            cfg.start_next_at, cfg.inference_delay_steps, cfg.blend_steps)

    def _build_runner(self, mode: int) -> AsyncChunkRunner:
        cfg = _build_config_for_mode(
            mode, self._chunk_len, self._target_hz,
            expected_latency_ms=self._expected_latency_ms)
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
