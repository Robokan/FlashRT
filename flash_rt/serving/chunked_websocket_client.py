"""Client-side chunk consumption with selectable blending modes.

Wraps a ``WebsocketClientPolicy`` (or any ``BasePolicy``-compatible
policy) with FlashRT's :class:`~flash_rt.runtime.AsyncChunkRunner`,
offering four named blending modes that match the convention in
``examples/libero_playground.py`` keys 1..4. Backend-agnostic by
construction: the same wrapper works against openpi-JAX (port 8000)
and FlashRT (port 8002) — the blending is on the *consumer* side,
so a Phase 6 backend comparison can swap the server URL without
changing any blending characteristics.

Modes (1..4 mirror ``libero_playground.py``)::

    1: sync truncate-replan k=5  - block per chunk, replan every 5 actions.
                                    Chunk length effectively becomes 5.
                                    Robot visibly hitches at each replan.
    2: async pipelined, no blend - background inference, hard chunk swap
                                    at horizon/2. DEFAULT.
    3: async + tail blend = 3    - same as 2 + 3-step end-of-chunk damping
                                    (linearly pulls last 3 actions of each
                                    chunk toward the last served action).
    4: async + tail blend = 5    - same with 5-step damping window.

Tail blend is end-of-chunk smoothing per ``flash_rt.runtime.rtc``'s
docstring, not cross-chunk seam smoothing. See ``docs/spark_status.md``
G6 for the full rationale.

The chunk length ``H`` is auto-detected from the server's metadata
(``chunk_size`` field) on first call; override via constructor for
servers that don't publish it (vanilla openpi-JAX < 2026-05 returns
empty metadata; we fall back to 50, the Pi0.5 default).

Usage::

    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from flash_rt.serving.chunked_websocket_client import ChunkedWebsocketClient

    policy = WebsocketClientPolicy(host="localhost", port=8002)
    client = ChunkedWebsocketClient(policy, blending_mode=2, target_hz=25.0)

    for step in range(N):
        obs = build_obs_from_robot()       # dict with state, images, prompt
        action = client.next_action(obs)   # shape (action_dim,)
        send_to_robot(action)

    # Runtime mode change (e.g. from a ROS service):
    client.set_blending_mode(3)
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

VALID_MODES = (1, 2, 3, 4)

MODE_DESCRIPTIONS = {
    1: "sync truncate-replan k=5",
    2: "async pipelined, no blend (default)",
    3: "async + tail blend = 3",
    4: "async + tail blend = 5",
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


def _latency_aware_start_next_at(
    horizon: int, target_hz: float, expected_latency_ms: float
) -> int:
    """Pick the kick-off action index so the next chunk arrives by exhaustion.

    Mirror of ``motus_rtc_lite._default_start_next_at``: with control
    rate ``f`` and inference latency ``L``, the chunk consumer needs
    ``ceil(L * f)`` control ticks of runway. We submit the next chunk
    at index ``horizon - delay_steps - 1`` so that the result is ready
    when the last action of the current chunk is served (and we still
    have one cushion tick for the runner's promote/lock pass).

    Clamped to ``[1, horizon - 1]``:
      * ``>= 1`` so we always observe one tick of state evolution
        before re-inferring (avoids re-inferring on the exact same
        observation the current chunk was conditioned on).
      * ``<= horizon - 1`` so we never set it past chunk end.

    Returns 1 if the model is slower than the chunk duration — there's
    no asymmetric scheduling that fixes a fundamental
    latency-vs-chunk-budget mismatch, but kicking off ASAP is still
    the best we can do.
    """
    delay_steps = max(1, int(np.ceil((expected_latency_ms / 1000.0) * target_hz)))
    return max(1, min(horizon - 1, horizon - delay_steps - 1))


def _build_config_for_mode(
    mode: int,
    chunk_len: int,
    target_hz: float,
    *,
    expected_latency_ms: float,
) -> RTCConfig:
    """Translate a blending mode (1..4) into an RTCConfig.

    See module docstring for the mode catalogue. The mapping is the
    single source of truth — any future mode (e.g. 5: cross-chunk
    new-chunk-head blend) lands here and nowhere else.

    ``expected_latency_ms`` sizes ``start_next_at`` for the async
    modes (2/3/4). At ``50 Hz`` with ``chunk_len=10`` and FlashRT's
    ~165 ms round-trip this gives ``start_next_at=1`` (kick off
    immediately after first action served), matching what the openpi
    ``AsyncActionChunkBroker`` achieves with ``inference_delay=9``.
    The previous mode-2 default (``horizon // 2 = 5``) only gave
    inference 100 ms of runway — guaranteed deadline miss every chunk.
    """
    if mode == 1:
        # Sync truncate-replan k=5: action_horizon=5 forces a swap every
        # 5 steps; start_next_at=5 means "submit the next chunk request
        # at chunk exhaustion" — combined with max_workers=1 (RTCConfig
        # enforces) this is effectively synchronous. The consumer blocks
        # on _handle_exhausted_locked while inference runs.
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=5,
            start_next_at=5,
            miss_policy="block",
            blend_steps=0)
    start_next_at = _latency_aware_start_next_at(
        chunk_len, target_hz, expected_latency_ms)
    if mode == 2:
        # Default: async pipelined, hard swap. start_next_at is
        # latency-aware (see helper above) so inference fires early
        # enough to land before the current chunk exhausts.
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=chunk_len,
            start_next_at=start_next_at,
            miss_policy="hold_last",
            blend_steps=0)
    if mode == 3:
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=chunk_len,
            start_next_at=start_next_at,
            miss_policy="hold_last",
            blend_steps=3)
    if mode == 4:
        return RTCConfig(
            target_hz=target_hz,
            action_horizon=chunk_len,
            start_next_at=start_next_at,
            miss_policy="hold_last",
            blend_steps=5)
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
            Default 2 (async pipelined, no blend). Production-safe.
        target_hz: Controller rate the robot loop runs at. Used by
            ``AsyncChunkRunner`` only for stats / period bookkeeping;
            the consumer drives actual timing.
        chunk_len_override: If set, skip server metadata and use this
            value as H. Useful when talking to a server that doesn't
            publish ``chunk_size`` and the default 50 is wrong.
        expected_latency_ms: Expected per-call round-trip inference
            latency in ms. Sizes ``start_next_at`` for async modes
            (2/3/4) so background inference fires early enough to
            land before chunk exhaustion. The default 200 ms is
            conservative and works across both FlashRT (~165 ms
            steady-state on Spark) and openpi-JAX (~175 ms steady).
            Set higher only if you observe deadline misses; setting
            it too high just wastes a few actions per chunk to
            premature swap (harmless, model sees fresh state). Mode 1
            (sync) ignores this. Has no effect when latency exceeds
            ``chunk_len / target_hz`` — at that point no asymmetric
            scheduling can save you and you'll get deadline misses
            regardless.
        action_output_key: Key in the server response that holds the
            action chunk. Defaults to ``"actions"`` (openpi / FlashRT
            convention).
    """

    def __init__(
        self,
        policy: Any,
        *,
        blending_mode: int = 2,
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
        self._adapter = CallablePolicyAdapter(
            fn=policy.infer, output_key=action_output_key)
        self._runner = self._build_runner(self._blending_mode)
        cfg = self._runner.config
        logger.info(
            "ChunkedWebsocketClient ready: mode=%d (%s), chunk_len=%d, "
            "target_hz=%.1f, expected_latency_ms=%.0f, start_next_at=%s",
            self._blending_mode, MODE_DESCRIPTIONS[self._blending_mode],
            self._chunk_len, self._target_hz, self._expected_latency_ms,
            cfg.start_next_at)

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
