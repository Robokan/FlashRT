"""Lightweight asynchronous execution for action chunk policies.

This module stays outside model frontends. A frontend only needs to expose a
callable that maps the latest observation to an action chunk. The runner
handles background chunk generation and foreground action consumption.

Scheduling model (see docs/rtc_lite_design.md for the long form)
----------------------------------------------------------------

The whole point of action chunking at runtime is to bridge the gap between
slow inference (~5-6 Hz on Spark) and a fast control loop (~50 Hz). A chunk
is a *forecast* anchored to the observation at the time inference was
launched; the further into the chunk we play, the staler the forecast is
relative to the robot's current state. When the next inference completes,
it is a strictly better estimate of "what to do now" than any remaining
action in the old chunk's tail, so we swap to it as soon as it arrives.

The runner exposes three knobs that together implement that policy:

* ``start_next_at = 0`` — fire the next inference as soon as the previous
  one completes (capped by inference rate itself, since only one inference
  is in flight at a time). This is the RTC paper's intent.
* ``inference_delay_steps`` (``d``) — index into the freshly-arrived chunk
  to start serving from. The first ``d`` actions of the new chunk correspond
  to control ticks that already elapsed while inference was running; we
  served the OLD chunk's actions for those ticks. So at swap time we jump
  ahead to ``new_chunk[d]``. When ``None``, the runner auto-tracks d from
  observed inference latency.
* ``blend_steps`` — number of control ticks at the START of each new chunk
  to alpha-blend from the last served action toward the new chunk's raw
  action. Smooths the discontinuity at the swap seam. The OLD semantics
  of ``blend_steps`` (tail damping on deadline miss) is preserved as
  ``tail_blend_steps`` for the deadline-miss path.

References
----------
* "π₀.₅: a VLA with Open-World Generalization" — Black et al. 2025
  (https://arxiv.org/abs/2504.16054), §IV-E for the 50 Hz control rate.
* "Real-Time Execution of Action Chunking Flow Policies" — Black et al.
  2025 (https://arxiv.org/abs/2506.07339), the formal treatment of
  splice-at-d and seam blending. Server-side inpainting (mode 5 in the
  ChunkedWebsocketClient mode catalogue) is deferred.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np


class ActionChunkAdapter(Protocol):
    """Minimal adapter contract for a chunked action model."""

    def infer_actions(self, observation: Any) -> np.ndarray:
        """Return an action chunk shaped ``[horizon, action_dim]``."""


@dataclass(frozen=True)
class CallablePolicyAdapter:
    """Wrap a Python callable as an :class:`ActionChunkAdapter`.

    ``output_key`` covers frontends that return ``{"actions": array}``.
    ``tuple_index`` covers frontends that return tuples such as
    ``(frames, actions)``.
    """

    fn: Callable[[Any], Any]
    output_key: str | None = "actions"
    tuple_index: int | None = None

    def infer_actions(self, observation: Any) -> np.ndarray:
        out = self.fn(observation)
        if self.tuple_index is not None:
            out = out[self.tuple_index]
        elif self.output_key is not None and isinstance(out, Mapping):
            out = out[self.output_key]
        actions = np.asarray(out)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                f"expected action chunk [horizon, action_dim], got {actions.shape}")
        return actions


@dataclass(frozen=True)
class RTCConfig:
    """Configuration for asynchronous chunk execution.

    Knobs grouped by purpose:

    Scheduling
    ~~~~~~~~~~
    ``target_hz``
        Foreground controller rate. Used for ``period_s`` and to convert
        latency-in-seconds to latency-in-ticks for auto ``inference_delay``.
    ``action_horizon``
        Maximum horizon to serve from a chunk. ``None`` means use the full
        chunk as returned by the policy.
    ``start_next_at``
        Action index at which to fire the next background inference. Set to
        ``0`` for "fire as soon as the previous one completes" (the RTC
        paper's intent — gives the freshest possible plan). ``None`` falls
        back to ``max(1, horizon // 2)`` for backward compatibility with the
        original RTC-lite scheduling.

    Splice-at-d (RTC paper §3 — "the executed prefix" handling)
    ~~~~~~~~~~~
    ``inference_delay_steps``
        ``d`` — the number of control ticks the foreground loop is expected
        to consume while one inference runs. When a fresh chunk lands we
        skip past ``new_chunk[:d]`` (those actions correspond to time we
        already lived through serving the old chunk) and start serving from
        ``new_chunk[d]``. ``None`` (default) means keep the legacy "splice
        at 0" behavior. Set to a positive int, or set ``auto_inference_delay``
        below, to enable RTC-paper splicing.
    ``auto_inference_delay``
        If True, track the EMA of measured inference latency and compute
        ``d = ceil(ema_latency_s * target_hz)`` on every swap. Overrides
        ``inference_delay_steps``.
    ``latency_ema_alpha``
        EMA smoothing factor for ``auto_inference_delay`` (closer to 1 =
        more responsive to recent latency, closer to 0 = more stable).

    Seam smoothing
    ~~~~~~~~~~~~~~
    ``blend_steps``
        Number of control ticks at the START of each freshly-promoted chunk
        to alpha-blend from ``last_served_action`` toward the new chunk's
        raw action. Alpha ramps linearly: ``alpha_k = (k+1) / (N+1)`` for
        ``k=0..N-1``, so the first emitted action is mostly the previous
        target and the Nth-out emitted action is almost the new target.
        Set to ``0`` to disable seam blending entirely.
    ``tail_blend_steps``
        Legacy "tail damping" for the deadline-miss path: when the chunk is
        about to run out and no replacement is ready, the last
        ``tail_blend_steps`` actions are pulled toward the last served
        action so the PD controller does not jerk on the held-target.
        Defaults to ``0``.

    Miss handling
    ~~~~~~~~~~~~~
    ``miss_policy``
        ``"hold_last"`` repeats the last served action when the chunk is
        exhausted before a replacement arrives. ``"block"`` synchronously
        runs another inference (turns the runner into a sync broker; useful
        for the mode-1 baseline).

    Internal
    ~~~~~~~~
    ``max_workers``
        Must be ``1``. RTC-lite supports exactly one background inference
        in flight; with one worker the executor naturally caps inference
        rate at ``1 / latency``.
    """

    target_hz: float = 20.0
    action_horizon: int | None = None
    start_next_at: int | None = None
    miss_policy: str = "hold_last"
    blend_steps: int = 0
    inference_delay_steps: int | None = None
    auto_inference_delay: bool = False
    latency_ema_alpha: float = 0.3
    tail_blend_steps: int = 0
    max_workers: int = 1

    def __post_init__(self) -> None:
        if self.target_hz <= 0:
            raise ValueError("target_hz must be positive")
        if self.action_horizon is not None and self.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if self.start_next_at is not None and self.start_next_at < 0:
            raise ValueError("start_next_at must be non-negative")
        if self.miss_policy not in {"hold_last", "block"}:
            raise ValueError("miss_policy must be 'hold_last' or 'block'")
        if self.blend_steps < 0:
            raise ValueError("blend_steps must be non-negative")
        if self.tail_blend_steps < 0:
            raise ValueError("tail_blend_steps must be non-negative")
        if (
            self.inference_delay_steps is not None
            and self.inference_delay_steps < 0
        ):
            raise ValueError("inference_delay_steps must be non-negative")
        if not 0.0 < self.latency_ema_alpha <= 1.0:
            raise ValueError("latency_ema_alpha must lie in (0, 1]")
        if self.max_workers != 1:
            raise ValueError("RTC-lite supports exactly one model worker")

    @property
    def period_s(self) -> float:
        return 1.0 / self.target_hz


@dataclass
class ChunkResult:
    actions: np.ndarray
    latency_s: float
    observation_time_s: float
    ready_time_s: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RTCStats:
    chunks_started: int = 0
    chunks_completed: int = 0
    actions_served: int = 0
    deadline_misses: int = 0
    held_actions: int = 0
    swaps: int = 0
    last_latency_s: float = 0.0
    max_latency_s: float = 0.0
    ema_latency_s: float = 0.0
    last_splice_d: int = 0
    actions_blended: int = 0


class AsyncChunkRunner:
    """Run an action chunk model asynchronously while actions are consumed.

    The runner does not sleep or own the controller loop. Call ``next_action``
    once per controller tick with the latest observation. The first call blocks
    to produce the initial chunk. Later calls trigger background inference when
    enough of the current chunk has been consumed and serve the freshest
    available action.

    Per-tick state machine inside ``next_action``:

    1. Promote a freshly-completed pending chunk if one is ready.
       - Splice index: ``idx = d`` where ``d`` is either the configured
         ``inference_delay_steps`` or the auto-tracked latency-derived value.
       - Snapshot ``last_served_action`` as the seam anchor so the next
         ``blend_steps`` actions ramp from it to the new chunk.
    2. Fire the next background inference if (a) the current ``idx`` has
       reached ``start_next_at`` AND (b) no inference is already in flight.
       With ``start_next_at == 0`` this means "fire on every tick after a
       promotion", which the executor naturally serialises to at most one
       inference at a time.
    3. If the current chunk is exhausted, apply the miss policy.
    4. Serve the action at ``idx``. Apply seam-blend if we're within
       ``blend_steps`` of the last swap; apply tail-blend if we're within
       ``tail_blend_steps`` of the current chunk's end.
    """

    def __init__(self, adapter: ActionChunkAdapter, config: RTCConfig):
        self.adapter = adapter
        self.config = config
        self.stats = RTCStats()
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._current: ChunkResult | None = None
        self._pending: Future[ChunkResult] | None = None
        self._idx = 0
        self._last_action: np.ndarray | None = None
        # Seam-blend anchor: the action that was being served immediately
        # before the most recent swap. ``_blend_step`` counts how many seam
        # actions have been emitted since that swap (0..blend_steps).
        self._seam_anchor: np.ndarray | None = None
        self._blend_step = 0
        self._closed = False

    def close(self, *, wait: bool = False) -> None:
        """Shut down the background executor.

        Args:
            wait: If True, block until any in-flight inference completes.
                Useful when the underlying policy holds shared resources
                (e.g. a single websocket) that the next consumer needs
                in a quiesced state. Default False keeps the historical
                fire-and-forget behaviour for in-process consumers that
                hold no shared state.
        """
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=not wait)

    def __enter__(self) -> "AsyncChunkRunner":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def reset(self, observation: Any) -> None:
        """Synchronously initialize the first action chunk."""
        result = self._run_inference(observation)
        with self._lock:
            self._current = result
            self._pending = None
            self._idx = 0
            self._last_action = None
            self._seam_anchor = None
            self._blend_step = 0
            self._record_latency_locked(result.latency_s)

    def next_action(self, observation: Any, *, block_if_empty: bool = True) -> np.ndarray:
        """Return the next action for the foreground control loop."""
        self._raise_if_closed()
        if self._current is None:
            if not block_if_empty:
                raise RuntimeError("RTC runner has no current chunk")
            self.reset(observation)

        with self._lock:
            self._promote_ready_locked()
            current = self._current
            if current is None:
                raise RuntimeError("RTC runner failed to initialize")
            horizon = self._configured_horizon(current.actions)
            start_next_at = self._start_next_at(horizon)
            if self._idx >= start_next_at:
                self._submit_locked(observation)
            if self._idx >= horizon:
                self._handle_exhausted_locked(observation)
                current = self._current
                if current is None:
                    raise RuntimeError("RTC runner has no chunk after exhaustion")
            action = np.asarray(current.actions[self._idx]).copy()
            if self.config.blend_steps > 0 and self._blend_step < self.config.blend_steps:
                action = self._seam_blend_locked(action)
            elif self.config.tail_blend_steps > 0:
                action = self._tail_blend_locked(action)
            self._idx += 1
            self._last_action = action
            self.stats.actions_served += 1
            return action

    def _run_inference(self, observation: Any) -> ChunkResult:
        t0 = time.perf_counter()
        actions = self.adapter.infer_actions(observation)
        t1 = time.perf_counter()
        return ChunkResult(
            actions=np.asarray(actions),
            latency_s=t1 - t0,
            observation_time_s=t0,
            ready_time_s=t1,
        )

    def _submit_locked(self, observation: Any) -> None:
        if self._pending is not None:
            return
        self.stats.chunks_started += 1
        self._pending = self._executor.submit(self._run_inference, observation)

    def _promote_ready_locked(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        result = self._pending.result()
        self._pending = None
        self._record_latency_locked(result.latency_s)
        horizon = self._configured_horizon(result.actions)
        d = self._effective_splice_d_locked(horizon)
        self._current = result
        # Snapshot the seam anchor BEFORE overwriting _idx. If
        # blend_steps>0, the next emitted action will linearly interpolate
        # from this anchor toward the new chunk's actions[d].
        if self._last_action is not None:
            self._seam_anchor = np.asarray(self._last_action).copy()
        else:
            self._seam_anchor = None
        self._blend_step = 0
        self._idx = d
        self.stats.chunks_completed += 1
        self.stats.swaps += 1
        self.stats.last_splice_d = d

    def _handle_exhausted_locked(self, observation: Any) -> None:
        self._promote_ready_locked()
        current = self._current
        if current is not None and self._idx < self._configured_horizon(current.actions):
            return
        self.stats.deadline_misses += 1
        if self.config.miss_policy == "block":
            self._submit_locked(observation)
            if self._pending is None:
                raise RuntimeError("failed to submit recovery chunk")
            result = self._pending.result()
            self._pending = None
            self._record_latency_locked(result.latency_s)
            horizon = self._configured_horizon(result.actions)
            d = self._effective_splice_d_locked(horizon)
            self._current = result
            if self._last_action is not None:
                self._seam_anchor = np.asarray(self._last_action).copy()
            else:
                self._seam_anchor = None
            self._blend_step = 0
            self._idx = d
            self.stats.chunks_completed += 1
            self.stats.swaps += 1
            self.stats.last_splice_d = d
            return
        self.stats.held_actions += 1
        if self._last_action is None:
            raise RuntimeError("cannot hold last action before any action was served")
        now = time.perf_counter()
        self._current = ChunkResult(
            actions=self._last_action[None, :],
            latency_s=0.0,
            observation_time_s=now,
            ready_time_s=now,
            metadata={"held": True},
        )
        # A "held" chunk is a stop-gap, not a real swap — don't restart
        # the seam blend, don't snapshot a new anchor.
        self._idx = 0

    def _seam_blend_locked(self, action: np.ndarray) -> np.ndarray:
        """Linearly ramp from ``_seam_anchor`` to ``action`` over ``blend_steps`` ticks.

        With ``N = blend_steps`` and ``k = _blend_step`` (0-indexed), the
        emitted action is ``alpha * action + (1 - alpha) * anchor`` where
        ``alpha = (k + 1) / (N + 1)``. So ``k=0`` emits mostly anchor,
        ``k=N-1`` emits mostly action, and from ``k=N`` onward we serve
        the raw new chunk.
        """
        if self._seam_anchor is None:
            self._blend_step = self.config.blend_steps
            return action
        n = self.config.blend_steps
        k = self._blend_step
        alpha = (k + 1) / (n + 1)
        blended = alpha * action + (1.0 - alpha) * self._seam_anchor
        self._blend_step += 1
        self.stats.actions_blended += 1
        return blended.astype(action.dtype, copy=False)

    def _tail_blend_locked(self, action: np.ndarray) -> np.ndarray:
        """Legacy tail-damp: smooth the LAST ``tail_blend_steps`` actions of
        a chunk toward the last served action when no replacement chunk is
        ready yet. Defensive smoothing for the deadline-miss path.
        """
        if self._last_action is None:
            return action
        current = self._current
        if current is None:
            return action
        horizon = self._configured_horizon(current.actions)
        remaining = horizon - self._idx
        if remaining > self.config.tail_blend_steps:
            return action
        alpha = 1.0 / (remaining + 1)
        blended = (1.0 - alpha) * self._last_action + alpha * action
        return blended.astype(action.dtype, copy=False)

    def _configured_horizon(self, actions: np.ndarray) -> int:
        horizon = actions.shape[0]
        if self.config.action_horizon is not None:
            horizon = min(horizon, self.config.action_horizon)
        return horizon

    def _start_next_at(self, horizon: int) -> int:
        if self.config.start_next_at is not None:
            return min(self.config.start_next_at, horizon)
        return max(1, horizon // 2)

    def _record_latency_locked(self, latency_s: float) -> None:
        self.stats.last_latency_s = latency_s
        self.stats.max_latency_s = max(self.stats.max_latency_s, latency_s)
        ema = self.stats.ema_latency_s
        if ema <= 0.0:
            self.stats.ema_latency_s = latency_s
        else:
            a = self.config.latency_ema_alpha
            self.stats.ema_latency_s = a * latency_s + (1.0 - a) * ema

    def _effective_splice_d_locked(self, horizon: int) -> int:
        """Compute the splice index ``d`` for the chunk about to be promoted.

        Resolution order: explicit ``inference_delay_steps`` > auto-tracked
        EMA latency > 0 (legacy "splice at start"). Always clipped to
        ``[0, horizon - 1]`` so we serve at least one action from the new
        chunk before the runner can swap again.
        """
        if self.config.inference_delay_steps is not None:
            d = int(self.config.inference_delay_steps)
        elif self.config.auto_inference_delay:
            ema = self.stats.ema_latency_s
            d = int(math.ceil(ema * self.config.target_hz)) if ema > 0 else 0
        else:
            d = 0
        if d < 0:
            d = 0
        if horizon > 0 and d > horizon - 1:
            d = horizon - 1
        return d

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("RTC runner is closed")
