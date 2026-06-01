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
  splice-at-d, seam blending, and the §3.2 "hard inpainting" prefix-
  freeze that mode 5 of the ChunkedWebsocketClient surfaces here.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import threading
import time
from typing import Any, Callable, Mapping, Protocol

import numpy as np


_logger = logging.getLogger(__name__)


class ActionChunkAdapter(Protocol):
    """Minimal adapter contract for a chunked action model.

    Backends that participate in RTC prefix-freeze should also expose an
    ``infer_actions_with_meta`` method returning ``(actions, metadata)``;
    when present, ``AsyncChunkRunner`` will use that path and store the
    metadata on :class:`ChunkResult` so the next submission can pull the
    cached model-space chunk back out. Backends that only implement
    ``infer_actions`` keep working — they just can't drive prefix-freeze.
    """

    def infer_actions(self, observation: Any) -> np.ndarray:
        """Return an action chunk shaped ``[horizon, action_dim]``."""


@dataclass(frozen=True)
class CallablePolicyAdapter:
    """Wrap a Python callable as an :class:`ActionChunkAdapter`.

    ``output_key`` covers frontends that return ``{"actions": array}``.
    ``tuple_index`` covers frontends that return tuples such as
    ``(frames, actions)``.

    ``meta_keys`` names extra dict entries to lift out of the callable's
    response into the per-chunk metadata. Used by the RTC prefix-freeze
    path to pull ``_rtc_chunk_model_space`` back out of an openpi-style
    response. Has no effect when the callable returns a non-dict.
    """

    fn: Callable[[Any], Any]
    output_key: str | None = "actions"
    tuple_index: int | None = None
    meta_keys: tuple[str, ...] = ()

    def infer_actions(self, observation: Any) -> np.ndarray:
        actions, _ = self.infer_actions_with_meta(observation)
        return actions

    def infer_actions_with_meta(
            self, observation: Any) -> tuple[np.ndarray, dict[str, Any]]:
        out = self.fn(observation)
        meta: dict[str, Any] = {}
        if self.tuple_index is not None:
            actions_raw = out[self.tuple_index]
        elif self.output_key is not None and isinstance(out, Mapping):
            actions_raw = out[self.output_key]
            for key in self.meta_keys:
                if key in out:
                    meta[key] = out[key]
        else:
            actions_raw = out
        actions = np.asarray(actions_raw)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2:
            raise ValueError(
                f"expected action chunk [horizon, action_dim], got {actions.shape}")
        return actions, meta


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
        at 0" behavior. With ``auto_inference_delay=True`` this field
        degrades to a one-time seed for the very first promotion (before
        any latency has been measured); every subsequent promotion uses
        the per-call measured latency for ``d``.
    ``auto_inference_delay``
        If True (RTC-paper-correct), compute ``d`` from MEASURED inference
        latency at promotion time: ``d = ceil(this_call_latency_s * target_hz)``.
        Per-call (not EMA) because variable latency (e.g. FlashRT pipeline
        rebuilds, network jitter) makes the EMA underestimate fresh long
        outliers and lag fresh short recoveries, both of which produce
        time-skip jumps at the splice. EMA is used only as a fallback
        when no per-call measurement is available (i.e. inside
        :meth:`reset`).
    ``latency_ema_alpha``
        EMA smoothing factor (closer to 1 = more responsive to recent
        latency, closer to 0 = more stable). Used only on the EMA
        fallback path inside :meth:`reset`; steady-state ``d`` is
        per-call, not EMA-smoothed.

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

    Server-side prefix-freeze inpainting (RTC paper §3.2 "soft / hard
    inpainting")
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ``enable_prefix_freeze``
        If True, the runner attaches ``_rtc_prev_chunk`` and
        ``_rtc_inference_delay`` to every observation submitted to a
        background inference. The model server is expected to force the
        first ``d`` positions of the new chunk to match those values
        during diffusion denoising (hard-freeze inpainting). The runner
        also caches the **model-space** chunk returned by the policy
        (``_rtc_chunk_model_space`` field in the result dict) so the
        prefix it sends back is in the same coordinate frame the model
        was trained on (post-norm, pre-unnorm). This is the smoothness
        guarantee RTC was designed for: chunk boundaries become
        continuous trajectories instead of two independent plans
        crudely stitched together by client-side seam blending. Has no
        effect on backends that ignore the ``_rtc_*`` keys.

        ``d_predicted`` for the prefix is computed from the EMA
        latency (``stats.ema_latency_s * target_hz``) plus the safety
        margin in ``prefix_freeze_margin_steps``. The actual splice
        ``d_measured`` (used to skip into the returned chunk) still
        comes from the per-call latency on promotion. If
        ``d_predicted >= d_measured`` the splice always lands in the
        frozen prefix region → continuous. If
        ``d_predicted < d_measured`` it lands in the free region and
        we fall back to whatever the model plans (i.e. degraded to
        ordinary RTC-lite). The default margin biases toward the
        first case.
    ``prefix_freeze_margin_steps``
        Extra ticks added to the EMA-predicted ``d`` for the freeze
        prefix length. Bigger margin = more positions frozen = more
        likely the splice lands in the frozen region (smoother) but
        more positions where the model has no freedom. Default 2.
    ``prefix_freeze_max_steps``
        Hard cap on the freeze prefix length, regardless of latency.
        Defaults to half the chunk horizon — freezing more than half
        of a chunk leaves the model with too little freedom to plan
        anything new.

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
    enable_prefix_freeze: bool = False
    prefix_freeze_margin_steps: int = 2
    prefix_freeze_max_steps: int | None = None
    # RTC soft-guidance (Phase 6 / G11). Forwarded to the server via
    # ``_rtc_execution_horizon`` / ``_rtc_schedule`` so the model's
    # per-Euler-step guidance kernel knows the merge window shape.
    # None means "use the server frontend's default" (lerobot default
    # is execution_horizon=10, schedule='linear'). Setting these on the
    # client side overrides per-call without changing the model
    # config / requiring a graph recapture.
    execution_horizon: int | None = None
    rtc_schedule: str | None = None
    # Hard cap on the splice index ``d`` (latency-miss guard). A latency
    # spike (e.g. a server pipeline rebuild) makes the per-call
    # ``d = ceil(latency * target_hz)`` jump far into the fresh chunk —
    # past ``execution_horizon`` the model never guided continuity, so the
    # splice becomes a raw discontinuity (the joint whip that trips the
    # follower's per-step safety gate). Clamping ``d`` to this value keeps
    # the splice inside the guided merge window: the trajectory lags a few
    # ticks in real time but stays CONTINUOUS, and continuous replanning
    # (``start_next_at=0``) catches the lag back up within a chunk or two.
    # ``None`` (default) = no cap (legacy behavior). Set to
    # ``execution_horizon`` to bound spikes to the guided region.
    max_splice_d_steps: int | None = None
    # Relative-action prefix re-anchoring (delta-policy continuity). The
    # cached ``_rtc_prev_chunk`` holds the previous chunk's NORMALIZED
    # per-step actions. For a delta-action policy those are deltas
    # relative to the state at the inference that produced them — not the
    # state the next inference sees. Left as-is the server guides the new
    # chunk toward continuity with a STALE frame, so the seam reappears
    # scaled by however far the robot moved during the inference window
    # (this is what lerobot's ``_reanchor_relative_rtc_prefix`` corrects).
    # When set, the runner reads ``observation[ref_state_key]`` at submit
    # time, caches it with the chunk, and forwards it back as
    # ``_rtc_ref_state`` so the server can re-express the prefix relative
    # to the current state before guidance. ``None`` = absolute-action
    # policy / no re-anchoring (legacy behavior).
    ref_state_key: str | None = None
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
        if self.prefix_freeze_margin_steps < 0:
            raise ValueError(
                "prefix_freeze_margin_steps must be non-negative")
        if (
            self.prefix_freeze_max_steps is not None
            and self.prefix_freeze_max_steps < 0
        ):
            raise ValueError(
                "prefix_freeze_max_steps must be non-negative")
        if (
            self.execution_horizon is not None
            and self.execution_horizon <= 0
        ):
            raise ValueError("execution_horizon must be positive")
        if (
            self.max_splice_d_steps is not None
            and self.max_splice_d_steps < 0
        ):
            raise ValueError("max_splice_d_steps must be non-negative")
        if self.rtc_schedule is not None and self.rtc_schedule not in (
                "linear", "exp", "ones", "zeros"):
            raise ValueError(
                "rtc_schedule must be one of "
                "'linear', 'exp', 'ones', 'zeros'")
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
    # Normalised model-space action chunk (e.g. Pi0.5 32-dim, [-1, 1])
    # cached so the next inference can pass it back via
    # ``_rtc_prev_chunk`` for server-side hard-freeze inpainting. None
    # when the adapter does not expose one (older backends).
    chunk_model_space: np.ndarray | None = None
    # State (physical, model layout) this chunk's deltas are anchored to,
    # i.e. the ``observation[cfg.ref_state_key]`` that produced it. Used
    # to re-anchor the delta prefix to the next inference's state (see
    # ``RTCConfig.ref_state_key``). None when re-anchoring is disabled.
    ref_state: np.ndarray | None = None


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
        # Prefix-freeze splice coupling: the augmentation function records
        # the length of the frozen prefix it just sent to the backend so
        # that, when the resulting chunk arrives, the promotion code can
        # set the splice index ``d`` to land at the FIRST FREE position
        # (i.e. past the frozen region). Without this, a latency-derived
        # ``d`` smaller than ``d_pred`` causes us to serve actions from
        # inside the frozen region — which in sync mode are replays of
        # OLD chunk actions, producing a backward jump at the boundary.
        # See ``_maybe_augment_with_prefix_locked`` and the promotion
        # paths for the math; bug surfaced in chocolate_bars on 2026-05.
        self._last_d_pred: int = 0
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
            self._last_d_pred = 0
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
        if hasattr(self.adapter, "infer_actions_with_meta"):
            actions, meta = self.adapter.infer_actions_with_meta(observation)
        else:
            actions = self.adapter.infer_actions(observation)
            meta = {}
        t1 = time.perf_counter()
        chunk_ms = meta.pop("_rtc_chunk_model_space", None)
        ref_state = None
        key = self.config.ref_state_key
        if key is not None and isinstance(observation, Mapping):
            rs = observation.get(key)
            if rs is not None:
                ref_state = np.asarray(rs, dtype=np.float32).reshape(-1).copy()
        return ChunkResult(
            actions=np.asarray(actions),
            latency_s=t1 - t0,
            observation_time_s=t0,
            ready_time_s=t1,
            metadata=meta,
            chunk_model_space=(
                np.asarray(chunk_ms) if chunk_ms is not None else None),
            ref_state=ref_state,
        )

    def _submit_locked(self, observation: Any) -> None:
        if self._pending is not None:
            return
        submit_obs = self._maybe_augment_with_prefix_locked(observation)
        self.stats.chunks_started += 1
        self._pending = self._executor.submit(self._run_inference, submit_obs)

    def _maybe_augment_with_prefix_locked(self, observation: Any) -> Any:
        """Attach ``_rtc_prev_chunk`` + ``_rtc_inference_delay`` to a dict obs.

        No-op if ``enable_prefix_freeze`` is False, if there is no current
        chunk to take a prefix from, if the current chunk has no cached
        model-space form, or if ``observation`` is not dict-like (we never
        mutate non-dict observations because the backend's idea of what
        ``observation`` should look like is opaque to us).
        """
        cfg = self.config
        if not cfg.enable_prefix_freeze:
            return observation
        if not isinstance(observation, Mapping):
            return observation
        current = self._current
        if current is None or current.chunk_model_space is None:
            return observation
        horizon_actions = current.actions.shape[0]
        horizon = self._configured_horizon(current.actions)
        idx_at_submit = self._idx
        remaining = max(0, horizon - idx_at_submit)

        cap = cfg.prefix_freeze_max_steps
        if cap is None:
            cap = max(1, horizon_actions // 2)

        cm = current.chunk_model_space

        if remaining <= 0:
            # Sync block-mode case (e.g. mode 1 + prefix_freeze): submit
            # fires at chunk exhaustion (self._idx == horizon). There are
            # no future actions of the current chunk to project the
            # prefix from — instead, take the LAST d_pred actions we
            # already played as the anchor prefix. The server constrains
            # the new chunk's first d_pred positions to MATCH those (the
            # actions the low-level controller is currently tracking
            # toward), so the inter-chunk boundary is continuous by
            # construction. After block, _idx is set to d_pred so we
            # skip the constrained prefix and serve the model's free
            # continuation from the new chunk's d_pred-th action.
            #
            # IMPORTANT: in sync mode no control ticks elapse during
            # inference (the control loop is blocked on the result), so
            # we do NOT need d_pred to predict elapsed-tick count. We
            # only need enough anchored positions to give the soft-
            # guidance kernel a stable boundary condition. Keep d_pred
            # small so we waste as few of the new chunk's model-free
            # predictions as possible.
            d_pred = int(cfg.prefix_freeze_margin_steps) + 2
            prev_end = min(cm.shape[0], idx_at_submit)
            d_pred = max(1, min(d_pred, cap, prev_end))
            if d_pred <= 0:
                return observation
            # Soft-guidance Phase 6 (G11): in sync block mode there's
            # nothing future of the current chunk to merge against, so
            # the prev_chunk is just the anchor positions. The
            # frontend's exec_horizon will cap to d_pred and the merge
            # window will be empty — effectively a hard anchor + free
            # continuation, which is the right behavior here.
            prev_prefix = np.asarray(cm[prev_end - d_pred : prev_end]).copy()
        else:
            # Async case (modes 2..5): submit fires while the current
            # chunk is still being consumed. The first d_pred positions
            # of the new chunk overlap actions we WILL HAVE played by
            # the time the new chunk arrives (committed). Positions
            # past d_pred are the previous chunk's free plan — these
            # are the merge window for soft guidance.
            ema_ticks = max(0.0, self.stats.ema_latency_s * cfg.target_hz)
            d_pred = (int(math.ceil(ema_ticks))
                      + int(cfg.prefix_freeze_margin_steps))
            d_pred = max(0, min(d_pred, cap, remaining))
            if d_pred <= 0:
                return observation
            if cm.shape[0] < idx_at_submit + d_pred:
                d_pred = max(0, cm.shape[0] - idx_at_submit)
                if d_pred <= 0:
                    return observation
            # Soft-guidance Phase 6 (G11): send the entire unconsumed
            # tail of the previous chunk (not just the d_pred anchor
            # positions). The frontend uses this to populate the merge
            # window past d_pred where the model is softly nudged
            # toward continuity but not hard-anchored. Without the
            # tail, the merge-window weights would multiply against
            # zero-padded prev values and steer the trajectory toward
            # 0, collapsing the action stream. The runtime cost is
            # negligible: a numpy slice of a (chunk_size, ACTION_DIM)
            # tensor that the runner already has cached.
            #
            # Wire-format-compatible: ``_rtc_inference_delay`` is still
            # d_pred (the anchor length); the prefix tensor itself is
            # longer. Old hard-freeze servers will only consume the
            # first d_pred rows; soft-guidance servers will use the
            # whole thing.
            prev_prefix = np.asarray(cm[idx_at_submit:]).copy()

        augmented = dict(observation)
        augmented["_rtc_prev_chunk"] = prev_prefix
        augmented["_rtc_inference_delay"] = int(d_pred)
        # Relative-action re-anchoring: tell the server which state the
        # prefix's deltas are anchored to, so it can re-express them
        # relative to the state in THIS observation before guidance
        # (delta-policy continuity — see RTCConfig.ref_state_key). The
        # server no-ops this when it isn't a delta-action model.
        if current.ref_state is not None:
            augmented["_rtc_ref_state"] = current.ref_state
        # Per-call soft-guidance config (Phase 6 / G11). Only set when
        # the config explicitly overrides — None means "let the model
        # frontend use its own default" (lerobot defaults
        # execution_horizon=10, schedule='linear').
        if cfg.execution_horizon is not None:
            augmented["_rtc_execution_horizon"] = int(cfg.execution_horizon)
        if cfg.rtc_schedule is not None:
            augmented["_rtc_schedule"] = str(cfg.rtc_schedule)
        # Coupling for the splice path: when the resulting chunk arrives,
        # _idx must land at d_pred (the first FREE position past the
        # frozen prefix). Stored here, consumed by _promote_ready_locked
        # and the block branch of _handle_exhausted_locked.
        self._last_d_pred = int(d_pred)
        return augmented

    def _promote_ready_locked(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        result = self._pending.result()
        self._pending = None
        self._record_latency_locked(result.latency_s)
        horizon = self._configured_horizon(result.actions)
        # Async splice: ``d`` must equal the number of control ticks of
        # the OLD chunk we played while the new inference was in flight.
        # We splice at chunk_new[d] which then sits at exactly the same
        # trajectory position as chunk_old[idx_at_submit + d] — and the
        # last action we just served was chunk_old[idx_at_submit + d - 1].
        # With prefix-freeze active, chunk_new[d-1] is FROZEN to equal
        # chunk_old[idx_at_submit + d - 1] in model space, so the splice
        # is a smooth single-tick continuation. (When d == d_pred_used
        # we land at the first FREE position; when d < d_pred_used we
        # land inside the frozen region and serve a constrained action
        # that equals what we would have served from chunk_old anyway
        # — also smooth.)
        #
        # IMPORTANT: do NOT lift ``d`` up to ``max(d, d_pred_used)`` in
        # the async path. That sounds defensive but actually makes the
        # splice JUMP FORWARD by (d_pred_used - d) ticks: we'd serve
        # chunk_new[d_pred_used] (free continuation from frozen anchor
        # at index d_pred_used-1 = chunk_old[idx+d_pred-1]) when our
        # actual last-served was chunk_old[idx+d-1]. That's a (d_pred-d)-
        # tick "fast-forward" worth ~0.03 rad/tick → visible jerkiness
        # at every swap. Symptom observed in mode 5 chocolate_bars,
        # 2026-05; remove of the max() restored smooth async boundaries.
        # The ``_last_d_pred`` is still consumed here so the sync block
        # path (which DOES need it) gets a clean slate next time.
        d = self._effective_splice_d_locked(
            horizon, measured_latency_s=result.latency_s)
        self._last_d_pred = 0
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
        # Promoted from DEBUG to INFO in the FlashRT G7 hotfix follow-up
        # (2026-05): SparkJAX's ROS2 nodes default the Python logger to
        # INFO, so DEBUG telemetry never reaches their log file. This is
        # bounded at the chunk-swap rate (~5-10 Hz worst case at 50 Hz
        # control + 50-step chunks), so the log volume is fine, and it
        # gives us the only direct evidence that auto-d is tracking
        # real latency vs. regressing to a stale seed. Move back to
        # DEBUG only if the line shows up as a measurable hot-path cost.
        _logger.info(
            "swap: latency=%.1f ms -> d=%d (horizon=%d, seam_blend=%d)",
            result.latency_s * 1000.0, d, horizon, self.config.blend_steps)

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
            # Sync block path: the control loop was blocked on the
            # result while inference ran, so NO control ticks elapsed
            # during inference. The measured-latency-derived ``d`` is
            # semantically meaningless here — it would tell us how
            # many ticks of chunk_A we played during inference, but
            # the answer is zero by construction.
            #
            # The correct splice in sync mode is exactly ``d_pred_used``:
            # land at the first model-free position past the frozen
            # prefix. Splicing later (e.g. at measured_d when the
            # inference was slow due to a pipeline rebuild) lands in
            # chunk_B's free region with NO continuity guarantee
            # against chunk_A[49] — visible as a multi-tenth-radian
            # jump at the boundary. Splicing earlier (inside the
            # frozen region) re-serves OLD chunk_A actions (the bug
            # the original G9 fix already addressed for ``d_pred > d``).
            #
            # If no prefix was sent (older configs or prefix-freeze
            # disabled), fall back to the latency-derived d.
            d_pred_used = self._last_d_pred
            self._last_d_pred = 0
            if d_pred_used > 0:
                d = d_pred_used
                if horizon > 0 and d > horizon - 1:
                    d = horizon - 1
            else:
                d = self._effective_splice_d_locked(
                    horizon, measured_latency_s=result.latency_s)
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

    def _effective_splice_d_locked(
        self, horizon: int, *, measured_latency_s: float | None = None
    ) -> int:
        """Compute the splice index ``d`` for the chunk about to be promoted.

        Resolution order (highest priority first):
          1. ``auto_inference_delay`` + a per-call ``measured_latency_s``
             from the just-completed inference. This is the RTC-paper-
             correct quantity: "how many control ticks elapsed while
             this specific inference was in flight". Required for any
             scenario where inference latency varies between calls
             (e.g. FlashRT Pi05 pipeline rebuilds, network jitter).
          2. ``auto_inference_delay`` + EMA of measured latencies. Used
             when no per-call measurement is available (e.g. during
             ``reset``). Equivalent to per-call once steady state.
          3. Explicit ``inference_delay_steps`` (fixed value). Used
             only when ``auto_inference_delay`` is False, OR as a
             one-time seed before any latency has been measured.
          4. ``0`` ("splice at start" — the legacy non-RTC behavior).

        Always clipped to ``[0, horizon - 1]`` so we serve at least one
        action from the new chunk before the runner can swap again.

        IMPORTANT precedence change (FlashRT G7 hotfix, 2026-05): the
        previous resolution order had ``inference_delay_steps`` winning
        over ``auto_inference_delay``, contradicting the docstring on
        ``RTCConfig.auto_inference_delay``. That made ``d`` a constant
        equal to the a-priori seed regardless of real latency, which on
        the OpenArm + FlashRT runtime (~140 ms inference, 50 Hz, d_seed
        based on 200 ms expected_latency_ms = 10) translated to a
        per-swap forward-time-jump of ~3 ticks ≈ 60 ms × 6-7 swaps/s ≈
        visibly jerky motion. Now ``auto_inference_delay`` wins and
        ``inference_delay_steps`` is demoted to a seed-only role.
        """
        cfg = self.config
        if cfg.auto_inference_delay:
            if measured_latency_s is not None and measured_latency_s > 0:
                lat = float(measured_latency_s)
            elif self.stats.ema_latency_s > 0:
                lat = float(self.stats.ema_latency_s)
            elif cfg.inference_delay_steps is not None:
                # Fall back to the a-priori seed for the very first
                # promotion, before any latency has been measured.
                d_seed = int(cfg.inference_delay_steps)
                if horizon > 0 and d_seed > horizon - 1:
                    d_seed = horizon - 1
                return max(0, d_seed)
            else:
                return 0
            d = int(math.ceil(lat * cfg.target_hz))
        elif cfg.inference_delay_steps is not None:
            d = int(cfg.inference_delay_steps)
        else:
            d = 0
        if d < 0:
            d = 0
        # Latency-miss guard: clamp a spike-inflated ``d`` to the guided
        # merge window so a rebuild/jitter spike degrades to a bounded,
        # still-continuous splice instead of a raw forward skip.
        if cfg.max_splice_d_steps is not None and d > cfg.max_splice_d_steps:
            _logger.warning(
                "splice d=%d exceeded max_splice_d_steps=%d (latency=%.0f ms); "
                "clamping to keep the splice inside the guided window",
                d,
                cfg.max_splice_d_steps,
                (lat * 1000.0) if cfg.auto_inference_delay else float("nan"),
            )
            d = cfg.max_splice_d_steps
        if horizon > 0 and d > horizon - 1:
            d = horizon - 1
        return d

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("RTC runner is closed")
