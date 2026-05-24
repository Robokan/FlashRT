import time

import numpy as np
import pytest

from flash_rt.runtime.rtc import AsyncChunkRunner, CallablePolicyAdapter, RTCConfig


def test_callable_policy_adapter_accepts_dict_output():
    adapter = CallablePolicyAdapter(
        lambda obs: {"actions": np.ones((4, 3), dtype=np.float32)}
    )

    out = adapter.infer_actions({"step": 0})

    assert out.shape == (4, 3)
    assert out.dtype == np.float32


def test_callable_policy_adapter_accepts_tuple_output():
    adapter = CallablePolicyAdapter(
        lambda obs: ("frames", np.ones((1, 4, 3), dtype=np.float32)),
        output_key=None,
        tuple_index=1,
    )

    out = adapter.infer_actions({"step": 0})

    assert out.shape == (4, 3)


def test_callable_policy_adapter_rejects_bad_shape():
    adapter = CallablePolicyAdapter(lambda obs: {"actions": np.ones((3,))})

    with pytest.raises(ValueError, match="expected action chunk"):
        adapter.infer_actions(None)


def test_async_runner_prefetches_next_chunk():
    """Legacy splice-at-0 path: no inference_delay_steps, no auto-d.

    Mirrors the original RTC-lite behavior so existing motus / libero
    callers that don't opt into splice-at-d still get the old semantics.
    """
    calls = []

    def policy(obs):
        calls.append(obs["chunk"])
        base = obs["chunk"] * 10
        return np.arange(base, base + 4, dtype=np.float32)[:, None]

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(target_hz=1000.0, action_horizon=4, start_next_at=2),
    )
    try:
        runner.reset({"chunk": 0})
        assert runner.next_action({"chunk": 1}).item() == 0.0
        assert runner.next_action({"chunk": 1}).item() == 1.0
        time.sleep(0.02)
        action = runner.next_action({"chunk": 2}).item()
        assert action in {2.0, 10.0}
        assert runner.stats.chunks_started >= 1
    finally:
        runner.close()


def test_async_runner_hold_last_on_miss():
    def slow_policy(obs):
        time.sleep(0.03)
        return np.array([[1.0], [2.0]], dtype=np.float32)

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(slow_policy),
        RTCConfig(target_hz=1000.0, action_horizon=2, start_next_at=1),
    )
    try:
        runner.reset({"step": 0})
        assert runner.next_action({"step": 1}).item() == 1.0
        assert runner.next_action({"step": 2}).item() == 2.0
        held = runner.next_action({"step": 3}).item()
        assert held == 2.0
        assert runner.stats.deadline_misses == 1
        assert runner.stats.held_actions == 1
    finally:
        runner.close()


def test_async_runner_fires_immediately_when_start_next_at_zero():
    """start_next_at=0 should refire as soon as the previous inference
    promotes; no need to wait for chunk exhaustion.
    """
    seen_obs: list[int] = []

    def policy(obs):
        seen_obs.append(int(obs["step"]))
        base = int(obs["step"]) * 100
        return np.arange(base, base + 8, dtype=np.float32)[:, None]

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(target_hz=1000.0, action_horizon=8, start_next_at=0),
    )
    try:
        runner.reset({"step": 0})
        for step in range(1, 12):
            runner.next_action({"step": step})
            time.sleep(0.001)
        assert runner.stats.chunks_started >= 2
        assert len(seen_obs) >= 2
    finally:
        runner.close()


def test_async_runner_splice_at_d_skips_stale_prefix():
    """With inference_delay_steps=d, the runner serves new_chunk[d] (not 0)
    on the first action after a swap.
    """
    chunks = []

    def policy(obs):
        idx = int(obs["chunk_idx"])
        base = idx * 100
        out = np.arange(base, base + 10, dtype=np.float32)[:, None]
        chunks.append(out.copy())
        return out

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=10,
            start_next_at=0,
            inference_delay_steps=3,
        ),
    )
    try:
        runner.reset({"chunk_idx": 0})
        served = []
        for tick in range(20):
            served.append(runner.next_action({"chunk_idx": 1}).item())
            time.sleep(0.002)
        assert runner.stats.swaps >= 1
        assert runner.stats.last_splice_d == 3
        post_swap = [v for v in served if v >= 100.0]
        assert post_swap, f"never observed chunk-1 actions: served={served}"
        assert post_swap[0] == 103.0, (
            f"expected first chunk-1 served action = new[d=3] = 103, "
            f"got {post_swap[0]}; served={served}"
        )
    finally:
        runner.close()


def test_async_runner_seam_blend_monotone():
    """With blend_steps=N, the first N actions after a swap should be a
    strictly monotonic ramp from anchor (last_action of prev chunk) toward
    new chunk's raw actions.

    The first bg inference is slow so the test window catches exactly one
    swap (chunk 0 -> chunk 1), giving a clean 4-tick ramp before the next
    inference would land. Subsequent calls block forever so the runner
    stays on chunk 1.
    """
    n_calls = [0]

    def policy(obs):
        idx = n_calls[0]
        n_calls[0] += 1
        if idx == 0:
            return np.full((20, 1), 0.0, dtype=np.float64)
        if idx == 1:
            time.sleep(0.03)
            return np.full((20, 1), 1.0, dtype=np.float64)
        time.sleep(5.0)
        return np.full((20, 1), 1.0, dtype=np.float64)

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=20,
            start_next_at=0,
            inference_delay_steps=0,
            blend_steps=4,
        ),
    )
    try:
        runner.reset({"chunk_idx": 0})
        served = []
        for _ in range(60):
            served.append(runner.next_action({"chunk_idx": 1}).item())
            time.sleep(0.002)
        anchor_zeros = [v for v in served if v == 0.0]
        assert anchor_zeros, "never served the anchor (chunk 0) actions"
        blend = []
        for v in served:
            if 0.0 < v < 1.0:
                blend.append(v)
            elif v == 1.0 and blend:
                break
        assert len(blend) == 4, (
            f"expected 4 seam-blend ticks, got {len(blend)}: {blend}"
        )
        for prev, nxt in zip(blend, blend[1:]):
            assert nxt > prev, (
                f"seam blend not monotonically increasing: {blend}"
            )
        expected = [(k + 1) / 5.0 for k in range(4)]
        assert blend == pytest.approx(expected, rel=1e-6), (
            f"blend ramp != expected linear schedule: got {blend} "
            f"expected {expected}"
        )
        assert runner.stats.actions_blended == 4
        assert runner.stats.swaps == 1
    finally:
        runner.close()


def test_async_runner_auto_inference_delay_tracks_latency():
    """auto_inference_delay should pick d ~ ceil(ema_latency * target_hz)."""
    def policy(obs):
        time.sleep(0.010)
        return np.arange(20, dtype=np.float32)[:, None]

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=20,
            start_next_at=0,
            auto_inference_delay=True,
        ),
    )
    try:
        runner.reset({"step": 0})
        for tick in range(30):
            runner.next_action({"step": tick})
            time.sleep(0.001)
        assert runner.stats.swaps >= 1
        assert runner.stats.ema_latency_s > 0.005
        assert runner.stats.last_splice_d >= 5, (
            f"expected splice d to track ~10 ms / 1 ms tick = ~10, "
            f"got {runner.stats.last_splice_d}; ema={runner.stats.ema_latency_s}"
        )
    finally:
        runner.close()


def test_async_runner_auto_d_ignores_fixed_seed_after_first_swap():
    """Regression test for the FlashRT G7 hotfix (2026-05).

    Bug: ``_effective_splice_d_locked`` previously returned the fixed
    ``inference_delay_steps`` value whenever it was set, even when
    ``auto_inference_delay=True`` was also requested. That contradicted
    the documented "auto overrides fixed" precedence and caused
    ``d`` to be locked at the a-priori seed for the entire run.

    On Spark (chunk_size=50, 50 Hz, ~140 ms actual inference vs 200 ms
    seed) this translated to splicing at ``d=10`` instead of ``d=7``
    every swap — a 3-tick (60 ms) forward time-jump per swap, visibly
    jerky on the robot.

    With the hotfix, when ``auto_inference_delay=True`` AND a per-call
    measured latency is available, the per-call latency wins. The
    ``inference_delay_steps`` field degrades to a one-shot seed for the
    very first promotion only.
    """
    def policy(obs):
        time.sleep(0.005)  # 5 ms inference, 5 ticks at 1000 Hz
        return np.arange(20, dtype=np.float32)[:, None]

    # Seed claims 20 ticks (= 20 ms latency). Auto should override with
    # the actual ~5 ticks (= 5 ms latency).
    runner = AsyncChunkRunner(
        CallablePolicyAdapter(policy),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=20,
            start_next_at=0,
            inference_delay_steps=20,   # WAY larger than actual
            auto_inference_delay=True,
        ),
    )
    try:
        runner.reset({"step": 0})
        # Burn enough ticks to get >=2 swaps so we know we're past the
        # initial seed-only promotion.
        for tick in range(30):
            runner.next_action({"step": tick})
            time.sleep(0.001)
        assert runner.stats.swaps >= 2, (
            f"need >=2 swaps to verify auto-d, got swaps={runner.stats.swaps}")
        # Per-call latency was ~5 ms => d ~ 5 (not the seed's 20).
        assert 3 <= runner.stats.last_splice_d <= 8, (
            f"auto-d should follow real ~5 ms latency, got d="
            f"{runner.stats.last_splice_d} (likely regressed to seed=20 "
            f"or stuck at 0)")
    finally:
        runner.close()


def test_async_runner_splice_d_clipped_to_horizon():
    """If d >= horizon (inference slower than chunk duration), the runner
    must clip d to horizon-1 so we still make forward progress.
    """
    def slow_policy(obs):
        return np.arange(3, dtype=np.float32)[:, None] + 10.0

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(slow_policy),
        RTCConfig(
            target_hz=100.0,
            action_horizon=3,
            start_next_at=0,
            inference_delay_steps=100,
        ),
    )
    try:
        runner.reset({"step": 0})
        for _ in range(20):
            runner.next_action({"step": 1})
            time.sleep(0.0005)
        assert runner.stats.last_splice_d == 2
    finally:
        runner.close()


def test_rtc_config_rejects_negative_inference_delay():
    with pytest.raises(ValueError, match="inference_delay_steps"):
        RTCConfig(target_hz=10.0, action_horizon=4, inference_delay_steps=-1)


def test_rtc_config_rejects_out_of_range_ema_alpha():
    with pytest.raises(ValueError, match="latency_ema_alpha"):
        RTCConfig(target_hz=10.0, latency_ema_alpha=0.0)
    with pytest.raises(ValueError, match="latency_ema_alpha"):
        RTCConfig(target_hz=10.0, latency_ema_alpha=1.5)


def test_callable_policy_adapter_lifts_meta_keys():
    """Adapter should pull ``meta_keys`` out of dict responses so the
    runner can stash auxiliary fields (e.g. ``_rtc_chunk_model_space``)
    on :class:`ChunkResult`. Backends that don't return the key get
    a silently-missing entry; never a KeyError.
    """
    response = {
        "actions": np.ones((4, 3), dtype=np.float32),
        "_rtc_chunk_model_space": np.full((4, 3), 0.5, dtype=np.float32),
    }
    adapter = CallablePolicyAdapter(
        lambda obs: response,
        meta_keys=("_rtc_chunk_model_space", "missing_key"),
    )
    actions, meta = adapter.infer_actions_with_meta({"step": 0})

    assert actions.shape == (4, 3)
    assert np.array_equal(meta["_rtc_chunk_model_space"],
                          response["_rtc_chunk_model_space"])
    assert "missing_key" not in meta


def test_async_runner_caches_model_space_chunk_from_adapter():
    """When the policy returns ``_rtc_chunk_model_space``, AsyncChunkRunner
    must lift it onto ``ChunkResult.chunk_model_space``. Without that,
    the prefix-freeze submission path has nothing to send back as
    ``_rtc_prev_chunk``.
    """
    def policy(obs):
        return {
            "actions": np.arange(8, dtype=np.float32)[:, None] + 100.0,
            "_rtc_chunk_model_space": np.arange(8, dtype=np.float32)[:, None],
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(target_hz=1000.0, action_horizon=8, start_next_at=2),
    )
    try:
        runner.reset({"step": 0})
        assert runner._current is not None
        cms = runner._current.chunk_model_space
        assert cms is not None
        assert cms.shape == (8, 1)
        # Actions are the offset version (+100); model-space is raw.
        np.testing.assert_array_equal(
            cms[:, 0], np.arange(8, dtype=np.float32))
    finally:
        runner.close()


def test_async_runner_attaches_prefix_when_freeze_enabled():
    """With ``enable_prefix_freeze=True``, every async submission should
    carry ``_rtc_prev_chunk`` (sliced from the cached model-space chunk
    starting at the current consume index) and ``_rtc_inference_delay``
    (the EMA-predicted d). Without freeze enabled, neither key appears.
    """
    submitted_obs: list[dict] = []

    def policy(obs):
        submitted_obs.append(dict(obs))
        return {
            "actions": np.arange(10, dtype=np.float32)[:, None],
            "_rtc_chunk_model_space": np.arange(
                10, dtype=np.float32)[:, None] * 0.1,
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=10,
            start_next_at=0,
            auto_inference_delay=True,
            enable_prefix_freeze=True,
            prefix_freeze_margin_steps=1,
        ),
    )
    try:
        runner.reset({"step": "init"})
        for tick in range(20):
            runner.next_action({"step": tick})
            time.sleep(0.001)
        # The very first submission is the reset() call which has no
        # cached prefix; subsequent submissions should carry it.
        assert len(submitted_obs) >= 2
        later = submitted_obs[-1]
        assert "_rtc_prev_chunk" in later, (
            "expected _rtc_prev_chunk on later submissions when "
            "enable_prefix_freeze=True")
        assert "_rtc_inference_delay" in later
        d = later["_rtc_inference_delay"]
        prev = later["_rtc_prev_chunk"]
        assert isinstance(d, int) and d >= 1
        # Phase 6 (G11): async submission carries the entire unconsumed
        # tail of the prev chunk (not just the d_pred anchor positions)
        # so the server's soft-guidance kernel has prev values across
        # the merge window. The length is therefore >= d, bounded by
        # the chunk size (10 here).
        assert prev.shape[0] >= d
        assert prev.shape[1] == 1
        assert prev.dtype == np.float32
    finally:
        runner.close()


def test_async_runner_skips_prefix_when_freeze_disabled():
    """Without ``enable_prefix_freeze``, even backends that return
    ``_rtc_chunk_model_space`` should not see ``_rtc_*`` keys on the
    submitted observation. Guards the default (legacy) path.
    """
    submitted_obs: list[dict] = []

    def policy(obs):
        submitted_obs.append(dict(obs))
        return {
            "actions": np.arange(10, dtype=np.float32)[:, None],
            "_rtc_chunk_model_space": np.arange(
                10, dtype=np.float32)[:, None],
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=10,
            start_next_at=0,
            auto_inference_delay=True,
        ),
    )
    try:
        runner.reset({"step": "init"})
        for tick in range(20):
            runner.next_action({"step": tick})
            time.sleep(0.001)
        assert len(submitted_obs) >= 2
        for obs in submitted_obs:
            assert "_rtc_prev_chunk" not in obs
            assert "_rtc_inference_delay" not in obs
    finally:
        runner.close()


def test_async_runner_prefix_freeze_caps_at_max_steps():
    """``prefix_freeze_max_steps`` must clip the freeze prefix length
    even when EMA latency would predict longer. Prevents the model from
    being asked to freeze more than half of the chunk by default.
    """
    submitted_obs: list[dict] = []

    def slow_policy(obs):
        submitted_obs.append(dict(obs))
        time.sleep(0.030)  # 30 ms => 30 ticks at 1000 Hz
        return {
            "actions": np.arange(10, dtype=np.float32)[:, None],
            "_rtc_chunk_model_space": np.arange(
                10, dtype=np.float32)[:, None],
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            slow_policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=10,
            start_next_at=0,
            auto_inference_delay=True,
            enable_prefix_freeze=True,
            prefix_freeze_margin_steps=0,
            prefix_freeze_max_steps=3,
        ),
    )
    try:
        runner.reset({"step": "init"})
        for tick in range(20):
            runner.next_action({"step": tick})
            time.sleep(0.001)
        assert any(
            "_rtc_prev_chunk" in obs for obs in submitted_obs), (
            "no submission carried a freeze prefix")
        for obs in submitted_obs:
            if "_rtc_inference_delay" in obs:
                # ``prefix_freeze_max_steps`` caps the ANCHOR length
                # (d_pred = positions hard-pinned to the prefix in
                # the merge kernel). Phase 6 (G11): the
                # ``_rtc_prev_chunk`` payload itself is the full
                # unconsumed tail (variable length, bounded by the
                # chunk size), independent of the cap.
                assert obs["_rtc_inference_delay"] <= 3
                assert obs["_rtc_prev_chunk"].shape[0] <= 10
    finally:
        runner.close()


def test_sync_block_runner_prefix_from_chunk_tail():
    """Sync block mode (mode 1 + prefix_freeze) must take the prefix from
    the END of the just-completed chunk, not project forward from idx.

    At exhaustion ``self._idx == horizon``, so there are no future
    actions of the current chunk to project as the prefix. The
    correct behaviour is to send the last ``d`` actions of the
    cached ``chunk_model_space`` so the server can constrain the
    new chunk's first ``d`` positions to match what the controller
    is currently tracking → continuous inter-chunk boundary.

    Regression test for the sync-exhaustion case added alongside the
    chocolate_bars mode-1 boundary fix on 2026-05.
    """
    submitted_obs: list[dict] = []
    # Two distinct chunks so we can tell which one the prefix was sliced
    # from: chunk0 is values 0..9, chunk1 is values 100..109. The prefix
    # attached to the SECOND submission must come from chunk0's tail.
    chunks_returned = [
        np.arange(10, dtype=np.float32)[:, None],
        (np.arange(10, dtype=np.float32) + 100.0)[:, None],
    ]
    call_idx = [0]

    def sync_policy(obs):
        submitted_obs.append(dict(obs))
        idx = min(call_idx[0], len(chunks_returned) - 1)
        call_idx[0] += 1
        time.sleep(0.005)  # ~5 ticks of "inference" at 1000 Hz
        actions = chunks_returned[idx]
        return {
            "actions": actions,
            "_rtc_chunk_model_space": actions * 0.5,  # arbitrary mapping
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            sync_policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=10,
            start_next_at=10,        # don't pre-fire (sync)
            miss_policy="block",     # block at exhaustion
            auto_inference_delay=True,
            enable_prefix_freeze=True,
            prefix_freeze_margin_steps=0,
        ),
    )
    try:
        runner.reset({"step": "init"})
        # Play 11 ticks: 10 to exhaust chunk0, then 1 more which
        # triggers the block-and-replan boundary.
        for tick in range(11):
            runner.next_action({"step": tick})
        # First submission (from reset) has no prefix; the second
        # submission fires at exhaustion of chunk0 and MUST carry a
        # prefix sliced from the TAIL of chunk0's model-space form
        # (i.e. values * 0.5 of indices [10-d : 10]).
        assert len(submitted_obs) >= 2, (
            f"expected 2+ submissions, got {len(submitted_obs)}")
        boundary = submitted_obs[1]
        assert "_rtc_prev_chunk" in boundary, (
            "sync exhaustion submission must carry _rtc_prev_chunk; "
            "got keys=" + str(list(boundary.keys())))
        assert "_rtc_inference_delay" in boundary
        d = boundary["_rtc_inference_delay"]
        prev = boundary["_rtc_prev_chunk"]
        assert d >= 1
        assert prev.shape == (d, 1)
        # chunk0's model_space is np.arange(10) * 0.5 = [0, 0.5, ..., 4.5]
        # Tail of length d should be [10-d, 11-d, ..., 9] * 0.5.
        expected_tail = (np.arange(10 - d, 10, dtype=np.float32) * 0.5
                         )[:, None]
        np.testing.assert_array_equal(prev, expected_tail), (
            f"sync prefix must equal chunk0's last {d} model-space "
            f"actions; expected {expected_tail.ravel()}, "
            f"got {prev.ravel()}")
    finally:
        runner.close()


def test_sync_block_splice_lands_at_d_pred_exactly():
    """Regression: in sync block mode with prefix-freeze, the splice
    index must equal ``d_pred_used`` EXACTLY, regardless of measured
    inference latency.

    Two failure modes are ruled out:

    1. Splice INSIDE the frozen region (d < d_pred). The frozen
       positions are replays of the previous chunk's tail; serving
       them is a backward-time jump (the original G9 bug).

    2. Splice PAST the frozen region (d > d_pred). The frozen
       region only constrains ``chunk_B[0..d_pred-1]`` to match the
       previous chunk's tail. ``chunk_B[d_pred..49]`` are free
       predictions with no continuity guarantee against
       ``chunk_A[49]``. Splicing at ``measured_d > d_pred`` (e.g.
       because of a slow pipeline rebuild) produces an arbitrarily
       large jump at the boundary — the chocolate_bars 0.56 rad
       symptom observed 2026-05.

    In sync mode no control ticks elapse during inference (loop is
    blocked on the result), so ``measured_d`` is semantically
    meaningless for the splice. Only ``d_pred_used`` matters.
    """
    chunks_returned = [
        np.arange(20, dtype=np.float32)[:, None],
        (np.arange(20, dtype=np.float32) + 100.0)[:, None],
    ]
    call_idx = [0]

    def policy(obs):
        idx = min(call_idx[0], len(chunks_returned) - 1)
        call_idx[0] += 1
        # The SECOND inference simulates a slow pipeline rebuild:
        # 50ms at 1000Hz = measured_d would be 50 if used. With
        # the sync-mode splice fix, d_pred=10 is what gets used.
        if idx == 1:
            time.sleep(0.050)
        else:
            time.sleep(0.001)
        actions = chunks_returned[idx]
        return {
            "actions": actions,
            "_rtc_chunk_model_space": actions,
        }

    runner = AsyncChunkRunner(
        CallablePolicyAdapter(
            policy, meta_keys=("_rtc_chunk_model_space",)),
        RTCConfig(
            target_hz=1000.0,
            action_horizon=20,
            start_next_at=20,        # sync (block at exhaustion)
            miss_policy="block",
            auto_inference_delay=True,
            enable_prefix_freeze=True,
            # margin=8 → d_pred = 10 in sync mode. Measured latency
            # for the boundary inference is 50ms × 1000Hz = 50 ticks,
            # which would (under the broken max() splice) become the
            # splice index. With the fix, splice = d_pred = 10.
            prefix_freeze_margin_steps=8,
        ),
    )
    try:
        runner.reset({"step": "init"})
        for tick in range(20):
            runner.next_action({"step": tick})
        # Boundary tick: triggers exhaustion -> block on slow inference.
        boundary_action = runner.next_action({"step": 20})
        # Splice must land at chunk1[d_pred=10] = 110.0 exactly.
        # If broken (splice at measured_d=50), would clamp to
        # horizon-1=19 -> chunk1[19] = 119 (or whatever).
        assert boundary_action[0] == 110.0, (
            f"sync splice should land at d_pred=10 exactly -> "
            f"chunk1[10]=110.0, got {boundary_action[0]}. "
            f"With measured_d=50 and the broken max() splice this "
            f"would land at chunk1[19]=119 (clamped to horizon-1).")
    finally:
        runner.close()
