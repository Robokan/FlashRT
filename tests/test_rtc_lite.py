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
