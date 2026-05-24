"""Smoke + unit tests for :mod:`flash_rt.serving.chunked_websocket_client`.

The chunked client is mostly a thin shim over
:class:`flash_rt.runtime.rtc.AsyncChunkRunner`; the heavy logic is
covered in ``tests/test_rtc_lite.py``. These tests pin the mode-to-config
mapping so a refactor that quietly changes the production default does
not slip through review.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from flash_rt.serving.chunked_websocket_client import (
    DEFAULT_MODE,
    MODE_DESCRIPTIONS,
    VALID_MODES,
    ChunkedWebsocketClient,
    _build_config_for_mode,
)


class _FakeServerMetaPolicy:
    """Stand-in for ``openpi_client.WebsocketClientPolicy``.

    Returns deterministic action chunks and a static server metadata dict
    so the constructor's auto-chunk-len logic exercises the metadata
    code-path without a real socket.
    """

    def __init__(self, chunk_size: int = 10, action_dim: int = 4) -> None:
        self._chunk_size = chunk_size
        self._action_dim = action_dim
        self._n_calls = 0

    def get_server_metadata(self) -> dict:
        return {"chunk_size": self._chunk_size, "robot_action_dim": self._action_dim}

    def infer(self, obs: dict) -> dict:
        self._n_calls += 1
        actions = np.full(
            (self._chunk_size, self._action_dim),
            float(self._n_calls),
            dtype=np.float32,
        )
        return {"actions": actions}


def test_default_mode_is_three():
    assert DEFAULT_MODE == 3
    assert DEFAULT_MODE in VALID_MODES
    assert "default" in MODE_DESCRIPTIONS[DEFAULT_MODE].lower()


def test_mode_1_is_sync_truncate_replan_k25_with_freeze_by_default():
    """Mode 1 must replicate the historical openpi ``ActionChunkBroker``.

    The openpi broker that SparkJAX used pre-2026-05-22 hardcoded
    ``action_horizon=25`` ("ALOHA default") and only served the first
    25 actions of each 50-action chunk before synchronously
    re-inferring. On-robot A/B testing on the pi0.5 chocolate-bars
    checkpoint showed that:

      * Playing the full 50-action chunk back-to-back (the previous
        revision of this function) regressed grasp success: the model
        committed the robot to a stale forecast of the manipulation
        phase before any fresh visual observation could correct it.

      * Truncating to k=5 was catastrophic (see git history): plans
        disagreed sharply on first-action commands.

    k=25 is the empirical sweet spot. This test pins mode 1's config
    to that baseline regardless of the negotiated chunk_len (clamped
    to chunk_len if smaller).
    """
    cfg = _build_config_for_mode(1, chunk_len=50, target_hz=50.0,
                                 expected_latency_ms=200.0)
    assert cfg.miss_policy == "block"
    assert cfg.action_horizon == 25, (
        "mode 1 must replicate the openpi ALOHA-default execute-horizon "
        f"of 25; got action_horizon={cfg.action_horizon}")
    assert cfg.start_next_at == 25, (
        "mode 1's start_next_at must match action_horizon so the sync "
        f"block fires exactly at execute-horizon; got start_next_at="
        f"{cfg.start_next_at}")
    assert cfg.blend_steps == 5, (
        "mode 1 must ramp the first 5 actions of each new chunk to "
        "absorb chunk-boundary discontinuities (absolute-space gripper "
        "joints can step-jump >1 rad between chunks; without blending "
        "SparkJAX's per-step safety trips). Got "
        f"blend_steps={cfg.blend_steps}")
    assert cfg.enable_prefix_freeze is True, (
        "mode 1 must enable prefix_freeze when the caller leaves it "
        "default-True (used for A/B vs. the no-freeze baseline)")
    assert cfg.auto_inference_delay is True, (
        "mode 1 + prefix_freeze must use measured latency to size the "
        "freeze window; auto_inference_delay=True is required")


def test_mode_1_freeze_can_be_disabled():
    """``prefix_freeze=False`` must produce the freeze-off baseline.

    Used for A/B comparison to attribute smoothness changes to the
    freeze and nothing else. SparkJAX's ROS param defaults to False
    so this is the production-default configuration. Seam blending
    stays on either way — it's required by the absolute-space gripper
    safety constraint and is orthogonal to the prefix-freeze choice.
    """
    cfg = _build_config_for_mode(1, chunk_len=50, target_hz=50.0,
                                 expected_latency_ms=200.0,
                                 prefix_freeze=False)
    assert cfg.miss_policy == "block"
    assert cfg.action_horizon == 25
    assert cfg.start_next_at == 25
    assert cfg.blend_steps == 5, (
        "seam blending is independent of prefix_freeze; mode 1 keeps "
        "blend_steps=5 even when prefix_freeze is off")
    assert cfg.enable_prefix_freeze is False
    assert cfg.auto_inference_delay is False


def test_mode_1_execute_horizon_clamped_to_small_chunk_len():
    """If the server publishes a smaller chunk than 25, fall back to it.

    Prevents action_horizon > chunk_len, which the RTC runner would
    silently clamp anyway but is clearer to handle at config time so
    log lines reflect the actual horizon. The blend window is also
    clamped so it can't exceed the execute horizon — a 4-step blend
    inside a 4-tick chunk is the same as "blend the whole chunk".
    """
    cfg = _build_config_for_mode(1, chunk_len=10, target_hz=50.0,
                                 expected_latency_ms=200.0,
                                 prefix_freeze=False)
    assert cfg.action_horizon == 10
    assert cfg.start_next_at == 10
    assert cfg.blend_steps == 5, (
        "blend_steps fits inside execute_horizon=10 (=min(5,10)), so "
        "the 5-step blend is preserved unmodified")

    cfg_tiny = _build_config_for_mode(1, chunk_len=4, target_hz=50.0,
                                      expected_latency_ms=200.0,
                                      prefix_freeze=False)
    assert cfg_tiny.action_horizon == 4
    assert cfg_tiny.blend_steps == 4, (
        "when execute_horizon=4 the blend must clamp to 4 (= the whole "
        f"chunk); got blend_steps={cfg_tiny.blend_steps}")


@pytest.mark.parametrize("mode,expected_blend", [(2, 0), (3, 3), (4, 5)])
def test_async_modes_use_rtc_paper_scheduling(mode, expected_blend):
    cfg = _build_config_for_mode(mode, chunk_len=50, target_hz=50.0,
                                 expected_latency_ms=200.0)
    assert cfg.miss_policy == "hold_last"
    assert cfg.action_horizon == 50
    assert cfg.start_next_at == 0, (
        f"async mode {mode} must fire ASAP (start_next_at=0), got "
        f"{cfg.start_next_at}"
    )
    assert cfg.auto_inference_delay is True
    assert cfg.inference_delay_steps == 10, (
        "expected d_seed = ceil(200 ms * 50 Hz / 1000) = 10, got "
        f"{cfg.inference_delay_steps}"
    )
    assert cfg.blend_steps == expected_blend


def test_build_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="blending_mode"):
        _build_config_for_mode(6, chunk_len=10, target_hz=50.0,
                               expected_latency_ms=200.0)


def test_build_config_mode_5_enables_prefix_freeze():
    """Mode 5 must turn on server-side prefix-freeze AND keep a 5-step
    client-side seam blend as a safety net for the case where the
    realised inference latency overshoots ``d_pred`` (most common
    cause on Spark: a Pi05Pipeline rebuild blowing past the EMA
    latency by 4x). Inside the frozen region the blend is a near
    no-op; outside it, it caps the per-step joint delta below
    SparkJAX's 0.5 rad/step arm-joint safety limit. See
    ``_build_config_for_mode``'s mode-5 comment for the analysis.

    Also asserts the ``prefix_freeze_max_steps`` cap is set to
    ``horizon // 4``. The default ``horizon // 2`` cap pins the
    freeze coverage to half the chunk under EMA-poisoning from
    pipeline rebuilds, leaving only 500 ms of play time at
    horizon=50/50 Hz — exactly the SparkJAX deadline-miss window.
    """
    cfg = _build_config_for_mode(5, chunk_len=50, target_hz=50.0,
                                 expected_latency_ms=200.0)
    assert cfg.start_next_at == 0
    assert cfg.auto_inference_delay is True
    assert cfg.enable_prefix_freeze is True
    assert cfg.blend_steps == 5, (
        "mode 5 keeps a 5-step seam blend as a safety net for "
        "pipeline-rebuild latency spikes that push d_actual past "
        "d_pred; inside the frozen region the blend is a near no-op")
    assert cfg.prefix_freeze_max_steps == 12, (
        "mode 5 caps prefix-freeze coverage at horizon//4 so each "
        "chunk has ~3/4 horizon of free play time, absorbing typical "
        "Pi05Pipeline rebuild latencies without deadline-missing")
    assert cfg.miss_policy == "block", (
        "mode 5 must override the async miss_policy to 'block' so "
        "deadline misses pause the robot briefly instead of "
        "repeating the last action 25 times into SparkJAX's safety "
        "abort threshold")

    # Spot-check the floor: tiny chunks must still get at least 1 step.
    cfg_small = _build_config_for_mode(5, chunk_len=2, target_hz=50.0,
                                       expected_latency_ms=200.0)
    assert cfg_small.prefix_freeze_max_steps == 1


def test_chunked_client_default_construction_uses_mode_3():
    policy = _FakeServerMetaPolicy(chunk_size=10, action_dim=4)
    client = ChunkedWebsocketClient(policy, target_hz=50.0,
                                    expected_latency_ms=200.0)
    try:
        assert client.blending_mode == 3
        assert client.chunk_len == 10
        # The runner config exposes the mode-3 knobs:
        cfg = client._runner.config  # type: ignore[attr-defined]
        assert cfg.start_next_at == 0
        assert cfg.auto_inference_delay is True
        assert cfg.blend_steps == 3
    finally:
        client.close()


def test_chunked_client_serves_and_records_latency():
    policy = _FakeServerMetaPolicy(chunk_size=8, action_dim=3)
    client = ChunkedWebsocketClient(policy, target_hz=200.0,
                                    expected_latency_ms=50.0,
                                    blending_mode=2)  # no blend; values stay raw
    try:
        a0 = client.next_action({"step": 0})
        assert a0.shape == (3,)
        assert float(a0[0]) == 1.0, (
            "first chunk should be filled with 1.0 (n_calls=1)"
        )
        for tick in range(20):
            client.next_action({"step": tick + 1})
            time.sleep(0.002)
        stats = client.stats
        assert stats.actions_served == 21
        assert stats.chunks_started >= 1
        assert stats.ema_latency_s > 0.0
    finally:
        client.close()


def test_chunked_client_runtime_mode_switch():
    policy = _FakeServerMetaPolicy(chunk_size=8, action_dim=2)
    client = ChunkedWebsocketClient(policy, target_hz=200.0,
                                    expected_latency_ms=50.0,
                                    blending_mode=2)
    try:
        _ = client.next_action({"step": 0})
        client.set_blending_mode(4)
        assert client.blending_mode == 4
        cfg = client._runner.config  # type: ignore[attr-defined]
        assert cfg.blend_steps == 5
        client.set_blending_mode(4)
        assert client.blending_mode == 4
    finally:
        client.close()
