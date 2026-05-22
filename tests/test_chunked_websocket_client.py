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


def test_mode_1_is_sync_baseline():
    cfg = _build_config_for_mode(1, chunk_len=50, target_hz=50.0,
                                 expected_latency_ms=200.0)
    assert cfg.miss_policy == "block"
    assert cfg.action_horizon == 5
    assert cfg.start_next_at == 5
    assert cfg.blend_steps == 0
    assert cfg.inference_delay_steps is None
    assert cfg.auto_inference_delay is False


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
        _build_config_for_mode(5, chunk_len=10, target_hz=50.0,
                               expected_latency_ms=200.0)


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
