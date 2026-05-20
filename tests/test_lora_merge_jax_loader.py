"""Unit tests for _maybe_merge_lora in flash_rt.frontends.jax.pi05_rtx.

These tests run on synthetic numpy data so they do not require a real
Orbax checkpoint or GPU — they verify the numerical correctness of the
merge math and the key-naming detector for both openpi LoRA patterns
(Einsum and FeedForward).

Run with::

    pytest tests/test_lora_merge_jax_loader.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


def _try_import():
    """Skip the whole module if the jax frontend can't be imported (e.g. no
    torch on this host). The merge function itself is pure numpy, but it
    lives in a module that imports torch."""
    try:
        from flash_rt.frontends.jax.pi05_rtx import (
            _maybe_merge_lora,
            _resolve_lora_pair,
        )
        return _maybe_merge_lora, _resolve_lora_pair
    except Exception as e:  # pragma: no cover
        pytest.skip(f"frontends.jax.pi05_rtx not importable on this host: {e}")


def test_no_lora_keys_is_noop():
    _maybe_merge_lora, _ = _try_import()
    raw = {
        "PaliGemma.llm.layers.attn.q_einsum.w": np.ones((2, 4, 8), np.float32),
    }
    snapshot = {k: v.copy() for k, v in raw.items()}
    out = _maybe_merge_lora(raw)
    assert out is raw
    assert set(out.keys()) == set(snapshot.keys())
    for k in out:
        np.testing.assert_array_equal(out[k], snapshot[k])


def test_einsum_pattern_resolution():
    _maybe_merge_lora, _resolve = _try_import()
    raw = {
        "PaliGemma.llm.layers.attn.q_einsum.w": np.zeros((2, 4, 8), np.float32),
        "PaliGemma.llm.layers.attn.q_einsum.lora_a": np.zeros((2, 4, 3), np.float32),
        "PaliGemma.llm.layers.attn.q_einsum.lora_b": np.zeros((2, 3, 8), np.float32),
    }
    pair = _resolve("PaliGemma.llm.layers.attn.q_einsum.lora_a", raw)
    assert pair == (
        "PaliGemma.llm.layers.attn.q_einsum.w",
        "PaliGemma.llm.layers.attn.q_einsum.lora_b",
    )


def test_feedforward_pattern_resolution():
    _maybe_merge_lora, _resolve = _try_import()
    raw = {
        "PaliGemma.llm.layers.mlp.gating_einsum": np.zeros((2, 4, 8), np.float32),
        "PaliGemma.llm.layers.mlp.gating_einsum_lora_a": np.zeros((2, 4, 3), np.float32),
        "PaliGemma.llm.layers.mlp.gating_einsum_lora_b": np.zeros((2, 3, 8), np.float32),
    }
    pair = _resolve("PaliGemma.llm.layers.mlp.gating_einsum_lora_a", raw)
    assert pair == (
        "PaliGemma.llm.layers.mlp.gating_einsum",
        "PaliGemma.llm.layers.mlp.gating_einsum_lora_b",
    )


def test_merge_math_matches_jax_einsum():
    """Numerical-correctness check: result must equal w + scaling * (la @ lb).
    This is the same formula openpi's diag_lora_merge_exact.py verified
    against the JAX runtime path."""
    _maybe_merge_lora, _ = _try_import()
    rng = np.random.default_rng(7)
    # Single-layer Einsum shape (8 heads, in=64, out=32) with rank=4.
    H, D, O, R = 8, 64, 32, 4
    w = rng.standard_normal((H, D, O), dtype=np.float32)
    la = rng.standard_normal((H, D, R), dtype=np.float32) * 0.01
    lb = rng.standard_normal((H, R, O), dtype=np.float32) * 0.01

    raw = {
        "X.w": w.copy(),
        "X.lora_a": la.copy(),
        "X.lora_b": lb.copy(),
    }
    _maybe_merge_lora(raw, scaling=1.0)
    # After merge: only the base key remains.
    assert "X.lora_a" not in raw
    assert "X.lora_b" not in raw
    assert "X.w" in raw

    expected = w + 1.0 * np.matmul(la, lb)
    np.testing.assert_allclose(raw["X.w"], expected, rtol=1e-6, atol=1e-6)


def test_merge_with_non_unit_scaling():
    _maybe_merge_lora, _ = _try_import()
    rng = np.random.default_rng(2)
    w = rng.standard_normal((2, 4, 8), dtype=np.float32)
    la = rng.standard_normal((2, 4, 2), dtype=np.float32)
    lb = rng.standard_normal((2, 2, 8), dtype=np.float32)
    raw = {"X.w": w.copy(), "X.lora_a": la.copy(), "X.lora_b": lb.copy()}

    _maybe_merge_lora(raw, scaling=0.5)
    expected = w + 0.5 * np.matmul(la, lb)
    np.testing.assert_allclose(raw["X.w"], expected, rtol=1e-6, atol=1e-6)


def test_merge_handles_multiple_pairs():
    _maybe_merge_lora, _ = _try_import()
    rng = np.random.default_rng(3)
    raw = {}
    expected = {}
    for prefix in ["A.q.w", "A.kv.w", "A.mlp.gating_einsum"]:
        w = rng.standard_normal((2, 4, 8), dtype=np.float32)
        la = rng.standard_normal((2, 4, 2), dtype=np.float32) * 0.1
        lb = rng.standard_normal((2, 2, 8), dtype=np.float32) * 0.1
        raw[prefix] = w.copy()
        if prefix.endswith(".w"):
            la_key = prefix[:-2] + ".lora_a"
            lb_key = prefix[:-2] + ".lora_b"
        else:
            la_key = prefix + "_lora_a"
            lb_key = prefix + "_lora_b"
        raw[la_key] = la.copy()
        raw[lb_key] = lb.copy()
        expected[prefix] = w + np.matmul(la, lb)

    _maybe_merge_lora(raw, scaling=1.0)

    # All LoRA entries should be gone.
    assert not any(k.endswith(".lora_a") for k in raw)
    assert not any(k.endswith(".lora_b") for k in raw)
    assert not any(k.endswith("_lora_a") for k in raw)
    assert not any(k.endswith("_lora_b") for k in raw)

    for k, v in expected.items():
        np.testing.assert_allclose(raw[k], v, rtol=1e-6, atol=1e-6,
                                   err_msg=f"mismatch at {k}")


def test_orphan_lora_a_skipped():
    """If lora_a exists but lora_b doesn't, skip and warn (do not crash)."""
    _maybe_merge_lora, _ = _try_import()
    raw = {
        "X.w": np.zeros((2, 4, 8), np.float32),
        "X.lora_a": np.zeros((2, 4, 2), np.float32),
        # X.lora_b deliberately missing
    }
    _maybe_merge_lora(raw, scaling=1.0)
    # Function should not crash, and the orphan lora_a stays (so the
    # downstream loader will fail loudly with a "unexpected key" error
    # rather than silently producing wrong weights).
    assert "X.lora_a" in raw


def test_idempotent_double_call():
    """Calling _maybe_merge_lora twice should be a no-op the second time."""
    _maybe_merge_lora, _ = _try_import()
    rng = np.random.default_rng(11)
    w = rng.standard_normal((2, 4, 8), dtype=np.float32)
    la = rng.standard_normal((2, 4, 2), dtype=np.float32)
    lb = rng.standard_normal((2, 2, 8), dtype=np.float32)
    raw = {"X.w": w.copy(), "X.lora_a": la.copy(), "X.lora_b": lb.copy()}

    _maybe_merge_lora(raw, scaling=1.0)
    after_first = raw["X.w"].copy()
    _maybe_merge_lora(raw, scaling=1.0)   # no LoRA keys → no-op
    np.testing.assert_array_equal(raw["X.w"], after_first)
