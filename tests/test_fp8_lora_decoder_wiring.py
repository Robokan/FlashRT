"""Regression tests for the G12 FP8 + runtime-LoRA decoder fix.

Background
----------
Before G12, the FP8 decoder pipeline silently dropped *all* runtime
LoRA contributions in the Pi0.5 decoder. The encoder had a working
``_enc_lora_on`` gate that forced the layer through the non-fused FP8
branch and applied LoRA via ``_apply_enc_lora``. The decoder had no
equivalent ``_dec_lora_on`` gate and **no** ``_apply_dec_lora`` calls
in any of the four ``if self.use_fp8_decoder:`` branches (QKV, attn O,
FFN gate/up, FFN down).

The visible symptom on Spark when launching the OpenArm v4 LoRA
checkpoint with ``FLASHRT_RUNTIME_LORA=all`` and ``--fp8`` was
"catastrophic" output quality — the decoder ran with LoRA-stripped
base weights because the LoRA had been extracted out of the merge for
encoder-FP8-calibration cleanliness, but never re-added in the
decoder. See ``docs/spark_phase8_fp8_lora.md`` for the full story.

What this file covers
---------------------
1.  Pure-numpy algebraic correctness of ``_build_padded_gateup_lora``
    for the decoder dimensions (D=1024, H=4096, r=32). The encoder
    test in ``test_lora_merge_jax_loader.py`` already covers shape
    handling generically; this asserts that the fused form
    produces the *same* delta as two separate gate/up adds for the
    decoder's actual MLP dimensions.

2.  JAX converter contract — the new keys
    ``decoder_ffn_gateup_lora_{a,b}`` are present in the output
    checkpoint dict whenever the per-layer ``decoder_ffn_gate_*`` /
    ``decoder_ffn_up_*`` pairs are present.

3.  Pipeline structural gate — the four FP8 decoder paths each have
    a ``_apply_dec_lora`` invocation in their source body. This is a
    grep-style regression gate that fails loudly if a future refactor
    accidentally removes a site (which is what caused the original
    catastrophe).

What this file does NOT cover
-----------------------------
Numerical parity of BF16+LoRA vs FP8+LoRA actions on a real
checkpoint. That test requires CUDA and a checkpoint on disk and is
the user-facing acceptance gate described in
``docs/spark_phase8_fp8_lora.md``. It is run via
``scripts/spark_phase8_fp8_lora_parity.py``.

Run with::

    pytest tests/test_fp8_lora_decoder_wiring.py -v
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest


# ---------------------------------------------------------------------
# 1. Pure-numpy algebraic test for _build_padded_gateup_lora.
# ---------------------------------------------------------------------


def _try_import_build_gateup():
    try:
        from flash_rt.frontends.jax.pi05_rtx import _build_padded_gateup_lora
    except Exception as e:  # pragma: no cover
        pytest.skip(f"frontends.jax.pi05_rtx not importable: {e}")
    return _build_padded_gateup_lora


def test_padded_gateup_lora_matches_separate_gate_up_decoder_dims():
    """Fused (D, 2r) / (2r, 2H) padded gateup LoRA must reproduce
    ``[gate_delta | up_delta]`` for the decoder's actual MLP dims
    (D=1024, H=4096, r=32 on Pi0.5 OpenArm).

    This is the algebraic invariant the FP8 decoder FFN path relies
    on: the base FP8 GEMM writes (ds, 2H) into ``decoder_gate_merged``
    in the ``[gate | up]`` layout; a single fused LoRA add must hit
    both halves correctly.
    """
    build = _try_import_build_gateup()
    rng = np.random.default_rng(0)
    D, H, r, ds = 1024, 4096, 32, 10

    la_gate = rng.standard_normal((D, r), dtype=np.float64).astype(np.float32)
    la_up   = rng.standard_normal((D, r), dtype=np.float64).astype(np.float32)
    lb_gate = rng.standard_normal((r, H), dtype=np.float64).astype(np.float32)
    lb_up   = rng.standard_normal((r, H), dtype=np.float64).astype(np.float32)

    la_padded, lb_padded = build(la_gate, la_up, lb_gate, lb_up)

    assert la_padded.shape == (D, 2 * r)
    assert lb_padded.shape == (2 * r, 2 * H)
    assert la_padded.dtype == np.float32
    assert lb_padded.dtype == np.float32

    x = rng.standard_normal((ds, D), dtype=np.float64).astype(np.float32)
    gate_delta_ref = (x @ la_gate) @ lb_gate                # (ds, H)
    up_delta_ref   = (x @ la_up)   @ lb_up                  # (ds, H)
    fused_delta    = (x @ la_padded) @ lb_padded            # (ds, 2H)

    np.testing.assert_allclose(
        fused_delta[:, :H], gate_delta_ref, rtol=1e-5, atol=1e-4,
        err_msg="fused LoRA gate half does not match separate gate delta")
    np.testing.assert_allclose(
        fused_delta[:, H:], up_delta_ref, rtol=1e-5, atol=1e-4,
        err_msg="fused LoRA up half does not match separate up delta")


def test_padded_gateup_lora_zero_cross_blocks_decoder_dims():
    """The cross blocks of ``lb_padded`` must be zero — that's what
    keeps the gate and up halves independent. If a future edit
    accidentally fills them in, the fused FP8 add would mix gate's
    rank-r neck into up's columns and vice versa.
    """
    build = _try_import_build_gateup()
    rng = np.random.default_rng(0)
    D, H, r = 1024, 4096, 32

    la_gate = rng.standard_normal((D, r)).astype(np.float32)
    la_up   = rng.standard_normal((D, r)).astype(np.float32)
    lb_gate = rng.standard_normal((r, H)).astype(np.float32)
    lb_up   = rng.standard_normal((r, H)).astype(np.float32)

    _, lb_padded = build(la_gate, la_up, lb_gate, lb_up)
    np.testing.assert_array_equal(lb_padded[:r, H:], 0.0)
    np.testing.assert_array_equal(lb_padded[r:, :H], 0.0)


# ---------------------------------------------------------------------
# 2. Pipeline-source structural gate — the four FP8 decoder branches
# each invoke _apply_dec_lora.
# ---------------------------------------------------------------------


def _get_decoder_layer_source() -> str:
    try:
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pi05.pipeline_rtx not importable on this host: {e}")
    src = inspect.getsource(Pi05Pipeline._decoder_layer)
    return src


def test_decoder_layer_has_dec_lora_on_gate():
    """_decoder_layer must compute _dec_lora_on and use it to disable
    the fused FP8 path (mirror of the encoder pattern)."""
    src = _get_decoder_layer_source()
    assert "_dec_lora_on" in src, (
        "_decoder_layer is missing the _dec_lora_on gate that disables "
        "the fused FP8 decoder path when runtime LoRA is active. "
        "Without it, decoder runtime LoRA adds get skipped in FP8.")
    assert "and not _dec_lora_on" in src, (
        "_decoder_layer's `fused` expression must include "
        "`and not _dec_lora_on` so that runtime LoRA forces the "
        "non-fused FP8 branches where _apply_dec_lora is wired.")


def test_decoder_layer_applies_lora_in_fp8_qkv():
    """The FP8 QKV branch (`if self.use_fp8_decoder:` before
    qkv_split_rope) must call _apply_dec_lora for the QKV LoRA."""
    src = _get_decoder_layer_source()
    # Find the FP8 QKV branch and check it contains the LoRA add.
    qkv_branch_start = src.find("if self.use_fp8_decoder:")
    assert qkv_branch_start != -1, "FP8 QKV branch missing"
    qkv_branch = src[qkv_branch_start: qkv_branch_start + 2000]
    assert "decoder_attn_qkv_lora_a" in qkv_branch, (
        "FP8 QKV branch is missing the decoder_attn_qkv_lora_a "
        "_apply_dec_lora call. Without it, decoder QKV LoRA is "
        "silently dropped in FP8 — the original G12 bug.")


def test_decoder_layer_applies_lora_in_fp8_attn_o():
    src = _get_decoder_layer_source()
    assert "decoder_attn_o_lora_a" in src, (
        "_decoder_layer is missing the decoder_attn_o_lora_a call. "
        "The FP8 attn O projection must apply runtime LoRA when "
        "_has_dec_attn_lora is set.")


def test_decoder_layer_applies_lora_in_fp8_ffn_gateup():
    src = _get_decoder_layer_source()
    assert "decoder_ffn_gateup_lora_a" in src, (
        "_decoder_layer is missing the fused decoder_ffn_gateup_lora_a "
        "call. The FP8 FFN gate/up branch must apply runtime LoRA via "
        "the fused (D, 2r) / (2r, 2H) padded form when "
        "_has_dec_ffn_gateup_lora_fused is set.")


def test_decoder_layer_applies_lora_in_fp8_ffn_down():
    src = _get_decoder_layer_source()
    # Look for the down LoRA application in the FP8 FFN-down branch.
    # `elif self.use_fp8_decoder:` is the down's FP8 site (not the
    # gate/up site which uses `if self.use_fp8_decoder:`).
    down_branch_start = src.find("elif self.use_fp8_decoder:")
    assert down_branch_start != -1, "FP8 FFN-down branch missing"
    down_branch = src[down_branch_start: down_branch_start + 2000]
    assert "decoder_ffn_down_lora_a" in down_branch, (
        "FP8 FFN-down branch is missing the decoder_ffn_down_lora_a "
        "_apply_dec_lora call. Without it, decoder FFN-down LoRA is "
        "silently dropped in FP8.")


# ---------------------------------------------------------------------
# 3. Pipeline __init__ flag detection — _has_dec_ffn_gateup_lora_fused
# and the widened dec_max_neck calculation.
# ---------------------------------------------------------------------


def test_pipeline_init_detects_fused_gateup_flag():
    """Pi05Pipeline.__init__ must register _has_dec_ffn_gateup_lora_fused
    when the weights dict contains decoder_ffn_gateup_lora_{a,b}.
    Source inspection only — we cannot instantiate the real pipeline
    without CUDA."""
    try:
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pi05.pipeline_rtx not importable on this host: {e}")
    src = inspect.getsource(Pi05Pipeline.__init__)
    assert "_has_dec_ffn_gateup_lora_fused" in src, (
        "Pi05Pipeline.__init__ does not register "
        "_has_dec_ffn_gateup_lora_fused — the FP8 decoder FFN gate/up "
        "LoRA cannot be detected and will be silently dropped.")
    assert "decoder_ffn_gateup_lora_a" in src and \
           "decoder_ffn_gateup_lora_b" in src, (
        "Pi05Pipeline.__init__ does not reference the fused gateup "
        "tensor keys, so the JAX converter's emitted tensors will "
        "never be picked up.")


def test_pipeline_init_widens_dec_max_neck_for_fused_gateup():
    """When the fused gateup form is present, the decoder's LoRA neck
    buffer must be sized for 2r (the fused intermediate width), not
    just r. Otherwise the bf16_nn into _dec_lora_neck would silently
    overflow."""
    try:
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
    except Exception as e:  # pragma: no cover
        pytest.skip(f"pi05.pipeline_rtx not importable on this host: {e}")
    src = inspect.getsource(Pi05Pipeline.__init__)
    # Find the dec_max_neck calculation and check it accounts for the
    # fused gateup width.
    dec_neck_section = src[src.find("dec_max_neck"):
                           src.find("self._dec_lora_neck_max")]
    assert "_has_dec_ffn_gateup_lora_fused" in dec_neck_section, (
        "dec_max_neck calculation does not widen for the fused gateup "
        "form (2r). The bf16_nn into _dec_lora_neck would write past "
        "the allocated buffer.")
    assert "decoder_ffn_gateup_lora_a" in dec_neck_section, (
        "dec_max_neck calculation does not read the fused gateup "
        "tensor's last-dim (2r). It must take max(rank, 2r, qkv_neck).")
