"""FlashRT -- RTX Pi0.5 JAX frontend.

Loads Pi0.5 Orbax checkpoints (the JAX-native format used by openpi) and
drives the same framework-agnostic ``Pi05Pipeline`` as the torch frontend.

Stage 1 design: this is a **thin shim** over :class:`Pi05TorchFrontendRtx`.
The JAX-specific work is the weight loader (Orbax -> bf16 torch tensors
with the same dict schema as ``convert_pi05_safetensors``). Once those
weights are in the same format, every other line of the frontend -- FP8
quantize, decoder style precompute, FP8 calibration, CUDA Graph capture,
infer -- is shared with the torch path.

This still imports torch; removing that last dependency is on the
roadmap. The current goal is to prove that JAX checkpoints produce the
same outputs as torch checkpoints through the rtx pipeline.

Usage::

    from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx
    pipe = Pi05JaxFrontendRtx("/path/to/pi05_libero", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs])
    out = pipe.infer({"image": img, "wrist_image": wrist})
    actions = out["actions"]   # (chunk_size, 7) numpy
"""

from __future__ import annotations

import logging
import math
import os
import pathlib
from typing import Optional, Union

import ml_dtypes
import numpy as np
import torch

from flash_rt.models.pi05.pipeline_rtx import (
    ACTION_DIM,
    DEC_L,
    ENC_L,
    NUM_STEPS_DEFAULT,
    VIS_L,
)
from flash_rt.frontends.torch.pi05_rtx import (
    CHUNK_SIZE,
    MAX_PROMPT_LEN_DEFAULT,
    Pi05TorchFrontendRtx,
    _interleave_qk,
    _select_fp8_layout,
)
from flash_rt.core.utils.hardware import supports_fp8

logger = logging.getLogger(__name__)

bf16 = torch.bfloat16


# ════════════════════════════════════════════════════════════════════
#   Orbax → bf16 torch dict (rtx schema)
# ════════════════════════════════════════════════════════════════════
#
# This routine produces a dict with the **exact same key names + shapes**
# as ``convert_pi05_safetensors`` so the rest of the rtx torch frontend
# works unchanged. The only difference is the source of the weights.
#
# JAX Orbax stores weights with PaliGemma's flax key names. We:
#  1. Load the Orbax checkpoint via the existing ``_load_orbax`` helper
#     (returns a flat numpy dict, fp32).
#  2. Bit-truncate fp32 → bf16 → fp32. JAX weights are stored as fp32 but
#     production loads them as bf16; truncating up-front guarantees the
#     FP8 quantization scales we compute later match the torch path
#     bit-for-bit.
#  3. Reshape JAX einsum layouts ((num_heads, in_dim, head_dim) etc) into
#     the row-major (in_dim, out_dim) layout the rtx pipeline expects.
#  4. Apply the same encoder RMSNorm fold ``w *= (1 + scale)`` in fp32 to
#     avoid bf16 rounding near -1.0 (the same trap that costs ~10% LIBERO
#     accuracy if missed).
#  5. Apply Q/K head-dim interleave for the fused RoPE kernel.


def _to_bf16_cuda(arr: np.ndarray) -> torch.Tensor:
    """Numpy → contiguous BF16 cuda tensor.

    The numpy array can be fp32, fp16, or ml_dtypes.bfloat16. We go via a
    contiguous fp32 staging step (when needed) so the final ``.to(bf16)``
    cast is well-defined regardless of input dtype.
    """
    if arr.dtype == ml_dtypes.bfloat16:
        # ml_dtypes.bfloat16 → uint16 → torch.uint16 → torch.bfloat16 view
        u16 = np.ascontiguousarray(arr).view(np.uint16)
        t = torch.from_numpy(u16).view(bf16)
        return t.to("cuda", non_blocking=False).contiguous()
    return torch.from_numpy(np.ascontiguousarray(arr)).to(
        device="cuda", dtype=bf16
    ).contiguous()


def _resolve_lora_pair(
    la_key: str, raw: dict
) -> Optional[tuple[str, str]]:
    """Resolve a LoRA-A key to its (base_key, lora_b_key) pair.

    openpi's LoRA module flattens (sep='.') into one of two patterns:

      1. **Einsum** (attention q/kv/o, action-expert q/kv/o):
         base = ``X.w``,   la = ``X.lora_a``,   lb = ``X.lora_b``
         → dot-separated. ``la_key.endswith('.lora_a')``.

      2. **FeedForward** (mlp gating / linear projections):
         base = ``X.gating_einsum``,
         la   = ``X.gating_einsum_lora_a``,
         lb   = ``X.gating_einsum_lora_b``
         → underscore-suffixed. ``la_key.endswith('_lora_a')`` and the
         segment before the underscore matches the base parameter name.

    Returns ``(base_key, lb_key)`` if both endpoints exist in ``raw``,
    or ``None`` if the LoRA-A is orphaned. Caller is expected to warn
    on ``None`` returns rather than crash, so a partially-broken
    checkpoint surfaces in logs (and downstream key-mismatch errors)
    rather than silently producing wrong weights.
    """
    # Pattern 1: dot-separated Einsum.
    if la_key.endswith(".lora_a"):
        prefix = la_key[: -len(".lora_a")]
        base_key = prefix + ".w"
        lb_key = prefix + ".lora_b"
        if base_key in raw and lb_key in raw:
            return base_key, lb_key

    # Pattern 2: underscore-suffixed FeedForward.
    if la_key.endswith("_lora_a") and "." in la_key:
        head, tail = la_key.rsplit(".", 1)
        if tail.endswith("_lora_a") and tail != "lora_a":
            base_name = tail[: -len("_lora_a")]
            base_key = f"{head}.{base_name}"
            lb_key = f"{head}.{base_name}_lora_b"
            if base_key in raw and lb_key in raw:
                return base_key, lb_key

    return None


def _build_padded_attn_lora_a(
    la_q: np.ndarray,         # (NH, D, r_q)
    la_k: np.ndarray,         # (D, r_kv)
    la_v: np.ndarray,         # (D, r_kv)
    fuse_attn: np.ndarray | None,  # (D,) for encoder RMSNorm; None for decoder AdaRMSNorm
) -> np.ndarray:
    """Stack Q/K/V LoRA ``lora_a`` matrices into one (D, NH*r_q + 2*r_kv) tensor.

    The runtime path computes ``neck = x @ la_full`` (single ``bf16_nn``),
    giving ``(seq, NH*r_q + 2*r_kv)`` where each contiguous rank-r block
    corresponds to one Q head (NH blocks) or to K / V (one block each).
    The matching ``_build_padded_attn_lora_b`` lays out the per-block
    output projections so a single ``bf16_nn_res`` accumulates the right
    LoRA delta into the fused QKV output slice in one pass.

    ``fuse_attn``: pass ``(1 + pre_attention_norm.scale)`` for the
    encoder, where the static RMSNorm scale can be folded into the LoRA
    input axis to match the base QKV weight's fold. Pass ``None`` for
    the decoder, where the norm is AdaRMSNorm (per-step time-conditioned
    modulation) and the normed activation is computed at runtime — no
    static fold possible.
    """
    NH, D, r_q = la_q.shape
    D_k, r_kv = la_k.shape
    assert D_k == D, f"K la in_dim {D_k} != Q la in_dim {D}"
    assert la_v.shape == (D, r_kv), f"V la shape {la_v.shape} != ({D}, {r_kv})"
    # Q: (NH, D, r_q) → flatten heads on the column axis → (D, NH*r_q)
    la_q_flat = la_q.transpose(1, 0, 2).reshape(D, NH * r_q)
    # Concatenate K and V columns. KV LoRA already has D as the second axis.
    la_full = np.concatenate([la_q_flat, la_k, la_v], axis=1).astype(np.float32)
    if fuse_attn is not None:
        la_full = la_full * fuse_attn[:, None]
    return la_full


def _interleave_lora_b_hd_axis(lb: np.ndarray) -> np.ndarray:
    """Apply the RoPE-friendly HD-axis interleave to a LoRA ``lb`` matrix.

    ``_interleave_qk_np`` operates on the FIRST axis (``out_dim``) of a
    ``(out_dim, in_dim)`` weight, swapping the first / second halves of
    each head_dim block. For LoRA ``lb`` we have shape ``(r, HD)`` and
    need to apply the SAME swap pattern to the LAST axis (HD). We
    transpose so HD becomes the first axis, reuse the existing helper
    with ``num_heads=1``, and transpose back. This produces a per-row
    permutation of HD identical to what the base Q / K weight already
    received in ``convert_pi05_orbax``.
    """
    r, HD = lb.shape
    # _interleave_qk_np expects (out_dim, in_dim). Pass (HD, r) so HD is
    # the "out" axis it operates on. num_heads=1 keeps the whole HD
    # within one head; head_dim = HD then splits into (2, HD//2).
    return _interleave_qk_np(lb.T, num_heads=1).T


def _build_padded_attn_lora_b(
    lb_q: np.ndarray,          # (NH, r_q, HD)
    lb_k: np.ndarray,          # (r_kv, HD)
    lb_v: np.ndarray,          # (r_kv, HD)
) -> np.ndarray:
    """Build the block-diagonal LoRA ``lora_b`` for fused QKV addition.

    Output shape: ``(NH * r_q + 2 * r_kv, NH * HD + 2 * NKV * HD)``.

    The QKV output buffer of the base GEMM is laid out as
    ``[Q_head_0 | Q_head_1 | ... | Q_head_{NH-1} | K | V]`` along the
    column axis, each block of width HD (or NKV*HD = HD for NKV=1).
    The neck buffer (output of ``_build_padded_attn_lora_a``) is laid
    out as ``[Q_head_0_neck | ... | Q_head_{NH-1}_neck | K_neck | V_neck]``,
    each block of width r_q (for Q) or r_kv (for K/V). We place each
    block's ``lb`` at the diagonal cell ``(rows_for_neck, cols_for_out)``
    so a single ``bf16_nn_res`` does the strided accumulation in one
    pass without needing a custom kernel.

    For Q and K, ``lb``'s HD axis is interleaved (RoPE swap pattern)
    so the runtime delta matches the base GEMM's output ordering.
    V has no interleave (V skips RoPE).
    """
    NH, r_q, HD = lb_q.shape
    r_kv, HD_k = lb_k.shape
    assert HD_k == HD, f"K lb HD {HD_k} != Q lb HD {HD}"
    assert lb_v.shape == (r_kv, HD), f"V lb shape {lb_v.shape} != ({r_kv}, {HD})"
    NKV = 1  # Pi0.5 GQA — see pipeline_rtx.ENC_NKV / DEC_NKV.
    total_in = NH * r_q + 2 * r_kv
    total_out = NH * HD + 2 * NKV * HD
    lb_full = np.zeros((total_in, total_out), dtype=np.float32)
    # Q: one block per head along the diagonal. Interleave HD per head.
    for h in range(NH):
        lb_h_int = _interleave_lora_b_hd_axis(lb_q[h])  # (r_q, HD)
        lb_full[h * r_q : (h + 1) * r_q,
                h * HD  : (h + 1) * HD] = lb_h_int
    # K: one block after all Q blocks, in the K columns. Interleave HD.
    lb_k_int = _interleave_lora_b_hd_axis(lb_k)
    lb_full[NH * r_q : NH * r_q + r_kv,
            NH * HD : NH * HD + NKV * HD] = lb_k_int
    # V: one block after K, in the V columns. NO interleave (V skips RoPE).
    lb_full[NH * r_q + r_kv : NH * r_q + 2 * r_kv,
            NH * HD + NKV * HD :] = lb_v
    return lb_full


def _build_padded_gateup_lora(
    la_gate: np.ndarray,   # (D, r)
    la_up:   np.ndarray,   # (D, r)
    lb_gate: np.ndarray,   # (r, H)
    lb_up:   np.ndarray,   # (r, H)
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse the gate + up LoRA matrices into a single padded pair.

    The FP8 (and INT8) encoder FFN path uses a single fused weight
    ``encoder_ffn_gate_up_w_{i}`` of shape ``(D, 2H)`` where the
    column layout is ``[gate_proj | up_proj]``. The single FP8 GEMM
    writes the (seq, 2H) output into ``encoder_gate_merged``. To add
    a LoRA delta in the same buffer with a single fused
    ``_apply_enc_lora`` call, we mirror the QKV-padded pattern:

      la_padded = [la_gate | la_up]   in (D, 2r)  — column-stack
      lb_padded[:r,  :H] = lb_gate    block-diagonal in (2r, 2H)
      lb_padded[ r:, H:] = lb_up
      (other blocks = 0)

    The runtime then runs
        neck = x @ la_padded                          # (seq, 2r)
        encoder_gate_merged += neck @ lb_padded       # (seq, 2H)
    which expands to
        encoder_gate_merged[:, :H] += (x @ la_gate) @ lb_gate
        encoder_gate_merged[:, H:] += (x @ la_up)   @ lb_up
    exactly (the cross-blocks in lb_padded are zero so they
    contribute nothing). One bf16_nn + one bf16_nn_res covers both
    LoRA contributions and matches the fused FP8 base GEMM's output
    layout — no per-half pointer offset / leading-dim trick needed.

    All inputs are fp32; the caller is responsible for the final
    bf16 cast (same convention as the rest of this file).
    """
    D, r = la_gate.shape
    r_check, H = lb_gate.shape
    assert la_up.shape == (D, r), f"la_up {la_up.shape} != ({D}, {r})"
    assert lb_up.shape == (r, H), f"lb_up {lb_up.shape} != ({r}, {H})"
    assert r == r_check, f"rank mismatch la r={r} vs lb r={r_check}"

    la_padded = np.concatenate(
        [la_gate.astype(np.float32), la_up.astype(np.float32)], axis=1)
    lb_padded = np.zeros((2 * r, 2 * H), dtype=np.float32)
    lb_padded[:r, :H] = lb_gate.astype(np.float32)
    lb_padded[r:, H:] = lb_up.astype(np.float32)
    return la_padded, lb_padded


def _build_o_lora(
    la_o: np.ndarray,    # (NH, HD, r)
    lb_o: np.ndarray,    # (NH, r, D)
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten + N-sum the attention output LoRA matrices.

    JAX's ``attn_vec_einsum`` LoRA computes
    ``lora_int = einsum("BTNH,NHL->BTL", x, la)`` (sums over N and H)
    and ``lora_out = einsum("BTL,NLD->BTD", lora_int, lb)`` (sums over N
    and L). The N axis in lb is a free axis (not in the output, not
    contracted with ``lora_int``), so it can be pre-summed once in fp32
    at conversion time. Per openpi (``lora_runtime.py::_patch_o_proj_forward``):

      la_full = la_o.reshape(NH*HD, r)  # match the (B, T, NH*HD) base input
      lb_summed = lb_o.sum(axis=0)       # (r, D), pre-summed in fp32

    Runtime then does a standard two-matmul pattern: ``out += x @ la_full @ lb_summed``.
    No norm fold here (O sits between FMHA and the residual_add, no norm).
    """
    NH, HD, r = la_o.shape
    r_check, D = lb_o.shape[1], lb_o.shape[2]
    assert lb_o.shape == (NH, r_check, D), (
        f"O lb shape {lb_o.shape} != ({NH}, {r_check}, {D})")
    assert r == r_check, f"O la r={r} != lb r={r_check}"
    la_full = la_o.reshape(NH * HD, r).astype(np.float32)
    lb_summed = lb_o.sum(axis=0).astype(np.float32)
    return la_full, lb_summed


def _extract_lora_pairs(
    raw: dict,
    *,
    base_key_patterns: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pop LoRA tensors whose base key matches any of ``base_key_patterns``.

    Mirrors openpi's runtime-LoRA extraction (see
    ``openpi/JAX_TO_PYTORCH_LORA_CONVERSION.md`` §3): instead of merging
    the LoRA delta into the base weight before quantization, we keep
    ``lora_a`` / ``lora_b`` separate so the runtime forward can apply them
    as two bf16 matmuls AFTER the (FP8) base GEMM. This is the only way
    openpi's PyTorch port reached cos > 0.9997 vs JAX on
    ``pi05_openarm_ngc_lora_v4`` — pre-merging at fp32 still lost ~8%
    magnitude (Bug #2 in ``PYTORCH_PARITY_DEBUG.md``).

    For each ``lora_a`` key that resolves to a (base_key, lora_b_key)
    triple AND whose ``base_key`` contains any of ``base_key_patterns`` as
    a substring, this function:

    * removes the ``lora_a`` and ``lora_b`` entries from ``raw`` (so the
      subsequent ``_maybe_merge_lora`` call leaves those bases alone), and
    * returns ``{base_key: (la_fp32, lb_fp32)}`` so the caller can stash
      the extracted pair into the pipeline's weights dict under whatever
      naming the runtime path expects.

    The base weight itself is left in ``raw`` untouched so the regular
    conversion code can pick it up exactly as before — just without the
    LoRA delta baked in.

    Args:
        raw: flat dict from ``_load_orbax``. Mutated in place: matching
            ``lora_a`` / ``lora_b`` entries are removed. The corresponding
            base entries are *not* touched.
        base_key_patterns: substrings; a LoRA pair is extracted iff its
            ``base_key`` contains at least one of these. Use e.g.
            ``("mlp.gating_einsum", "mlp.linear")`` to extract only the
            encoder/decoder FFN LoRA pairs. The "_1" suffix used by
            openpi for action-expert layers is matched too because the
            substring is contained in both variants.

    Returns:
        Dict ``{base_key: (lora_a_fp32, lora_b_fp32)}`` for every pair
        that was extracted. Both arrays are cast to fp32; the caller is
        responsible for the eventual bf16 truncation that matches the
        JAX runtime path.
    """
    lora_a_keys = sorted(k for k in raw if k.endswith("lora_a"))
    extracted: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for la_key in lora_a_keys:
        pair = _resolve_lora_pair(la_key, raw)
        if pair is None:
            continue
        base_key, lb_key = pair
        if not any(pat in base_key for pat in base_key_patterns):
            continue
        la = raw[la_key].astype(np.float32, copy=False)
        lb = raw[lb_key].astype(np.float32, copy=False)
        extracted[base_key] = (la, lb)
        del raw[la_key]
        del raw[lb_key]
    if extracted:
        logger.info(
            "Runtime LoRA: extracted %d LoRA pair(s) for patterns %s "
            "(those bases will skip merge and be applied at runtime)",
            len(extracted), list(base_key_patterns),
        )
    return extracted


def _maybe_merge_lora(
    raw: dict,
    *,
    scaling: float = 1.0,
    log_layers: bool = False,
) -> dict:
    """Merge LoRA params into base weights in fp32, in place on ``raw``.

    Why fp32: openpi's PyTorch parity work
    (`openpi/PYTORCH_PARITY_DEBUG.md`) showed that pre-merging LoRA in
    bf16 causes ~8% magnitude bias in the final action because the
    rank-r intermediate gets quantized. Doing the same merge in fp32 —
    BEFORE the subsequent fp32→bf16 truncation that downstream cares
    about — costs zero precision relative to the JAX bf16 inference
    path because the bf16 rounding now happens on the *full* merged
    weight, not on the narrow rank-r intermediate.

    Why a loader-level merge (vs. runtime LoRA): the FlashRT FP8 GEMM
    operates on the merged weight. Keeping ``lora_a`` / ``lora_b``
    separate at inference would require either (a) a separate FP8
    calibration / quantize pass for the LoRA neck (way more invasive),
    or (b) running the LoRA adds in bf16/fp32 alongside the FP8 base
    (sacrifices the FlashRT speed advantage). Offline fp32-merge
    sidesteps both.

    Args:
        raw: flat dict from ``_load_orbax``. Modified in place: lora_a /
            lora_b entries are removed; their corresponding base entries
            are replaced with the merged fp32 weight.
        scaling: LoRA scaling factor (alpha / rank, or alpha / sqrt(rank)
            for rslora). Default 1.0 matches openpi pi05's r=16/α=16 and
            r=32/α=32 LoRA configs. Override via the
            ``FLASHRT_LORA_SCALING`` env var at the call site.
        log_layers: emit a debug-level log line per merged tensor; useful
            for verifying every expected LoRA slot was picked up.

    Returns:
        ``raw`` (the same dict instance), with LoRA fused away.
    """
    # Catches both Einsum (.lora_a) and FeedForward (_lora_a) patterns.
    lora_a_keys = sorted(k for k in raw if k.endswith("lora_a"))
    if not lora_a_keys:
        logger.info("LoRA merge: no lora_a keys found, treating as base checkpoint")
        return raw

    merged = 0
    skipped: list[str] = []
    for la_key in lora_a_keys:
        pair = _resolve_lora_pair(la_key, raw)
        if pair is None:
            logger.warning(
                "LoRA merge: %s has no matching base weight + lora_b — skipping",
                la_key,
            )
            skipped.append(la_key)
            continue
        base_key, lb_key = pair

        w = raw[base_key].astype(np.float32, copy=False)
        la = raw[la_key].astype(np.float32, copy=False)
        lb = raw[lb_key].astype(np.float32, copy=False)
        # np.matmul broadcasts over leading dims, so multi-layer stacks
        # (la (L, ..., r), lb (L, ..., r, out)) merge in one call.
        delta = np.matmul(la, lb)
        if delta.shape != w.shape:
            logger.warning(
                "LoRA merge: shape mismatch at %s: base=%s, delta=%s — skipping",
                base_key, w.shape, delta.shape,
            )
            skipped.append(la_key)
            continue

        raw[base_key] = w + scaling * delta
        del raw[la_key]
        del raw[lb_key]
        merged += 1
        if log_layers:
            logger.debug("LoRA merge: %s ← + %.4f * (%s @ %s)", base_key, scaling, la_key, lb_key)

    if skipped:
        logger.warning("LoRA merge: skipped %d entries: %s", len(skipped), skipped[:5])
    logger.info("LoRA merge: fused %d tensor(s) at scaling=%.4f", merged, scaling)
    return raw


def convert_pi05_orbax(
    checkpoint_dir: Union[str, pathlib.Path]
) -> dict:
    """Convert a Pi0.5 Orbax JAX checkpoint to the rtx pipeline weight dict.

    Output schema is **identical** to ``convert_pi05_safetensors`` (torch
    bf16 cuda tensors with rtx key names) so the same downstream FP8
    quantize + style precompute + pipeline build code applies to both
    frontends. See
    ``flash_rt.frontends.torch.pi05_rtx.convert_pi05_safetensors`` for
    the full schema.
    """
    from flash_rt.core.weights.loader import _load_orbax

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    logger.info("Loading Pi0.5 Orbax checkpoint: %s", checkpoint_dir)
    raw = _load_orbax(str(checkpoint_dir))

    # LoRA scaling = alpha / rank (or alpha / sqrt(rank) for rslora).
    # Override via env if training used non-default alpha/rank.
    lora_scaling = float(os.environ.get("FLASHRT_LORA_SCALING", "1.0"))

    # Optional runtime LoRA extraction. See openpi/JAX_TO_PYTORCH_LORA_CONVERSION.md
    # §3 and PYTORCH_PARITY_DEBUG.md "★ 2026-05-19 RESOLVED": pre-merging
    # LoRA (even in fp32) costs ~8% magnitude vs JAX through 18 layers ×
    # 10 diffusion steps because the rank-r intermediate's rounding order
    # differs from JAX's two-matmul forward. Only runtime LoRA — keep
    # ``lora_a`` / ``lora_b`` separate, apply as two bf16 matmuls added
    # to the base GEMM output — recovers parity (cos > 0.9997 deployed
    # on the OpenArm robot per openpi's measurements). For FlashRT this
    # also fixes the FP8 cos 0.617 catastrophe: per-tensor FP8 calibrated
    # against LoRA-merged activations sees outliers up to amax/median 850×
    # at ``encoder_ffn_down_w_16``; once LoRA is removed from the merge,
    # the base-only activation distribution is well-behaved.
    #
    # FLASHRT_RUNTIME_LORA (default 0):
    #   0          — merge all LoRA into base (current production)
    #   1, "all"   — extract every LoRA pair (252 modules on Pi0.5)
    #   "encoder_ffn" — only encoder MLP gating_einsum + linear (54 pairs,
    #                   the suspected outlier source); day-1 scope.
    #
    # The extracted pairs are stashed into the *raw* dict under
    # ``__runtime_lora_pairs__`` for the converter loop below to pick up
    # and route to the right ckpt key names.
    runtime_lora_mode = os.environ.get("FLASHRT_RUNTIME_LORA", "0").lower()
    runtime_lora_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if runtime_lora_mode not in ("", "0", "false", "no"):
        if runtime_lora_mode in ("1", "true", "yes", "all"):
            # All 252 LoRA modules in Pi0.5: encoder + decoder, attention
            # (q/kv/o) + MLP (gating/down). The attn.* substrings also
            # match the action-expert "_1"-suffixed keys
            # (e.g. "layers.attn.q_einsum_1") because the encoder key is
            # a prefix of the decoder key. The mlp/mlp_1 split requires
            # two patterns each because "mlp." is NOT a substring of
            # "mlp_1." (the underscore breaks the match).
            patterns = (
                "layers.mlp.gating_einsum",
                "layers.mlp.linear",
                "layers.mlp_1.gating_einsum",
                "layers.mlp_1.linear",
                "layers.attn.q_einsum",         # matches q_einsum and q_einsum_1
                "layers.attn.kv_einsum",        # matches kv_einsum and kv_einsum_1
                "layers.attn.attn_vec_einsum",  # matches attn_vec_einsum and _1
            )
        elif runtime_lora_mode == "encoder_ffn":
            # Day-1 scope: only encoder MLP (54 pairs = 18 layers × 3 modules).
            # Substring "layers.mlp.gating_einsum" matches the encoder key
            # "PaliGemma.llm.layers.mlp.gating_einsum" but NOT the decoder
            # key "PaliGemma.llm.layers.mlp_1.gating_einsum" (the "_1."
            # between "mlp" and "gating" breaks the contiguous substring).
            patterns = (
                "layers.mlp.gating_einsum",
                "layers.mlp.linear",
            )
        elif runtime_lora_mode == "encoder":
            # Day-2 morning scope: full encoder LoRA — MLP gate/up/down +
            # attention q/kv/o. Closes the residual 5 % cosine gap that
            # encoder-FFN-only leaves (which is the bias from the
            # still-merged encoder attention LoRA). The decoder LoRA
            # stays merged. q_einsum and the others use the substring
            # form that matches encoder names only (no "_1" suffix).
            patterns = (
                "layers.mlp.gating_einsum",
                "layers.mlp.linear",
                "layers.attn.q_einsum.",          # encoder Q (trailing dot
                "layers.attn.kv_einsum.",         # excludes "_1." variant)
                "layers.attn.attn_vec_einsum.",
            )
        else:
            raise ValueError(
                f"FLASHRT_RUNTIME_LORA={runtime_lora_mode!r} not recognised; "
                "expected '0', '1'/'all', or 'encoder_ffn'.")
        runtime_lora_pairs = _extract_lora_pairs(
            raw, base_key_patterns=patterns)

    # LoRA merge (no-op if the checkpoint has no .lora_a entries, or if
    # FLASHRT_RUNTIME_LORA extracted everything above). Done BEFORE the
    # fp32→bf16 truncation below so the rank-r LoRA neck never sees bf16
    # — matches the JAX server's effective accuracy without the runtime
    # LoRA cost.
    raw = _maybe_merge_lora(raw, scaling=lora_scaling)

    # Bit-truncate fp32 → bf16 → fp32. Production loads everything as
    # bf16; truncating now guarantees byte-identical FP8 scales vs the
    # torch frontend (which loads from safetensors that were already
    # saved in bf16).
    raw = {
        k: v.astype(ml_dtypes.bfloat16).astype(np.float32)
        if v.dtype == np.float32 else v
        for k, v in raw.items()
    }

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) ──
    #
    # Patch embedding: JAX stores ``(14, 14, 3, 1152)`` (HWCO) which is
    # exactly what the rtx pipeline expects after its
    # ``permute(2, 3, 1, 0)`` step in the safetensors path. So we keep
    # the JAX layout as-is.
    pe_w = raw["PaliGemma.img.embedding.kernel"]   # (14, 14, 3, 1152)
    ckpt["vision_patch_embedding_w"] = _to_bf16_cuda(pe_w)
    ckpt["vision_patch_embedding_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.embedding.bias"])
    # JAX position embedding has a leading batch axis (1, 256, 1152)
    pos_emb = raw["PaliGemma.img.pos_embedding"].squeeze(0)  # (256, 1152)
    ckpt["vision_position_embedding"] = _to_bf16_cuda(pos_emb)

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    enc_blk = "PaliGemma.img.Transformer.encoderblock"
    for i in range(VIS_L):
        # LayerNorms (stacked)
        ln1_w_list.append(raw[f"{enc_blk}.LayerNorm_0.scale"][i])
        ln1_b_list.append(raw[f"{enc_blk}.LayerNorm_0.bias"][i])
        ln2_w_list.append(raw[f"{enc_blk}.LayerNorm_1.scale"][i])
        ln2_b_list.append(raw[f"{enc_blk}.LayerNorm_1.bias"][i])

        # Attention Q/K/V einsum: JAX (1152, 16, 72) → row-major (1152, 1152)
        # then concat into (1152, 3456). Note: rtx schema expects
        # in_dim-first, i.e. (in=1152, 3*out=3456) — same as the
        # safetensors path's ``torch.cat([q,k,v], dim=0).t()``.
        q_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.kernel"][i]
        k_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.kernel"][i]
        v_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.kernel"][i]
        # JAX einsum kernel for ``BTD,DNH->BTNH`` is (D, N, H) = (1152, 16, 72).
        # Reshape to (D, N*H) = (1152, 1152), no transpose.
        q_2d = q_w.reshape(1152, -1)
        k_2d = k_w.reshape(1152, -1)
        v_2d = v_w.reshape(1152, -1)
        qkv_w_list.append(np.concatenate([q_2d, k_2d, v_2d], axis=1))
        # Biases: JAX (16, 72) → flat (1152,)
        q_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.query.bias"][i].reshape(-1)
        k_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.key.bias"][i].reshape(-1)
        v_b = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.value.bias"][i].reshape(-1)
        qkv_b_list.append(np.concatenate([q_b, k_b, v_b]))

        # O projection: JAX (16, 72, 1152) — einsum ``BTNH,NHD->BTD``
        # In rtx schema we want (in=1152, out=1152). The N*H dim is the
        # input axis, so reshape to (1152, 1152) (no transpose).
        o_w = raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.kernel"][i]
        o_w_list.append(o_w.reshape(-1, 1152))
        o_b_list.append(raw[f"{enc_blk}.MultiHeadDotProductAttention_0.out.bias"][i])

        # FFN up: JAX (1152, 4304) — already in (in, out) layout.
        up_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.kernel"][i])
        up_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_0.bias"][i])

        # FFN down: JAX (4304, 1152) — already (in, out)
        down_w_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.kernel"][i])
        down_b_list.append(raw[f"{enc_blk}.MlpBlock_0.Dense_1.bias"][i])

    ckpt["vision_attn_qkv_w"] = _to_bf16_cuda(np.stack(qkv_w_list))
    ckpt["vision_attn_qkv_b"] = _to_bf16_cuda(np.stack(qkv_b_list))
    ckpt["vision_attn_o_w"] = _to_bf16_cuda(np.stack(o_w_list))
    ckpt["vision_attn_o_b"] = _to_bf16_cuda(np.stack(o_b_list))
    ckpt["vision_ffn_up_w"] = _to_bf16_cuda(np.stack(up_w_list))
    ckpt["vision_ffn_up_b"] = _to_bf16_cuda(np.stack(up_b_list))
    ckpt["vision_ffn_down_w"] = _to_bf16_cuda(np.stack(down_w_list))
    ckpt["vision_ffn_down_b"] = _to_bf16_cuda(np.stack(down_b_list))
    ckpt["vision_pre_attn_norm_w"] = _to_bf16_cuda(np.stack(ln1_w_list))
    ckpt["vision_pre_attn_norm_b"] = _to_bf16_cuda(np.stack(ln1_b_list))
    ckpt["vision_pre_ffn_norm_w"] = _to_bf16_cuda(np.stack(ln2_w_list))
    ckpt["vision_pre_ffn_norm_b"] = _to_bf16_cuda(np.stack(ln2_b_list))
    ckpt["vision_final_norm_w"] = _to_bf16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.scale"])
    ckpt["vision_final_norm_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.Transformer.encoder_norm.bias"])

    # ── Multi-modal projector ──
    # JAX kernel (1152, 2048) — already (in, out)
    ckpt["encoder_multi_modal_projector_w"] = _to_bf16_cuda(
        raw["PaliGemma.img.head.kernel"])
    ckpt["encoder_multi_modal_projector_b"] = _to_bf16_cuda(
        raw["PaliGemma.img.head.bias"])

    # ── Encoder (18 Gemma-2B layers with RMSNorm fold) ──
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []
    # Runtime-LoRA tensors per-encoder-layer (only populated when the
    # corresponding base key was extracted above). The same fp32 RMSNorm
    # fold that the base weights get also applies to ``lora_a`` here so
    # the runtime forward needs no extra norm-fold step.
    enc_gate_la_list, enc_gate_lb_list = [], []
    enc_up_la_list, enc_up_lb_list = [], []
    enc_down_la_list, enc_down_lb_list = [], []
    # FP8 path needs the fused (D, 2r) / (2r, 2H) form (matches
    # ``encoder_ffn_gate_up_w_{i}`` layout). Built in the same loop so
    # both BF16 (separate gate/up) and FP8 (fused gateup) paths have
    # the right per-layer slices ready. See _build_padded_gateup_lora.
    enc_gateup_la_list, enc_gateup_lb_list = [], []
    # Encoder attention LoRA (added in the "encoder" mode). Q, K, V are
    # combined into one padded la/lb per layer so the runtime path runs
    # a single bf16_nn + bf16_nn_res for all three projections at once
    # (see _build_padded_attn_lora_{a,b}). O is a standard 2D LoRA after
    # N-summing lb in fp32 (see _build_o_lora).
    enc_attn_qkv_la_list, enc_attn_qkv_lb_list = [], []
    enc_attn_o_la_list,   enc_attn_o_lb_list   = [], []
    _gating_base = "PaliGemma.llm.layers.mlp.gating_einsum"
    _linear_base = "PaliGemma.llm.layers.mlp.linear"
    _gating_pair = runtime_lora_pairs.get(_gating_base)
    _linear_pair = runtime_lora_pairs.get(_linear_base)
    _q_pair      = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.q_einsum.w")
    _kv_pair     = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.kv_einsum.w")
    _o_pair      = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.attn_vec_einsum.w")
    _has_enc_ffn_lora  = _gating_pair is not None or _linear_pair is not None
    _has_enc_attn_lora = (_q_pair is not None and _kv_pair is not None
                          and _o_pair is not None)
    if (_q_pair is not None) != (_kv_pair is not None) or \
       (_q_pair is not None) != (_o_pair is not None):
        raise ValueError(
            "Runtime LoRA: partial encoder-attention extraction is not "
            "supported — q/kv/o must all be present or all merged. Got "
            f"q={_q_pair is not None}, kv={_kv_pair is not None}, "
            f"o={_o_pair is not None}.")

    for i in range(ENC_L):
        # CRITICAL: fuse in fp32 — bf16 rounds values near -1.0 to exactly
        # -1.0, collapsing (1 + scale) to 0 and zeroing entire channels.
        attn_scale = raw[
            "PaliGemma.llm.layers.pre_attention_norm.scale"][i].astype(np.float32)
        fuse_attn = 1.0 + attn_scale  # (2048,)

        # Q einsum: JAX (8, 2048, 256) for "BTD,NDH->BTNH"
        #   → (N*H, D) = (2048, 2048) row-major (out, in)
        #   → interleave heads for RoPE
        #   → fold the LN scale into the in_dim
        #   → transpose to (in, out) = (2048, 2048) for rtx schema
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 2048)
        q_2d = _interleave_qk_np(q_2d, 8)
        q_2d = q_2d * fuse_attn[None, :]

        # KV einsum: JAX (2, 1, 2048, 256) for "BTD,NDH->BTNH" with N=1
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 2048)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 2048)
        k_2d = _interleave_qk_np(k_2d, 1)
        k_2d = k_2d * fuse_attn[None, :]
        v_2d = v_2d * fuse_attn[None, :]

        # Concat → (2560, 2048) → transpose → (2048, 2560) (in, out)
        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T
        enc_qkv_list.append(qkv)

        # O: JAX (8, 256, 2048) for "BTNH,NHD->BTD".
        # The einsum has D as the *output* axis and N*H as the *input* axis,
        # so reshape (N*H, D) is already in the (in, out) layout the rtx
        # pipeline expects — NO transpose. (The torch frontend's `.t()` is
        # because HF stores weights in (out, in) PyTorch convention.)
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum.w"][i].astype(np.float32)
        enc_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        # Runtime LoRA — encoder attention (Q + KV merged via padding; O via N-sum).
        if _has_enc_attn_lora:
            # Per-layer slices: orbax stacks all 18 layers leading-axis.
            la_q_full, lb_q_full = _q_pair       # (L, NH, D, r), (L, NH, r, HD)
            la_kv_full, lb_kv_full = _kv_pair    # (L, 2, NKV, D, r), (L, 2, NKV, r, HD)
            la_o_full, lb_o_full   = _o_pair     # (L, NH, HD, r), (L, NH, r, D)

            la_q  = la_q_full[i]                              # (NH, D, r)
            lb_q  = lb_q_full[i]                              # (NH, r, HD)
            # KV stacks K (axis 1 == 0) and V (axis 1 == 1). NKV=1 → squeeze.
            la_k  = la_kv_full[i, 0, 0]                       # (D, r)
            la_v  = la_kv_full[i, 1, 0]
            lb_k  = lb_kv_full[i, 0, 0]                       # (r, HD)
            lb_v  = lb_kv_full[i, 1, 0]
            la_o  = la_o_full[i]                              # (NH, HD, r)
            lb_o  = lb_o_full[i]                              # (NH, r, D)

            # Build padded QKV LoRA tensors. fuse_attn folds into la (input
            # axis), exactly like the base QKV weight got it above (`q_2d
            # = q_2d * fuse_attn[None, :]` etc.). The interleave_fn matches
            # the per-head RoPE-friendly interleave the base weights get.
            qkv_la = _build_padded_attn_lora_a(la_q, la_k, la_v, fuse_attn)
            qkv_lb = _build_padded_attn_lora_b(lb_q, lb_k, lb_v)
            enc_attn_qkv_la_list.append(qkv_la)
            enc_attn_qkv_lb_list.append(qkv_lb)

            # Build O LoRA (N-summed lb).
            o_la, o_lb = _build_o_lora(la_o, lb_o)
            enc_attn_o_la_list.append(o_la)
            enc_attn_o_lb_list.append(o_lb)

        # Gate / Up: JAX (2, 2048, 16384) — both already (in, out)
        ffn_scale = raw[
            "PaliGemma.llm.layers.pre_ffw_norm.scale"][i].astype(np.float32)
        fuse_ffn = 1.0 + ffn_scale

        gu_w = raw["PaliGemma.llm.layers.mlp.gating_einsum"][i].astype(np.float32)
        gate_w = gu_w[0] * fuse_ffn[:, None]   # (2048, 16384)
        up_w = gu_w[1] * fuse_ffn[:, None]
        enc_gate_list.append(gate_w)
        enc_up_list.append(up_w)

        # Runtime LoRA — encoder gate/up. JAX storage for the merged
        # gating_einsum LoRA pair is:
        #   la (2, D, r)   — gate/up stacked along leading axis
        #   lb (2, r, H)
        # We slice gate (idx 0) and up (idx 1), and fold the FFN RMSNorm
        # scale into ``la`` (in_dim axis) so it matches the corresponding
        # fold already baked into ``gate_w`` / ``up_w`` above. ``lb``
        # stays unfolded because the JAX einsum order is
        # ``(x * fuse_ffn) @ la → (x * fuse_ffn) @ la @ lb``: only the
        # first matmul touches the in_dim norm-folded path.
        if _gating_pair is not None:
            # Orbax stacks all 18 layers along the leading axis:
            #   la_gu shape: (ENC_L, 2, D, r) — (layers, gate/up, in, rank)
            #   lb_gu shape: (ENC_L, 2, r, H)
            # Pull this layer's slice and split gate (idx 0) from up (idx 1).
            la_gu, lb_gu = _gating_pair
            la_gate = la_gu[i, 0] * fuse_ffn[:, None]  # (D, r)
            la_up   = la_gu[i, 1] * fuse_ffn[:, None]
            lb_gate = lb_gu[i, 0]                       # (r, H)
            lb_up   = lb_gu[i, 1]
            enc_gate_la_list.append(la_gate)
            enc_gate_lb_list.append(lb_gate)
            enc_up_la_list.append(la_up)
            enc_up_lb_list.append(lb_up)

            # Fused (D, 2r) / (2r, 2H) padded LoRA for the FP8 path
            # whose base GEMM uses encoder_ffn_gate_up_w_{i} (D, 2H).
            gu_la, gu_lb = _build_padded_gateup_lora(
                la_gate, la_up, lb_gate, lb_up)
            enc_gateup_la_list.append(gu_la)
            enc_gateup_lb_list.append(gu_lb)

        # Down: JAX (16384, 2048) — already (in, out), no fold
        enc_down_list.append(
            raw["PaliGemma.llm.layers.mlp.linear"][i].astype(np.float32))

        # Runtime LoRA — encoder down. JAX storage:
        #   la (H, r)
        #   lb (r, D)
        # No norm fold (no RMSNorm sits between gate_geglu output and
        # the down projection input — fuse_ffn was already folded into
        # the gate/up weights).
        if _linear_pair is not None:
            # Orbax stacks all 18 layers along the leading axis:
            #   la_down shape: (ENC_L, H, r)
            #   lb_down shape: (ENC_L, r, D)
            la_down, lb_down = _linear_pair
            enc_down_la_list.append(la_down[i].astype(np.float32, copy=False))
            enc_down_lb_list.append(lb_down[i].astype(np.float32, copy=False))

    ckpt["encoder_attn_qkv_w"] = _to_bf16_cuda(np.stack(enc_qkv_list))
    ckpt["encoder_attn_o_w"] = _to_bf16_cuda(np.stack(enc_o_list))
    ckpt["encoder_ffn_gate_w"] = _to_bf16_cuda(np.stack(enc_gate_list))
    ckpt["encoder_ffn_up_w"] = _to_bf16_cuda(np.stack(enc_up_list))
    ckpt["encoder_ffn_down_w"] = _to_bf16_cuda(np.stack(enc_down_list))

    # Runtime LoRA — stash the extracted (la, lb) tensors next to the
    # base weights, stacked across all 18 encoder layers so the pipeline
    # can index them by layer in the encoder loop. Same dtype + device
    # convention as the base weights (bf16 cuda) so the bf16_nn GEMM in
    # the runtime forward sees a homogeneous bf16 input.
    if enc_gate_la_list:
        ckpt["encoder_ffn_gate_lora_a"] = _to_bf16_cuda(np.stack(enc_gate_la_list))
        ckpt["encoder_ffn_gate_lora_b"] = _to_bf16_cuda(np.stack(enc_gate_lb_list))
        ckpt["encoder_ffn_up_lora_a"]   = _to_bf16_cuda(np.stack(enc_up_la_list))
        ckpt["encoder_ffn_up_lora_b"]   = _to_bf16_cuda(np.stack(enc_up_lb_list))
        ckpt["encoder_ffn_gateup_lora_a"] = _to_bf16_cuda(np.stack(enc_gateup_la_list))
        ckpt["encoder_ffn_gateup_lora_b"] = _to_bf16_cuda(np.stack(enc_gateup_lb_list))
        logger.info(
            "Runtime LoRA: stashed encoder gate/up lora_a/lora_b (shape %s, %s) "
            "+ fused gateup (shape %s, %s) across %d layers",
            tuple(ckpt["encoder_ffn_gate_lora_a"].shape),
            tuple(ckpt["encoder_ffn_gate_lora_b"].shape),
            tuple(ckpt["encoder_ffn_gateup_lora_a"].shape),
            tuple(ckpt["encoder_ffn_gateup_lora_b"].shape),
            ENC_L,
        )
    if enc_down_la_list:
        ckpt["encoder_ffn_down_lora_a"] = _to_bf16_cuda(np.stack(enc_down_la_list))
        ckpt["encoder_ffn_down_lora_b"] = _to_bf16_cuda(np.stack(enc_down_lb_list))
        logger.info(
            "Runtime LoRA: stashed encoder down lora_a/lora_b (shape %s, %s) "
            "across %d layers",
            tuple(ckpt["encoder_ffn_down_lora_a"].shape),
            tuple(ckpt["encoder_ffn_down_lora_b"].shape),
            ENC_L,
        )
    if enc_attn_qkv_la_list:
        ckpt["encoder_attn_qkv_lora_a"] = _to_bf16_cuda(np.stack(enc_attn_qkv_la_list))
        ckpt["encoder_attn_qkv_lora_b"] = _to_bf16_cuda(np.stack(enc_attn_qkv_lb_list))
        ckpt["encoder_attn_o_lora_a"]   = _to_bf16_cuda(np.stack(enc_attn_o_la_list))
        ckpt["encoder_attn_o_lora_b"]   = _to_bf16_cuda(np.stack(enc_attn_o_lb_list))
        logger.info(
            "Runtime LoRA: stashed encoder attention qkv+o lora_a/lora_b "
            "(qkv la=%s, qkv lb=%s, o la=%s, o lb=%s) across %d layers",
            tuple(ckpt["encoder_attn_qkv_lora_a"].shape),
            tuple(ckpt["encoder_attn_qkv_lora_b"].shape),
            tuple(ckpt["encoder_attn_o_lora_a"].shape),
            tuple(ckpt["encoder_attn_o_lora_b"].shape),
            ENC_L,
        )
    # Stash the scaling factor for the pipeline to use at runtime.
    if enc_gate_la_list or enc_down_la_list or enc_attn_qkv_la_list:
        ckpt["runtime_lora_scaling"] = float(lora_scaling)

    # ── Decoder (18 Gemma-300M expert layers) ──
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    dec_attn_mod_w_list, dec_attn_mod_b_list = [], []
    dec_ffn_mod_w_list, dec_ffn_mod_b_list = [], []

    # Decoder runtime-LoRA lookups — same pattern as the encoder block
    # above but on the ``_1``-suffixed gemma_expert keys. AdaRMSNorm
    # modulation is time-conditioned and applied per-step at runtime, so
    # we do NOT fold any norm scale into the LoRA ``la`` tensors here
    # (encoder folds ``1 + pre_attention_norm.scale`` because that scale
    # is static).
    dec_attn_qkv_la_list, dec_attn_qkv_lb_list = [], []
    dec_attn_o_la_list,   dec_attn_o_lb_list   = [], []
    dec_ffn_gate_la_list, dec_ffn_gate_lb_list = [], []
    dec_ffn_up_la_list,   dec_ffn_up_lb_list   = [], []
    dec_ffn_down_la_list, dec_ffn_down_lb_list = [], []
    _dec_gating_base = "PaliGemma.llm.layers.mlp_1.gating_einsum"
    _dec_linear_base = "PaliGemma.llm.layers.mlp_1.linear"
    _dec_gating_pair = runtime_lora_pairs.get(_dec_gating_base)
    _dec_linear_pair = runtime_lora_pairs.get(_dec_linear_base)
    _dec_q_pair      = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.q_einsum_1.w")
    _dec_kv_pair     = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.kv_einsum_1.w")
    _dec_o_pair      = runtime_lora_pairs.get("PaliGemma.llm.layers.attn.attn_vec_einsum_1.w")
    _has_dec_ffn_lora  = (_dec_gating_pair is not None
                          and _dec_linear_pair is not None)
    _has_dec_attn_lora = (_dec_q_pair is not None
                          and _dec_kv_pair is not None
                          and _dec_o_pair is not None)
    if (_dec_q_pair is not None) != (_dec_kv_pair is not None) or \
       (_dec_q_pair is not None) != (_dec_o_pair is not None):
        raise ValueError(
            "Runtime LoRA: partial decoder-attention extraction is not "
            "supported — q/kv/o must all be present or all merged. Got "
            f"q={_dec_q_pair is not None}, kv={_dec_kv_pair is not None}, "
            f"o={_dec_o_pair is not None}.")
    if (_dec_gating_pair is not None) != (_dec_linear_pair is not None):
        raise ValueError(
            "Runtime LoRA: partial decoder-FFN extraction is not "
            "supported — gating + linear must both be present or both "
            f"merged. Got gating={_dec_gating_pair is not None}, "
            f"linear={_dec_linear_pair is not None}.")

    for i in range(DEC_L):
        # AdaRMSNorm modulation: JAX (1024, 3072) — already (in, out)
        dec_attn_mod_w_list.append(
            raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.kernel"][i])
        dec_attn_mod_b_list.append(
            raw["PaliGemma.llm.layers.pre_attention_norm_1.Dense_0.bias"][i])
        dec_ffn_mod_w_list.append(
            raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.kernel"][i])
        dec_ffn_mod_b_list.append(
            raw["PaliGemma.llm.layers.pre_ffw_norm_1.Dense_0.bias"][i])

        # Q einsum: JAX (8, 1024, 256) → (2048, 1024) (out, in) → interleave
        # → transpose → (1024, 2048) (in, out)
        q_w = raw["PaliGemma.llm.layers.attn.q_einsum_1.w"][i].astype(np.float32)
        q_2d = q_w.transpose(0, 2, 1).reshape(-1, q_w.shape[1])  # (2048, 1024)
        q_2d = _interleave_qk_np(q_2d, 8)

        # KV: JAX (2, 1, 1024, 256)
        kv_w = raw["PaliGemma.llm.layers.attn.kv_einsum_1.w"][i].astype(np.float32)
        k_2d = kv_w[0].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])  # (256, 1024)
        v_2d = kv_w[1].transpose(0, 2, 1).reshape(-1, kv_w.shape[2])
        k_2d = _interleave_qk_np(k_2d, 1)

        qkv = np.concatenate([q_2d, k_2d, v_2d], axis=0).T  # (1024, 2560)
        dec_qkv_list.append(qkv)

        # O: JAX (8, 256, 1024) for "BTNH,NHD->BTD".
        # Same logic as encoder: reshape to (N*H, D) = (2048, 1024) is
        # already (in, out). The torch frontend's .t() gets (1024, 2048)
        # which is wrong vs the schema; the rtx torch pipeline expects
        # (out=1024, in=2048) here per its decoder_attn_o_w[18, 2048, 1024]
        # buffer shape — that *is* (in=2048, out=1024) interpreting the
        # last two dims as the row-major matrix. Match the torch path.
        #
        # Look at the rtx pipeline shape declarations:
        #   decoder_attn_o_w (18, 2048, 1024) — used as A @ W where A is
        #   (10, 2048) and output is (10, 1024). For row-major NN GEMM
        #   that needs W shape (in=2048, out=1024). So torch's .t() is
        #   correct: HF stores (out=1024, in=2048), .t() → (in=2048, out=1024).
        #
        # JAX einsum BTNH,NHD->BTD has D as output, N*H as input. So the
        # raw (N*H, D) = (2048, 1024) is already (in, out). NO transpose
        # needed.
        o_w = raw["PaliGemma.llm.layers.attn.attn_vec_einsum_1.w"][i].astype(np.float32)
        dec_o_list.append(o_w.reshape(-1, o_w.shape[-1]))

        # Runtime LoRA — decoder attention (Q + KV merged via padding; O via N-sum).
        # No norm fold — AdaRMSNorm modulation is applied per-step at
        # runtime (see _decoder_layer's ada_rms_norm_style call).
        if _has_dec_attn_lora:
            la_q_full, lb_q_full = _dec_q_pair       # (L, NH, D, r), (L, NH, r, HD)
            la_kv_full, lb_kv_full = _dec_kv_pair    # (L, 2, NKV, D, r), (L, 2, NKV, r, HD)
            la_o_full, lb_o_full   = _dec_o_pair     # (L, NH, HD, r), (L, NH, r, D)

            la_q  = la_q_full[i]                              # (NH, D, r)
            lb_q  = lb_q_full[i]                              # (NH, r, HD)
            la_k  = la_kv_full[i, 0, 0]                       # (D, r)  (NKV=1, squeeze)
            la_v  = la_kv_full[i, 1, 0]
            lb_k  = lb_kv_full[i, 0, 0]                       # (r, HD)
            lb_v  = lb_kv_full[i, 1, 0]
            la_o  = la_o_full[i]                              # (NH, HD, r)
            lb_o  = lb_o_full[i]                              # (NH, r, D)

            qkv_la = _build_padded_attn_lora_a(la_q, la_k, la_v, fuse_attn=None)
            qkv_lb = _build_padded_attn_lora_b(lb_q, lb_k, lb_v)
            dec_attn_qkv_la_list.append(qkv_la)
            dec_attn_qkv_lb_list.append(qkv_lb)

            o_la, o_lb = _build_o_lora(la_o, lb_o)
            dec_attn_o_la_list.append(o_la)
            dec_attn_o_lb_list.append(o_lb)

        # Gate / Up: JAX (2, 1024, 4096) — already (in, out), no fold
        gu_w = raw["PaliGemma.llm.layers.mlp_1.gating_einsum"][i].astype(np.float32)
        dec_gate_list.append(gu_w[0])
        dec_up_list.append(gu_w[1])

        # Down: JAX (4096, 1024) — already (in, out)
        dec_down_list.append(
            raw["PaliGemma.llm.layers.mlp_1.linear"][i].astype(np.float32))

        # Runtime LoRA — decoder FFN gate/up + down. No norm fold —
        # AdaRMSNorm modulation is applied per-step at runtime.
        if _has_dec_ffn_lora:
            la_gu, lb_gu = _dec_gating_pair   # (L, 2, D, r), (L, 2, r, H)
            la_dn, lb_dn = _dec_linear_pair   # (L, H, r), (L, r, D)
            dec_ffn_gate_la_list.append(la_gu[i, 0].astype(np.float32))  # (D, r)
            dec_ffn_up_la_list.append(  la_gu[i, 1].astype(np.float32))
            dec_ffn_gate_lb_list.append(lb_gu[i, 0].astype(np.float32))  # (r, H)
            dec_ffn_up_lb_list.append(  lb_gu[i, 1].astype(np.float32))
            dec_ffn_down_la_list.append(la_dn[i].astype(np.float32))     # (H, r)
            dec_ffn_down_lb_list.append(lb_dn[i].astype(np.float32))     # (r, D)

    ckpt["decoder_attn_qkv_w"] = _to_bf16_cuda(np.stack(dec_qkv_list))
    ckpt["decoder_attn_o_w"] = _to_bf16_cuda(np.stack(dec_o_list))
    ckpt["decoder_ffn_gate_w"] = _to_bf16_cuda(np.stack(dec_gate_list))
    ckpt["decoder_ffn_up_w"] = _to_bf16_cuda(np.stack(dec_up_list))
    ckpt["decoder_ffn_down_w"] = _to_bf16_cuda(np.stack(dec_down_list))

    # Runtime LoRA tensors for the decoder (see pipeline_rtx.py:Phase C
    # for the apply side). Stacked across all 18 layers; the per-layer
    # slice is indexed inside ``_decoder_layer``. Same dtype contract
    # as the base decoder weights (bf16 cuda).
    if _has_dec_attn_lora:
        ckpt["decoder_attn_qkv_lora_a"] = _to_bf16_cuda(np.stack(dec_attn_qkv_la_list))
        ckpt["decoder_attn_qkv_lora_b"] = _to_bf16_cuda(np.stack(dec_attn_qkv_lb_list))
        ckpt["decoder_attn_o_lora_a"] = _to_bf16_cuda(np.stack(dec_attn_o_la_list))
        ckpt["decoder_attn_o_lora_b"] = _to_bf16_cuda(np.stack(dec_attn_o_lb_list))
    if _has_dec_ffn_lora:
        ckpt["decoder_ffn_gate_lora_a"] = _to_bf16_cuda(np.stack(dec_ffn_gate_la_list))
        ckpt["decoder_ffn_gate_lora_b"] = _to_bf16_cuda(np.stack(dec_ffn_gate_lb_list))
        ckpt["decoder_ffn_up_lora_a"]   = _to_bf16_cuda(np.stack(dec_ffn_up_la_list))
        ckpt["decoder_ffn_up_lora_b"]   = _to_bf16_cuda(np.stack(dec_ffn_up_lb_list))
        ckpt["decoder_ffn_down_lora_a"] = _to_bf16_cuda(np.stack(dec_ffn_down_la_list))
        ckpt["decoder_ffn_down_lora_b"] = _to_bf16_cuda(np.stack(dec_ffn_down_lb_list))

    ckpt["decoder_pre_attn_norm_mod_w"] = _to_bf16_cuda(np.stack(dec_attn_mod_w_list))
    ckpt["decoder_pre_attn_norm_mod_b"] = _to_bf16_cuda(np.stack(dec_attn_mod_b_list))
    ckpt["decoder_pre_ffn_norm_mod_w"] = _to_bf16_cuda(np.stack(dec_ffn_mod_w_list))
    ckpt["decoder_pre_ffn_norm_mod_b"] = _to_bf16_cuda(np.stack(dec_ffn_mod_b_list))

    ckpt["decoder_final_norm_mod_w"] = _to_bf16_cuda(
        raw["PaliGemma.llm.final_norm_1.Dense_0.kernel"])
    ckpt["decoder_final_norm_mod_b"] = _to_bf16_cuda(
        raw["PaliGemma.llm.final_norm_1.Dense_0.bias"])

    # ── Time MLP ──
    # JAX kernel (1024, 1024) is used as ``x @ kernel`` (in_dim, out_dim).
    # The rtx pipeline does NN GEMM ``x @ W`` so it also wants
    # (in, out) = (1024, 1024) — i.e. JAX layout directly, NO transpose.
    # The torch path does .t() because HF stores (out, in).
    ckpt["decoder_time_mlp_in_w"] = _to_bf16_cuda(raw["time_mlp_in.kernel"])
    ckpt["decoder_time_mlp_in_b"] = _to_bf16_cuda(raw["time_mlp_in.bias"])
    ckpt["decoder_time_mlp_out_w"] = _to_bf16_cuda(raw["time_mlp_out.kernel"])
    ckpt["decoder_time_mlp_out_b"] = _to_bf16_cuda(raw["time_mlp_out.bias"])

    # ── Sinusoidal time embeddings (10-step flow-matching schedule) ──
    # Identical to the safetensors path — schedule is determined by
    # num_steps + min/max_period only, not the checkpoint.
    num_steps = NUM_STEPS_DEFAULT
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    embedding_dim = 1024
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    time_emb_list = []
    for _ in range(num_steps):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb_list.append(
            torch.cat(
                [torch.sin(sinusoid_input), torch.cos(sinusoid_input)],
                dim=-1
            ).to(bf16)
        )
        t = t + dt
    ckpt["decoder_time_embeds"] = torch.cat(time_emb_list, dim=0).to("cuda")

    # ── Action projections ──
    # JAX action_in_proj.kernel: (32, 1024). The rtx pipeline expects
    # (in=32, out=1024) which is exactly this layout — torch's safetensors
    # path stores ``action_in_proj.weight.t()`` and the source HF weight
    # is (1024, 32), so safetensors does .t() to get (32, 1024). JAX has
    # no transpose to do.
    ckpt["decoder_action_in_proj_w"] = _to_bf16_cuda(raw["action_in_proj.kernel"])
    ckpt["decoder_action_in_proj_b"] = _to_bf16_cuda(raw["action_in_proj.bias"])
    # action_out_proj.kernel: JAX (1024, 32). Same logic — we want
    # (in=1024, out=32) which is the JAX layout directly.
    ckpt["decoder_action_out_proj_w"] = _to_bf16_cuda(raw["action_out_proj.kernel"])
    ckpt["decoder_action_out_proj_b"] = _to_bf16_cuda(raw["action_out_proj.bias"])

    # ── Embedding matrix (for prompt tokenisation) ──
    # JAX stores the input embedding under PaliGemma.llm.embedder, the lm_head
    # is tied to it. For prompt embedding we use the input embedder.
    ckpt["embedding_weight"] = _to_bf16_cuda(
        raw["PaliGemma.llm.embedder.input_embedding"])

    logger.info("Converted %d weight groups from Orbax", len(ckpt))
    return ckpt


def _interleave_qk_np(w: np.ndarray, num_heads: int) -> np.ndarray:
    """Numpy version of the QK head-dim interleave."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .transpose(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


# ════════════════════════════════════════════════════════════════════
#   Pi05JaxFrontendRtx — JAX Orbax frontend (thin shim over Pi05TorchFrontendRtx)
# ════════════════════════════════════════════════════════════════════


class Pi05JaxFrontendRtx(Pi05TorchFrontendRtx):
    """RTX consumer GPU Pi0.5 frontend backed by a JAX Orbax checkpoint.

    This class is a **thin override** of :class:`Pi05TorchFrontendRtx`: only the
    weight loader changes (Orbax instead of safetensors). Everything else
    — FP8 quantize, decoder style precompute, FP8 calibration, CUDA Graph
    capture, the ``infer`` hot path — is shared with the torch path. The
    pipeline math is therefore byte-identical between the two frontends.

    Future revision will move both frontends onto a pure-C BF16 FMHA
    backend so the JAX path can drop the torch dependency entirely.
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        chunk_size: int = CHUNK_SIZE,
        max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
        vision_pool_factor: int = 1,
        vision_num_layers: Optional[int] = None,
        cache_frames: int = 1,
        use_fp8: bool = True,
        hardware: Optional[str] = None,
        fp8_layout: Optional[str] = None,
        robot_action_dim: Optional[int] = None,
    ):
        # Don't chain to Pi05TorchFrontendRtx.__init__ — it expects a safetensors
        # file. We replicate the body and swap the loader.
        from flash_rt.core.utils.actions import LIBERO_ACTION_DIM
        from flash_rt.models.pi05.pipeline_rtx import ACTION_DIM as _ACTION_DIM

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        # See Pi05TorchFrontendRtx.__init__ for rationale: defaults to 7
        # (LIBERO); pass robot_action_dim=16 for OpenArm bimanual.
        if robot_action_dim is None:
            robot_action_dim = int(
                os.environ.get("FLASHRT_ROBOT_ACTION_DIM", LIBERO_ACTION_DIM))
        if not 1 <= robot_action_dim <= _ACTION_DIM:
            raise ValueError(
                f"robot_action_dim must be in [1, {_ACTION_DIM}], got {robot_action_dim}")
        self.robot_action_dim = int(robot_action_dim)
        self.max_prompt_len = int(max_prompt_len)
        self._num_steps = int(num_steps)
        self._vision_pool_factor = int(vision_pool_factor)
        if self._num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self._num_steps}")
        if self._vision_pool_factor not in (1, 2, 4):
            raise ValueError(
                "vision_pool_factor must be one of {1, 2, 4}; "
                f"got {self._vision_pool_factor}")
        self._cache_frames = int(cache_frames)
        if self._cache_frames < 1:
            raise ValueError(f"cache_frames must be >= 1, got {self._cache_frames}")
        self._frame_count = 0
        self._vision_num_layers = (
            VIS_L if vision_num_layers is None else int(vision_num_layers)
        )
        if not 1 <= self._vision_num_layers <= VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {VIS_L}], "
                f"got {self._vision_num_layers}")
        self.use_fp8 = bool(use_fp8)
        self.fp8_layout = _select_fp8_layout(hardware, fp8_layout)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        # See Pi05TorchFrontendRtx.__init__ for the rationale; cuBLASLt's
        # per-shape tuned algo is cached on the shared GemmRunner across
        # pipeline rebuilds, so re-running autotune is both wasted work
        # and a known crash trigger after the 3rd rebuild on Spark/SM121.
        self._gemm_autotune_done = False
        # FP8 scales snapshot — see Pi05TorchFrontendRtx for the rationale.
        # Per-rebuild single-frame re-calibration produces scales that
        # don't cover diffusion noise variance, tanking cos from ~0.96
        # (80-sample multi-frame) to ~0.6 (1-sample).
        self._fp8_scales_snapshot: dict[str, np.ndarray] = {}
        # Pipeline cache (keyed by exact prompt_len) — see
        # Pi05TorchFrontendRtx.__init__ for the full rationale and the
        # deferred-varlen note. Must be declared here too because this
        # __init__ does NOT chain to super; the torch path is replicated
        # body-style (see comment near the top of this method).
        from flash_rt.models.pi05.pipeline_rtx import Pi05Pipeline
        self._pipeline_cache: dict[int, Pi05Pipeline] = {}
        self._pipeline_cache_warn_threshold = int(
            os.environ.get("FLASHRT_PIPELINE_CACHE_WARN", "8"))
        self.current_prompt_len = 0
        self.pipeline = None

        # RL CFG state — kept in sync with the torch frontend so the JAX
        # path goes through the same set_prompt / infer hot path. Both
        # default to None (= standard non-CFG inference).
        self._rl_config: Optional[dict] = None
        self._rl_current_prompt_text: Optional[str] = None
        self._force_int8_decoder = False
        self._use_int8_encoder = False
        self._int8_encoder_only = False
        self._use_int8_vision = False
        self._use_int8_vision_static = False
        env_force_bf16 = os.environ.get("FVK_PI05_RTX_FORCE_BF16", "0") == "1"
        self._force_bf16 = env_force_bf16 or not supports_fp8()

        # ── norm_stats (same locations as torch frontend) ──
        self._load_norm_stats(checkpoint_dir)

        # ── Load + convert Orbax ──
        params_dir = checkpoint_dir
        if (checkpoint_dir / "params").is_dir():
            # The "real" checkpoint root may be the parent — but the loader
            # autodetects, so just pass through.
            pass
        self._checkpoint_path = str(checkpoint_dir)
        raw_ckpt = convert_pi05_orbax(checkpoint_dir)

        self._ckpt_bf16 = {
            k: v.contiguous() if isinstance(v, torch.Tensor) else v
            for k, v in raw_ckpt.items()
        }
        self.embedding_weight = self._ckpt_bf16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps
        # (matches the torch frontend's pre-scaling step that bakes the
        # flow-matching residual coefficient into the weights).
        self._ckpt_bf16["decoder_action_out_proj_w"] = (
            self._ckpt_bf16["decoder_action_out_proj_w"] * (-1.0 / self._num_steps)
        )
        self._ckpt_bf16["decoder_action_out_proj_b"] = (
            self._ckpt_bf16["decoder_action_out_proj_b"] * (-1.0 / self._num_steps)
        )

        # ── FP8 quantize large GEMM weights (shared method) ──
        self._fp8_weights: dict = {}
        self._fp8_store: list = []
        self._int8_weights: dict = {}
        self._int8_store: list = []
        self._int8_weight_scales: dict[str, torch.Tensor] = {}
        if self.use_fp8 and not self._force_bf16:
            self._quantize_all_fp8()

        # ── Pre-compute decoder styles (shared helper) ──
        from flash_rt.frontends.torch.pi05_rtx import _precompute_decoder_styles
        self._precomputed_styles = _precompute_decoder_styles(
            self._ckpt_bf16, self.chunk_size, num_steps=self._num_steps
        )

        # ── Attention backend, fvk, GemmRunner, reusable buffers ──
        from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
        from flash_rt import flash_rt_kernels as fvk

        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.chunk_size,
            num_encoder_layers=ENC_L,
        )
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        IMG_HW = 224  # local — matches Pi05TorchFrontendRtx's constant
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=bf16, device="cuda"
        )
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda"
        )
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda"
        )
        # ── RTC soft-guidance staging tensors (Phase 6 / G11) ──
        # Body-replicated from Pi05TorchFrontendRtx.__init__ because this
        # __init__ does not chain to super (see comment up top). Without
        # these allocations the inherited ``_stage_rtc_inputs`` would
        # AttributeError on the first inference. Both tensors default to
        # zero so the captured pipeline kernel is a numerical no-op
        # (``v_new = v``) for non-RTC traffic. See
        # ``Pi05Pipeline._rtc_apply_guidance`` for the algorithm.
        from flash_rt.frontends.torch.pi05_rtx import (
            _RTC_DEFAULT_EXECUTION_HORIZON, _RTC_DEFAULT_SCHEDULE,
        )
        self._rtc_prev_chunk_buf = torch.zeros(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._rtc_weights_buf = torch.zeros(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._rtc_last_call_active = False
        self._rtc_execution_horizon: int = _RTC_DEFAULT_EXECUTION_HORIZON
        self._rtc_schedule: str = _RTC_DEFAULT_SCHEDULE
        from flash_rt.core.cuda_buffer import _cudart
        self._cudart = _cudart

        logger.info(
            "Pi05JaxFrontendRtx initialised (num_views=%d, chunk=%d, fp8_layout=%s)",
            self.num_views, self.chunk_size, self.fp8_layout,
        )
