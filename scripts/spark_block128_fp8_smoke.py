#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Block-128 FP8 GEMM smoke test on Spark SM_121.

Validates that `flash_rt.flash_rt_kernels.fp8_block128_gemm_cutlass_sm120_bf16out`
compiles, runs, and produces numerically correct output on Spark (GB10
SM_121) for the Pi0.5 encoder ffn_down shape (M=896, N=2048, K=16384).

Used by the Spark port to confirm the block-128 CUTLASS infrastructure
is real and usable before committing to "Path B' per-layer mixed-FP8
granularity" implementation work — see docs/spark_status.md, section
"Block-128 FP8 smoke test on SM_121 (2026-05-21)".

Three checks:

  (1) BASE — random bf16 inputs, no outliers. Verifies the kernel
      computes the GEMM at all and that cos(block128, bf16_ref) > 0.99.
  (2) OUTLIER — sparse extreme outliers across 8 channels (|x| ≈ 25,
      median |x| ≈ 0.003, ratio ~14000×) modelled on the real
      encoder_ffn_down_w_16 amax pattern. Compares block-128 FP8 against
      per-tensor FP8 (emulated) and BF16 reference; reports per-token
      cosine distribution.
  (3) LATENCY — wall-clock per-call on the Pi0.5 ffn_down shape.

Run from the repo root:

    .venv/bin/python3 scripts/spark_block128_fp8_smoke.py
"""
from __future__ import annotations

import os
import sys
import time

# Make the repo importable when invoked from anywhere.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

import flash_rt.flash_rt_kernels as fvk

DEV = torch.device("cuda")

# Pi0.5 encoder ffn_down shape on Spark with 3 cams + max_prompt_len=128:
#   M = vision_seq_enc(768) + max_prompt_len(128) = 896
#   N = D (Gemma-2B model dim, 2048)
#   K = ffn_intermediate (4 * D for Gemma 2B, but pipeline uses 16384)
M, N, K = 896, 2048, 16384
FP8_MAX = 448.0


def bf16_to_per_token_block128_fp8(x_bf16: torch.Tensor):
    """Quantize bf16 (M, K) to per-token per-K-block-128 FP8 via fvk kernel."""
    M_, K_ = x_bf16.shape
    assert K_ % 128 == 0
    out_fp8 = torch.empty((M_, K_), dtype=torch.float8_e4m3fn, device=DEV)
    scale = torch.empty((M_, K_ // 128), dtype=torch.float32, device=DEV)
    fvk.fp8_per_token_block128_quant_bf16(
        x_bf16.data_ptr(), out_fp8.data_ptr(), scale.data_ptr(),
        M_, K_, torch.cuda.current_stream().cuda_stream,
    )
    return out_fp8, scale


def bf16_to_block128_fp8_weight(w_bf16: torch.Tensor):
    """Quantize bf16 (N, K) weight to per-128-N-block per-128-K-block FP8.

    Returns (q_e4m3, scale_fp32) where scale.shape = (N//128, K//128).
    Pure Python (one-shot at load time, not on hot path).
    """
    N_, K_ = w_bf16.shape
    assert N_ % 128 == 0 and K_ % 128 == 0
    blocks = w_bf16.float().reshape(N_ // 128, 128, K_ // 128, 128)
    block_amax = blocks.abs().amax(dim=(1, 3))
    scale = (block_amax / FP8_MAX).clamp(min=1e-12)
    inv_scale = (1.0 / scale).reshape(N_ // 128, 1, K_ // 128, 1)
    w_scaled = (blocks * inv_scale).clamp(-FP8_MAX, FP8_MAX)
    return w_scaled.reshape(N_, K_).to(torch.float8_e4m3fn), scale.contiguous()


def block128_gemm(a_fp8, a_scale, w_fp8, w_scale, M_, N_, K_):
    out = torch.empty((M_, N_), dtype=torch.bfloat16, device=DEV)
    fvk.fp8_block128_gemm_cutlass_sm120_bf16out(
        a_fp8.data_ptr(), w_fp8.data_ptr(), out.data_ptr(),
        M_, N_, K_,
        a_scale.data_ptr(), w_scale.data_ptr(),
        torch.cuda.current_stream().cuda_stream,
    )
    return out


def bf16_reference(a_bf16, w_bf16):
    """Ground-truth GEMM in fp32 cast back to bf16: (M,K) @ (N,K).T -> (M,N)."""
    return (a_bf16.float() @ w_bf16.float().T).bfloat16()


def per_tensor_fp8(a_bf16, w_bf16):
    """Per-tensor FP8 reference (one scalar each, dequant-then-matmul)."""
    a_amax = a_bf16.abs().max().float().item()
    w_amax = w_bf16.abs().max().float().item()
    a_scale = max(a_amax / FP8_MAX, 1e-12)
    w_scale = max(w_amax / FP8_MAX, 1e-12)
    a_q = (a_bf16.float() / a_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    w_q = (w_bf16.float() / w_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return ((a_q.float() * a_scale) @ (w_q.float() * w_scale).T).bfloat16()


def cos_sim(a, b):
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-9)).item()


def rel_err(pred, ref):
    return ((pred.float() - ref.float()).norm() / (ref.float().norm() + 1e-9)).item()


def bench(fn, name, n=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    per_call_us = (time.perf_counter() - t0) / n * 1e6
    print(f"  {name}: {per_call_us:.1f} us/call ({n} iters)")
    return per_call_us


def main() -> int:
    print(f"=== block-128 FP8 GEMM smoke test ===")
    print(f"Pi0.5 encoder ffn_down shape: M={M}, N={N}, K={K}")
    print(f"Device: {torch.cuda.get_device_name(0)}, "
          f"capability={torch.cuda.get_device_capability(0)}")
    print()

    torch.manual_seed(42)
    np.random.seed(42)

    # (1) BASE TEST
    print("--- (1) Base correctness (random bf16, no outliers) ---")
    a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.05
    w_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=DEV) * 0.03

    a_fp8, a_scale = bf16_to_per_token_block128_fp8(a_bf16)
    w_fp8, w_scale = bf16_to_block128_fp8_weight(w_bf16)
    torch.cuda.synchronize()

    out_block128 = block128_gemm(a_fp8, a_scale, w_fp8, w_scale, M, N, K)
    torch.cuda.synchronize()
    out_pertensor = per_tensor_fp8(a_bf16, w_bf16)
    out_ref = bf16_reference(a_bf16, w_bf16)

    base_cos_b128 = cos_sim(out_block128, out_ref)
    base_cos_pt = cos_sim(out_pertensor, out_ref)
    print(f"  cos(block128, bf16_ref):   {base_cos_b128:.6f}")
    print(f"  cos(per-tensor, bf16_ref): {base_cos_pt:.6f}")
    print(f"  rel_err(block128):   {rel_err(out_block128, out_ref):.4f}")
    print(f"  rel_err(per-tensor): {rel_err(out_pertensor, out_ref):.4f}")
    print()

    # (2) OUTLIER TEST — match real layer-16 amax pattern.
    print("--- (2) Outlier test (sparse extreme outliers, 14000x ratio) ---")
    a_out = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.005
    outlier_channels = [100, 247, 1500, 3000, 5050, 7700, 9999, 12345]
    for ch in outlier_channels:
        a_out[:, ch] = (torch.randn(M, dtype=torch.bfloat16, device=DEV) * 6.0
                        + (25.0 if ch % 2 == 0 else -25.0))

    ratio = a_out.abs().max().item() / a_out.abs().median().item()
    print(f"  Activation amax/median ratio: {ratio:.0f}x")

    a_fp8_out, a_scale_out = bf16_to_per_token_block128_fp8(a_out)
    out_b128_out = block128_gemm(a_fp8_out, a_scale_out, w_fp8, w_scale, M, N, K)
    torch.cuda.synchronize()
    out_pt_out = per_tensor_fp8(a_out, w_bf16)
    out_ref_out = bf16_reference(a_out, w_bf16)

    # Per-token cosine over all M tokens.
    b128_pt = np.array([cos_sim(out_b128_out[t], out_ref_out[t]) for t in range(M)])
    pt_pt = np.array([cos_sim(out_pt_out[t], out_ref_out[t]) for t in range(M)])
    print(f"  Per-token cos vs bf16_ref:")
    print(f"               median   p10     min     p99")
    print(f"    block-128 {np.median(b128_pt):.4f}   "
          f"{np.percentile(b128_pt, 10):.4f}  "
          f"{b128_pt.min():.4f}  "
          f"{np.percentile(b128_pt, 99):.4f}")
    print(f"    per-tensor {np.median(pt_pt):.4f}   "
          f"{np.percentile(pt_pt, 10):.4f}  "
          f"{pt_pt.min():.4f}  "
          f"{np.percentile(pt_pt, 99):.4f}")
    print()

    # (3) LATENCY
    print("--- (3) Latency on Pi0.5 ffn_down shape ---")
    block128_us = bench(
        lambda: block128_gemm(a_fp8, a_scale, w_fp8, w_scale, M, N, K),
        "block-128 FP8 (CUTLASS SM120a)",
    )
    print()

    # Summary + pass/fail gate.
    print("--- Summary ---")
    print(f"  Base cos: block-128 = {base_cos_b128:.4f}, "
          f"per-tensor = {base_cos_pt:.4f}")
    print(f"  Outlier cos (per-token median): "
          f"block-128 = {np.median(b128_pt):.4f}, "
          f"per-tensor = {np.median(pt_pt):.4f}")
    print(f"  Throughput: {block128_us:.1f} us/call")

    # Acceptance: kernel must compile + run + produce cos > 0.99 on base.
    ok = base_cos_b128 > 0.99 and np.median(b128_pt) > 0.99
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
