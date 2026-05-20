#!/usr/bin/env python3
"""Phase 0 acceptance gate — verify the FlashRT build is healthy on DGX Spark.

This script is the gate between Phase 0 (port FlashRT to Spark) and
Phase 1 (run pi05_libero). It checks:

  1. `import flash_rt` works (Python install / .so linkage OK)
  2. `flash_rt_kernels.so` and `flash_rt_fa2.so` are present (consumer
     Blackwell SM_121 should produce both)
  3. The host is aarch64 (sanity-check: don't accept a smoke pass on a
     desktop x86_64 box that happens to have a 5090 in it)
  4. Hardware-detection returns SM == 121
  5. supports_fp8() returns True (SM ≥ 89)
  6. supports_nvfp4() returns True (SM ≥ 120 — the SM_121-fix is what
     makes this pass on Spark)
  7. The bundled kernels actually launch on the GPU — small bf16 fwd
     through fa2.fwd_bf16 with realistic Pi0.5 shapes. NaN-free output
     confirms cuBLASLt heuristics resolved an SM_121-compatible tactic.

Exits non-zero on any failure with a clear message. The Dockerfile.spark
RUN of this script makes the docker build fail rather than producing a
broken image, matching the existing Dockerfile / Dockerfile.thor pattern.

Usage (inside the container):
    python3 scripts/spark_build_smoke.py

Acceptance:
    All 7 checks PASS → Phase 0 complete → proceed to Phase 1.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback
from pathlib import Path


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(name: str, detail: str = "") -> None:
    msg = f"{GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"  {DIM}{detail}{RESET}"
    print(msg)


def _fail(name: str, detail: str) -> None:
    print(f"{RED}FAIL{RESET}  {name}  {detail}")


def main() -> int:
    failures: list[str] = []

    # 1. Import works.
    try:
        import flash_rt  # noqa: F401
        _pass("import flash_rt", f"version={getattr(flash_rt, '__version__', '?')}")
    except Exception as e:  # pragma: no cover
        _fail("import flash_rt", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # 2. The two .so artifacts a Spark build should produce both exist.
    flash_rt_dir = Path(flash_rt.__file__).resolve().parent
    so_root = flash_rt_dir
    expected = ("flash_rt_kernels", "flash_rt_fa2")
    for stem in expected:
        # Match cpython-312-aarch64-linux-gnu.so (or x86_64 if mis-platformed).
        hits = list(so_root.glob(f"{stem}*.so"))
        if not hits:
            _fail(f"locate {stem}.so", f"no match under {so_root}")
            failures.append(stem)
        else:
            _pass(f"locate {stem}.so", str(hits[0].name))

    # 3. Host arch sanity (aarch64).
    machine = platform.machine()
    if machine == "aarch64":
        _pass("host arch", "aarch64 (Grace)")
    else:
        _fail("host arch", f"got '{machine}', expected 'aarch64' (DGX Spark Grace)")
        failures.append("host_arch")

    # 4–6. Hardware-detection probes.
    try:
        from flash_rt.core.utils import hardware

        sm = hardware.get_gpu_sm_version()
        if sm == 121:
            _pass("get_gpu_sm_version", "121 (DGX Spark GB10)")
        elif sm == 120:
            _fail("get_gpu_sm_version", "120 — that's RTX 5090, not Spark")
            failures.append("sm_version")
        else:
            _fail("get_gpu_sm_version", f"got {sm}, expected 121")
            failures.append("sm_version")

        if hardware.supports_fp8():
            _pass("supports_fp8()", "True")
        else:
            _fail("supports_fp8()", "False — SM < 89")
            failures.append("fp8")

        if hardware.supports_nvfp4():
            _pass("supports_nvfp4()", "True")
        else:
            _fail("supports_nvfp4()", "False — SM < 120")
            failures.append("nvfp4")

        name = hardware.get_gpu_name()
        # DGX Spark has not picked a stable marketing name in nvidia-smi as
        # of writing; accept any string but record it.
        _pass("get_gpu_name", repr(name))
    except Exception as e:  # pragma: no cover
        _fail("hardware probes", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        failures.append("hardware_probes")

    # 7. Run a small fa2 kernel to confirm cuBLASLt resolves an SM_121
    #    tactic and the result is finite. Uses Pi0.5-realistic shapes
    #    (batch=1, seqlen=1024, num_heads=8, head_dim=256) chosen to
    #    stress the same code path as Pi0.5 inference.
    #
    #    flash_rt_fa2.fwd_bf16 is the low-level pybind ABI: it expects
    #    raw device pointers + a pre-allocated O and softmax_lse, in
    #    BSHD layout (matches flash_rt/hardware/rtx/attn_backend.py
    #    `_call_fvk_fa2`). Mirror that call shape exactly so this smoke
    #    exercises the same code path Pi0.5 will hit at runtime.
    try:
        import torch

        from flash_rt import flash_rt_fa2  # type: ignore

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() == False")

        device = torch.device("cuda")
        # FA2 BSHD layout: (batch, seqlen, num_heads, head_dim)
        B, S, H, D = 1, 1024, 8, 256
        torch.manual_seed(0)
        q = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, S, H, D, dtype=torch.bfloat16, device=device)
        o = torch.empty(B, S, H, D, dtype=torch.bfloat16, device=device)
        # softmax_lse: (B, H, seqlen_q) fp32, rounded up to multiple of 128
        S_lse = ((S + 127) // 128) * 128
        lse = torch.empty(B, H, S_lse, dtype=torch.float32, device=device)
        num_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()).multi_processor_count

        flash_rt_fa2.fwd_bf16(
            Q=q.data_ptr(), K=k.data_ptr(), V=v.data_ptr(),
            O=o.data_ptr(), softmax_lse=lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0,
            batch=B, seqlen_q=S, seqlen_k=S,
            num_heads_q=H, num_heads_kv=H, head_dim=D,
            q_strides=(q.stride(0), q.stride(1), q.stride(2)),
            k_strides=(k.stride(0), k.stride(1), k.stride(2)),
            v_strides=(v.stride(0), v.stride(1), v.stride(2)),
            o_strides=(o.stride(0), o.stride(1), o.stride(2)),
            softmax_scale=1.0 / (D ** 0.5),
            num_sms=num_sms,
            stream=0,
        )
        torch.cuda.synchronize()

        if not torch.isfinite(o).all():
            n_nan = int(torch.isnan(o).sum())
            n_inf = int(torch.isinf(o).sum())
            _fail("fa2.fwd_bf16 finite", f"got {n_nan} NaN, {n_inf} Inf in O")
            failures.append("fa2_kernel")
        elif o.shape != (B, S, H, D):
            _fail("fa2.fwd_bf16 shape", f"got {tuple(o.shape)}, expected {(B, S, H, D)}")
            failures.append("fa2_kernel")
        elif float(o.abs().mean()) < 1e-6:
            _fail("fa2.fwd_bf16 nontrivial",
                  f"|O|.mean()={float(o.abs().mean()):.2e} — kernel may not have written")
            failures.append("fa2_kernel")
        else:
            _pass("fa2.fwd_bf16 launch",
                  f"shape={tuple(o.shape)} dtype={o.dtype} "
                  f"|O|.mean()={float(o.abs().mean()):.3f}")
    except Exception as e:  # pragma: no cover
        _fail("fa2 kernel launch", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        failures.append("fa2_kernel")

    print()
    if failures:
        print(f"{BOLD}{RED}Phase 0 FAILED{RESET}  ({len(failures)} check(s) bad: {', '.join(failures)})")
        print(f"{DIM}Do not advance to Phase 1 until this script exits 0.{RESET}")
        return 1
    print(f"{BOLD}{GREEN}Phase 0 PASSED{RESET}  — FlashRT runs on DGX Spark. Advance to Phase 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
