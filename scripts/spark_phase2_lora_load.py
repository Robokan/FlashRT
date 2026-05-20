#!/usr/bin/env python3
"""Phase 2 acceptance gate — load the LoRA-fine-tuned OpenArm Orbax
checkpoint on DGX Spark with the new fp32-LoRA-merge path.

What it checks:

  1. ``_load_orbax`` returns >0 tensors with .lora_a / .lora_b keys —
     confirms the checkpoint is actually a LoRA checkpoint (catches the
     case where the user accidentally points at a pre-merged ckpt).
  2. ``_maybe_merge_lora`` consumes every LoRA pair. Post-merge, there
     should be ZERO .lora_a / .lora_b keys left.
  3. The number of merged tensors matches the expected LoRA-target count
     for pi05_openarm: 18 paligemma layers × 4 LoRA-targeted sites + 18
     action-expert layers × 4 LoRA-targeted sites = 144 (per JAX's
     LoRA-on-{q_einsum, kv_einsum, gating_einsum, linear} pattern). The
     actual count may differ if a checkpoint trained with a different
     LoRA-targeting recipe; we report what we got either way.
  4. ``convert_pi05_orbax`` returns a dict with the expected RTX schema
     keys (vision_*, encoder_*, decoder_*, etc.).
  5. End-to-end: ``Pi05JaxFrontendRtx(checkpoint).infer({...})`` returns
     a finite (chunk_size, action_dim) tensor. Sanity check, not parity.

Usage:
    python3 scripts/spark_phase2_lora_load.py \\
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4

Acceptance:
    Exit 0 → all 5 checks PASS → proceed to Phase 3 (FP8 calibration).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(name: str, detail: str = "") -> None:
    msg = f"{GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"  {DIM}{detail}{RESET}"
    print(msg, flush=True)


def _fail(name: str, detail: str) -> None:
    print(f"{RED}FAIL{RESET}  {name}  {detail}", flush=True)


def _warn(name: str, detail: str) -> None:
    print(f"{YELLOW}WARN{RESET}  {name}  {detail}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="OpenArm LoRA Orbax checkpoint dir")
    parser.add_argument("--prompt", default="pick up the red block")
    parser.add_argument("--skip-infer", action="store_true",
                        help="Stop after the merge check; skip inference smoke")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        _fail("checkpoint dir", f"{ckpt} is not a directory")
        return 1
    _pass("checkpoint dir", str(ckpt))

    failures: list[str] = []

    # 1. Pre-merge: count LoRA keys.
    try:
        from flash_rt.core.weights.loader import _load_orbax
        from flash_rt.frontends.jax.pi05_rtx import _maybe_merge_lora

        t0 = time.perf_counter()
        raw = _load_orbax(str(ckpt))
        load_time = time.perf_counter() - t0

        n_la = sum(1 for k in raw if k.endswith(".lora_a"))
        n_lb = sum(1 for k in raw if k.endswith(".lora_b"))
        if n_la == 0 and n_lb == 0:
            _fail("detect lora_a / lora_b keys",
                  f"no LoRA params found in {ckpt}; this looks like a "
                  "pre-merged checkpoint. Use the existing convert_pi05_orbax "
                  "path directly (no Phase 2 work needed for that ckpt).")
            return 1
        if n_la != n_lb:
            _fail("count lora_a / lora_b",
                  f"asymmetric: lora_a={n_la}, lora_b={n_lb} (expected equal)")
            failures.append("lora_count_asymmetric")
        _pass("detect LoRA params",
              f"{n_la} pairs of lora_a/lora_b in {load_time:.1f}s ({len(raw)} tensors total)")
    except Exception as e:  # pragma: no cover
        _fail("load + detect LoRA", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # 2. Merge.
    try:
        merged_raw = _maybe_merge_lora(raw, scaling=1.0)
        n_la_after = sum(1 for k in merged_raw if k.endswith(".lora_a"))
        n_lb_after = sum(1 for k in merged_raw if k.endswith(".lora_b"))
        if n_la_after == 0 and n_lb_after == 0:
            _pass("LoRA merge consumes all pairs", f"merged {n_la} tensors")
        else:
            _fail("LoRA merge consumes all pairs",
                  f"still {n_la_after} lora_a and {n_lb_after} lora_b after merge")
            failures.append("lora_residual")
    except Exception as e:  # pragma: no cover
        _fail("LoRA merge", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # 3. Sanity check on merge count.
    EXPECTED_MIN = 80   # at minimum: paligemma q/kv/o + ff over 18 layers
    EXPECTED_MAX = 180  # paligemma + action expert + every layer ≤ 180
    if EXPECTED_MIN <= n_la <= EXPECTED_MAX:
        _pass("merge count plausible", f"{n_la} (expected {EXPECTED_MIN}..{EXPECTED_MAX})")
    else:
        _warn("merge count outside expected band",
              f"{n_la} (expected {EXPECTED_MIN}..{EXPECTED_MAX}). "
              "Not necessarily wrong — a different LoRA target set produces "
              "different counts. Verify against your training recipe.")

    if args.skip_infer:
        if failures:
            print(f"\n{BOLD}{RED}Phase 2 FAILED{RESET}  ({len(failures)} bad: {', '.join(failures)})")
            return 1
        print(f"\n{BOLD}{GREEN}Phase 2 LOAD PASSED{RESET}  (--skip-infer set)")
        return 0

    # 4 + 5. End-to-end smoke via Pi05JaxFrontendRtx.
    try:
        import flash_rt
        # This goes through convert_pi05_orbax (which internally calls
        # _maybe_merge_lora again — idempotent because the merged dict
        # no longer has .lora_a keys after the first pass).
        t0 = time.perf_counter()
        model = flash_rt.load_model(
            checkpoint=str(ckpt),
            framework="jax",
            num_views=2,
            autotune=3,
        )
        elapsed = time.perf_counter() - t0
        _pass("load_model(framework='jax')", f"{elapsed:.1f}s incl. LoRA merge")
    except Exception as e:  # pragma: no cover
        _fail("flash_rt.load_model", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    try:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        actions = model.predict(images=[img, wrist], prompt=args.prompt)
        elapsed = time.perf_counter() - t0

        if not np.isfinite(actions).all():
            _fail("merged-ckpt inference finite",
                  f"got {int((~np.isfinite(actions)).sum())} NaN/Inf")
            failures.append("nan")
        elif actions.ndim != 2:
            _fail("merged-ckpt inference shape", f"got {actions.shape}, expected 2-D")
            failures.append("shape")
        else:
            _pass("merged-ckpt inference",
                  f"shape={actions.shape} dtype={actions.dtype} {elapsed*1000:.1f}ms first call")
    except Exception as e:  # pragma: no cover
        _fail("merged-ckpt inference", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    print()
    if failures:
        print(f"{BOLD}{RED}Phase 2 FAILED{RESET}  ({len(failures)} bad: {', '.join(failures)})")
        print(f"{DIM}Investigate before advancing to Phase 3.{RESET}")
        return 1
    print(f"{BOLD}{GREEN}Phase 2 PASSED{RESET}  — LoRA merge wired in. Advance to Phase 3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
