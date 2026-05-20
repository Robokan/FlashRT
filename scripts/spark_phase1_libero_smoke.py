#!/usr/bin/env python3
"""Phase 1 acceptance gate — load pi05_libero on DGX Spark and infer.

This is the cheaper precursor to the full LIBERO eval. It only depends
on the FlashRT install — no LIBERO simulator, no MuJoCo. If this fails,
the simulator eval will fail for the same reason but slower and with
noisier output.

What it checks:
  1. `flash_rt.load_model(framework="jax")` loads the Orbax checkpoint.
  2. First infer (which includes calibration + CUDA Graph capture) runs
     to completion. ~3 s expected; will time out at 60 s if cuBLASLt
     heuristics fail to resolve an SM_121 tactic.
  3. Steady-state inference: 20 calls. Reports mean + p50 + p99 latency.
     Acceptance gate: mean < 200 ms (RTX 5090 is ~18 ms with 2 views;
     Spark's GB10 is roughly 2-3x slower at peak FP8, so 50-150 ms
     expected. 200 ms is a soft ceiling that catches kernel-fallback
     pathologies without being so tight we trip on first-call overhead.)
  4. Output shape == (chunk_size, action_dim) — LIBERO uses 10 timesteps
     and 7 dims. Confirms the pipeline ran end-to-end without truncation.
  5. All outputs finite (no NaN / Inf).

Usage:
    # Inside the flashrt:spark container:
    python3 scripts/spark_phase1_libero_smoke.py \\
        --checkpoint /openpi_assets/pi05_libero

    # With explicit num-iter override for longer steady-state sample:
    python3 scripts/spark_phase1_libero_smoke.py \\
        --checkpoint /openpi_assets/pi05_libero --num-iter 100

Acceptance:
    Exit 0 → all 5 checks PASS → proceed to full LIBERO eval (see
        scripts/spark_phase1_libero_run.sh).
    Exit 1 → at least one check FAILED → don't run the full eval.
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
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

CHUNK_SIZE_EXPECTED = 10  # Pi0.5 LIBERO ships with chunk_size=10
ACTION_DIM_EXPECTED = 7   # LIBERO is 7-DOF (xyz + rot + gripper)
LATENCY_CEILING_MS = 200.0


def _pass(name: str, detail: str = "") -> None:
    msg = f"{GREEN}PASS{RESET}  {name}"
    if detail:
        msg += f"  {DIM}{detail}{RESET}"
    print(msg, flush=True)


def _fail(name: str, detail: str) -> None:
    print(f"{RED}FAIL{RESET}  {name}  {detail}", flush=True)


def _synth_obs(rng: np.random.Generator) -> dict:
    """Realistic-shaped synthetic observation for Pi0.5 LIBERO.

    Real images are uint8 RGB 224x224. Random uint8 is a more honest
    workload for the FP8 calibration / quantize path than zeros (which
    would let cuBLASLt pick degenerate tactics that don't represent
    inference SASS).
    """
    img = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    return {"image": img, "wrist_image": wrist}


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 smoke test for FlashRT on Spark")
    parser.add_argument("--checkpoint", required=True,
                        help="Pi0.5 Orbax JAX checkpoint dir")
    parser.add_argument("--num-iter", type=int, default=20,
                        help="Number of steady-state inference calls (default 20)")
    parser.add_argument("--prompt", default="pick up the alphabet soup",
                        help="Task prompt (default is a real LIBERO task)")
    parser.add_argument("--autotune", type=int, default=3,
                        help="CUDA Graph autotune trials (0..5, default 3)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_dir():
        _fail("checkpoint dir", f"{ckpt} is not a directory")
        return 1
    _pass("checkpoint dir", str(ckpt))

    failures: list[str] = []
    rng = np.random.default_rng(0)

    # 1. Load.
    try:
        import flash_rt
        t0 = time.perf_counter()
        model = flash_rt.load_model(
            checkpoint=str(ckpt),
            framework="jax",
            num_views=2,
            autotune=args.autotune,
        )
        elapsed = time.perf_counter() - t0
        _pass("flash_rt.load_model(framework='jax')", f"{elapsed:.1f}s")
    except Exception as e:  # pragma: no cover
        _fail("flash_rt.load_model(framework='jax')", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # 2. First infer (calibration + graph capture).
    first_obs = _synth_obs(rng)
    try:
        t0 = time.perf_counter()
        # VLAModel.predict accepts the dict form directly; first call
        # triggers calibrate_with_real_data internally.
        actions = model.predict(
            images=[first_obs["image"], first_obs["wrist_image"]],
            prompt=args.prompt,
        )
        elapsed = time.perf_counter() - t0
        _pass("first infer (calibration + graph capture)", f"{elapsed:.2f}s")
    except Exception as e:  # pragma: no cover
        _fail("first infer", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # 3+4+5. Steady-state loop.
    latencies_ms: list[float] = []
    n_nan = 0
    bad_shape = None
    try:
        for i in range(args.num_iter):
            obs = _synth_obs(rng)
            t0 = time.perf_counter()
            actions = model.predict(
                images=[obs["image"], obs["wrist_image"]],
                prompt=args.prompt,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)
            if not np.isfinite(actions).all():
                n_nan += int((~np.isfinite(actions)).sum())
            if actions.shape != (CHUNK_SIZE_EXPECTED, ACTION_DIM_EXPECTED) and bad_shape is None:
                bad_shape = actions.shape
    except Exception as e:  # pragma: no cover
        _fail("steady-state loop", f"{type(e).__name__}: {e} at iter {i}")
        traceback.print_exc()
        return 1

    lats = np.asarray(latencies_ms)
    mean = float(lats.mean())
    p50 = float(np.percentile(lats, 50))
    p99 = float(np.percentile(lats, 99))

    if mean < LATENCY_CEILING_MS:
        _pass("steady-state latency",
              f"mean={mean:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms (ceiling={LATENCY_CEILING_MS:.0f}ms)")
    else:
        _fail("steady-state latency",
              f"mean={mean:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms — above ceiling {LATENCY_CEILING_MS:.0f}ms")
        failures.append("latency")

    if bad_shape is None:
        _pass("output shape", f"({CHUNK_SIZE_EXPECTED}, {ACTION_DIM_EXPECTED})")
    else:
        _fail("output shape",
              f"got {bad_shape}, expected ({CHUNK_SIZE_EXPECTED}, {ACTION_DIM_EXPECTED})")
        failures.append("shape")

    if n_nan == 0:
        _pass("finite outputs", f"{args.num_iter * CHUNK_SIZE_EXPECTED * ACTION_DIM_EXPECTED} values, 0 NaN/Inf")
    else:
        _fail("finite outputs", f"{n_nan} NaN/Inf across {args.num_iter} iterations")
        failures.append("nan")

    print()
    if failures:
        print(f"{BOLD}{RED}Phase 1 SMOKE FAILED{RESET}  ({len(failures)} check(s) bad: {', '.join(failures)})")
        print(f"{DIM}Do not run the full LIBERO eval until this script exits 0.{RESET}")
        return 1
    print(f"{BOLD}{GREEN}Phase 1 SMOKE PASSED{RESET}")
    print(f"{DIM}Next: run the full LIBERO eval via scripts/spark_phase1_libero_run.sh{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
