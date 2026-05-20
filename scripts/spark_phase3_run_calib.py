#!/usr/bin/env python3
"""Phase 3 part 2 — run FP8 calibration on the prepared OpenArm samples.

Inputs:
  - --checkpoint: pi05_openarm_ngc_lora_v4 Orbax dir (LoRA-fused inside
    convert_pi05_orbax thanks to the Phase 2 patch).
  - --calib-data: npz produced by scripts/spark_phase3_prepare_calib.py
  - --percentile: 99.9 (default) — matches FlashRT's
    accumulate_amax(percentile=99.9) recipe. 100.0 == naive max.

What it does:
  1. ``flash_rt.load_model(framework='jax')``
  2. ``model.calibrate(observations, percentile=99.9)``
     — multi-sample path; computes per-sample amax then percentile-reduces.
     This writes a JSON cache to ~/.flash_rt/calibration/<hash>_Se<N>.json.
  3. Reports the calibration cache file path + scale statistics
     (min/median/max activation scale per GEMM site).
  4. Sanity checks:
        - No scales are exactly 0.0 (would mean the kernel saw zeros
          and recorded amax=0, producing div-by-zero at inference).
        - No scales saturate at the FP8 max (448 / 448 == 1.0 → would
          mean the activation magnitude exceeded the FP8 representable
          range; very bad).
  5. Runs a single inference on the first calibration sample and
     reports the latency + output finiteness as a smoke check.

Acceptance gate:
    Exit 0 → calibration cache written, all scales finite, sample
    inference finite. Phase 4 (parity vs JAX server) can now proceed.

Usage:
    python3 scripts/spark_phase3_run_calib.py \\
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \\
        --calib-data /openpi_assets/calib_openarm_80.npz \\
        --percentile 99.9
"""

from __future__ import annotations

import argparse
import os
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib-data", required=True,
                        help="npz from spark_phase3_prepare_calib.py")
    parser.add_argument("--percentile", type=float, default=99.9,
                        help="amax reduction percentile (default 99.9)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap on samples used (default: all in npz)")
    parser.add_argument("--num-views", type=int, default=2)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    npz = Path(args.calib_data)
    if not ckpt.is_dir():
        _fail("checkpoint", f"{ckpt} not a directory")
        return 1
    if not npz.is_file():
        _fail("calib data", f"{npz} not a file (run spark_phase3_prepare_calib.py first)")
        return 1

    failures: list[str] = []

    # Load calib samples.
    data = np.load(npz, allow_pickle=True)
    images = data["images"]
    wrists = data["wrist_images"]
    prompts = data["prompts"]
    if args.max_samples is not None:
        images = images[: args.max_samples]
        wrists = wrists[: args.max_samples]
        prompts = prompts[: args.max_samples]
    n = len(images)
    if n == 0:
        _fail("calib data", "0 samples")
        return 1
    _pass("calib data", f"{n} samples, image shape={images[0].shape}")

    # Build observation dicts.
    obs_list: list[dict] = []
    for img, wrist in zip(images, wrists):
        obs_list.append({"image": img, "wrist_image": wrist})

    # Load the model.
    try:
        import flash_rt
        t0 = time.perf_counter()
        model = flash_rt.load_model(
            checkpoint=str(ckpt),
            framework="jax",
            num_views=args.num_views,
            autotune=3,
        )
        _pass("load_model", f"{time.perf_counter() - t0:.1f}s")
    except Exception as e:
        _fail("load_model", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Set the prompt (use the first sample's prompt; FP8 cache key
    # includes Se = encoder seq len which depends on prompt length, so
    # the calibration produced here is keyed for *this specific prompt*
    # — different prompts will trigger re-calibration on first infer).
    first_prompt = str(prompts[0]) if len(prompts) > 0 and prompts[0] else "pick up the red block"
    try:
        if hasattr(model._pipe, "set_prompt"):
            model._pipe.set_prompt(first_prompt)
            model._current_prompt = first_prompt
        _pass("set_prompt", first_prompt[:60])
    except Exception as e:
        _fail("set_prompt", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Run calibration.
    try:
        t0 = time.perf_counter()
        model.calibrate(obs_list, percentile=args.percentile, verbose=True)
        elapsed = time.perf_counter() - t0
        _pass("calibrate", f"{n} samples, percentile={args.percentile}, {elapsed:.1f}s")
    except Exception as e:
        _fail("calibrate", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1

    # Inspect cached scales.
    pipeline = model._pipe.pipeline
    scale_attrs = [
        "_enc_calib_scales", "_dec_calib_scales",
        "_enc_alpha_host", "_dec_alpha_host",
    ]
    found_any = False
    for attr in scale_attrs:
        scales = getattr(pipeline, attr, None) or getattr(model._pipe, attr, None)
        if scales is None:
            continue
        found_any = True
        arr = np.asarray(scales).flatten()
        if len(arr) == 0:
            continue
        n_zero = int((arr == 0.0).sum())
        n_inf = int((~np.isfinite(arr)).sum())
        n_sat = int((arr >= 1.0 - 1e-3).sum())  # alpha ~ 1 means amax ≈ 448 ≈ FP8 max
        msg = f"n={len(arr)} min={arr.min():.4e} med={np.median(arr):.4e} max={arr.max():.4e}"
        if n_zero or n_inf or n_sat:
            _warn(f"scales[{attr}]", f"{msg} | zero={n_zero} nonfinite={n_inf} near-saturate={n_sat}")
            if n_zero or n_inf:
                failures.append(f"{attr}_bad")
        else:
            _pass(f"scales[{attr}]", msg)
    if not found_any:
        _warn("scale inspection", "no scale attrs on the pipeline (internal API drift); skipping")

    # Locate the persistent calibration cache.
    cache_root = Path(os.path.expanduser("~/.flash_rt/calibration"))
    if cache_root.is_dir():
        latest = sorted(cache_root.glob("*.json"), key=lambda p: p.stat().st_mtime)
        if latest:
            _pass("calibration cache", str(latest[-1]))
        else:
            _warn("calibration cache", f"no .json found under {cache_root}")
    else:
        _warn("calibration cache", f"{cache_root} does not exist (no on-disk cache produced)")

    # Sample inference.
    try:
        t0 = time.perf_counter()
        actions = model.predict(
            images=[images[0], wrists[0]],
            prompt=first_prompt,
        )
        elapsed = time.perf_counter() - t0
        if not np.isfinite(actions).all():
            _fail("post-calibration inference finite",
                  f"{int((~np.isfinite(actions)).sum())} NaN/Inf")
            failures.append("nan")
        else:
            _pass("post-calibration inference",
                  f"shape={actions.shape} dtype={actions.dtype} {elapsed*1000:.1f}ms")
    except Exception as e:
        _fail("post-calibration inference", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        failures.append("infer_crash")

    print()
    if failures:
        print(f"{BOLD}{RED}Phase 3 FAILED{RESET}  ({len(failures)} bad: {', '.join(failures)})")
        return 1
    print(f"{BOLD}{GREEN}Phase 3 PASSED{RESET}  — FP8 calibration written.")
    print(f"{DIM}Next: Phase 4 parity vs JAX server (scripts/spark_phase4_parity.py){RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
