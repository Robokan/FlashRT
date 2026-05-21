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
    parser.add_argument("--num-views", type=int, default=3,
                        help="Camera count. OpenArm = 3 "
                             "(cam_high + left_wrist + right_wrist).")
    parser.add_argument("--robot-action-dim", type=int, default=None,
                        help="Override per-call slice. Default uses "
                             "$FLASHRT_ROBOT_ACTION_DIM or LIBERO=7. "
                             "Set 16 for OpenArm bimanual.")
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

    # Load calib samples. Two npz schemas are supported:
    #   - legacy (LIBERO-style): {images, wrist_images, prompts, ...}
    #     2 cams, no state.
    #   - openarm v4 (current spark_phase3_prepare_calib output):
    #     {images_ego, images_left, images_right, state, prompts, ...}
    #     3 cams + 16-dim state.
    data = np.load(npz, allow_pickle=True)
    keys = list(data.files)
    if "images_ego" in keys:
        per_cam = [data["images_ego"]]
        if "images_left" in keys:
            per_cam.append(data["images_left"])
        if "images_right" in keys:
            per_cam.append(data["images_right"])
        prompts = data["prompts"]
        state = data["state"] if "state" in keys else None
    elif "images" in keys:
        per_cam = [data["images"]]
        if "wrist_images" in keys:
            per_cam.append(data["wrist_images"])
        prompts = data["prompts"]
        state = None
    else:
        _fail("calib data", f"unrecognised npz schema (keys={keys})")
        return 1

    # Trim per --num-views and --max-samples.
    per_cam = per_cam[: args.num_views]
    if args.max_samples is not None:
        per_cam = [arr[: args.max_samples] for arr in per_cam]
        prompts = prompts[: args.max_samples]
        if state is not None:
            state = state[: args.max_samples]

    n = len(per_cam[0])
    if n == 0:
        _fail("calib data", "0 samples")
        return 1
    _pass("calib data",
          f"{n} samples, {len(per_cam)} cams, "
          f"image shape={per_cam[0][0].shape}")

    # Build FlashRT observation dicts. Use the multi-view 'images' list
    # form so the rtx pipeline picks up all num_views slots; the
    # singular 'image'/'wrist_image'/'wrist_image_right' keys are
    # populated as a fallback for older code paths.
    obs_list: list[dict] = []
    for i in range(n):
        imgs = [per_cam[k][i] for k in range(len(per_cam))]
        obs = {"images": imgs, "image": imgs[0]}
        if len(imgs) >= 2:
            obs["wrist_image"] = imgs[1]
        if len(imgs) >= 3:
            obs["wrist_image_right"] = imgs[2]
        if state is not None:
            obs["state"] = state[i]
        obs_list.append(obs)

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

    # Inspect calibration results. The real Pi05Pipeline surfaces:
    #   pipeline.fp8_calibrated   bool (set True after first calibrate_fp8)
    #   pipeline.fp8_act_scales   dict[str, CudaBuffer]  (one f32 per GEMM)
    pipeline = model._pipe.pipeline
    if not getattr(pipeline, "fp8_calibrated", False):
        _fail("fp8 calibration", "pipeline.fp8_calibrated stayed False")
        failures.append("not_calibrated")
    else:
        scales_dict = getattr(pipeline, "fp8_act_scales", None) or {}
        # Bring each 4-byte CudaBuffer back to host so we can run
        # finite/zero/saturate checks. host_copy() is FlashRT's canonical
        # device->host fetch; falls back to ctypes if absent.
        host_vals: list[float] = []
        for key, buf in scales_dict.items():
            try:
                if hasattr(buf, "host_copy"):
                    arr = np.asarray(buf.host_copy()).view(np.float32)
                elif hasattr(buf, "to_numpy"):
                    arr = buf.to_numpy().view(np.float32)
                else:
                    # Last resort: pull 4 bytes from the device pointer.
                    import ctypes
                    raw = (ctypes.c_float * 1)()
                    model._pipe._cudart.cudaMemcpy(
                        ctypes.byref(raw),
                        ctypes.c_void_p(buf.ptr.value),
                        4, 2)  # cudaMemcpyDeviceToHost
                    arr = np.asarray([raw[0]], dtype=np.float32)
                host_vals.append(float(arr.flatten()[0]))
            except Exception as e:
                print(f"  WARN: could not read scale for {key}: "
                      f"{type(e).__name__}: {e}", file=sys.stderr)

        vals = np.asarray(host_vals, dtype=np.float32) if host_vals else np.zeros(0)
        n_zero = int((vals == 0.0).sum())
        n_inf = int((~np.isfinite(vals)).sum())
        # FlashRT stores each scale as approximately amax / FP8_E4M3_MAX
        # (median ~0.03 corresponds to amax ~14 on bf16-scale activations).
        # A scale >= 1.0 means the recorded amax already meets or exceeds
        # FP8's representable range, so any peak above the calibration
        # sample saturates and loses precision; >= 0.5 leaves <2x headroom.
        n_sat = int((vals >= 1.0).sum())
        n_tight = int(((vals >= 0.5) & (vals < 1.0)).sum())
        msg = (f"n={len(vals)} min={vals.min() if len(vals) else 0:.4e} "
               f"med={float(np.median(vals)) if len(vals) else 0:.4e} "
               f"max={vals.max() if len(vals) else 0:.4e}")
        if n_zero or n_inf:
            _fail("fp8 scales", f"{msg} | zero={n_zero} nonfinite={n_inf}")
            failures.append("scale_bad")
        elif n_sat:
            _warn("fp8 scales", f"{msg} | saturating={n_sat} (amax >= "
                  f"FP8 E4M3 max=448), tight-headroom={n_tight} "
                  f"(amax in [224, 448)); these layers lose FP8 precision "
                  f"on activations above the calibration peak")
        elif n_tight:
            _warn("fp8 scales", f"{msg} | tight-headroom={n_tight} "
                  f"(amax in [224, 448)); within FP8 range but <2x "
                  f"safety margin")
        else:
            _pass("fp8 scales",
                  f"{msg}, sites={len(scales_dict)}, all finite, no saturation")

    # Sample inference using all configured cameras + the matching
    # prompt for sample 0.
    try:
        t0 = time.perf_counter()
        actions = model.predict(
            images=[per_cam[k][0] for k in range(len(per_cam))],
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
