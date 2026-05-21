#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G2 gate: FP8 base + BF16 runtime-LoRA encoder, vs BF16 merge.

Measures whether the runtime-LoRA pattern that fixed openpi's PyTorch
parity also rescues FP8 + LoRA quality on FlashRT/Spark. Three modes,
each run in its own subprocess (cold CUDA context per mode):

  1. ``bf16_merge``     — BF16 throughout, LoRA merged in fp32 at
                          conversion time. This is the ground truth
                          we compare against.

  2. ``fp8_merge``      — FP8 base GEMMs + merged LoRA. Historically
                          collapses to cos ~0.617 on this checkpoint
                          (the layer-16 amax 27.4 outlier is the
                          symptom of LoRA-merge contaminating FP8
                          activation distributions).

  3. ``fp8_runtime_enc`` — FP8 base GEMMs + BF16 runtime LoRA across
                          encoder FFN gate/up/down + attn QKV/O.
                          Encoder ``fused`` FP8 path is downgraded to
                          the non-fused FP8 path automatically (see
                          _enc_lora_on in Pi05Pipeline) so the
                          intermediate BF16 activations needed by
                          ``_apply_enc_lora`` are present.

Gate (per the plan checked into AGENTS.md / spark_status.md):

  * cos(BF16 merge, FP8 runtime LoRA) >= 0.95 → great
  * cos                              >= 0.85 → commit
  * cos                              <  0.85 → halt and pivot

Usage::

    .venv/bin/python3 scripts/spark_runtime_lora_g2.py \\
        --checkpoint /home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_h10/29999 \\
        --prompt "pick up the chocolate bar"

Notes:

  * FP8 uses dynamic activation quantization (no pre-calibration).
    G2's job is to verify the runtime-LoRA wiring works under FP8 at
    all; recalibration with static scales is a follow-up if dynamic
    quant already crosses the 0.95 great gate.
  * Decoder LoRA stays merged-then-quantized in all three modes
    (only encoder is patched in this G2). The decoder LoRA
    contribution shows up as a fixed bias in fp8_runtime_enc that
    bf16_merge does not have, so a 100 % match isn't expected.
    Extension to decoder is queued for G2.5 if G2 passes.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _run_one_mode(
    mode: str,
    checkpoint: str,
    prompt: str,
    out_path: str,
    seed: int,
) -> int:
    """Spawn the inner worker for one (mode) run."""
    env = os.environ.copy()
    env.pop("FVK_PI05_RTX_FORCE_BF16", None)
    env.pop("FLASHRT_RUNTIME_LORA", None)
    env.pop("FLASHRT_LORA_SCALING", None)
    if mode == "bf16_merge":
        env["FVK_PI05_RTX_FORCE_BF16"] = "1"
    elif mode == "fp8_merge":
        pass  # default: FP8 + merged LoRA
    elif mode == "fp8_runtime_enc":
        env["FLASHRT_RUNTIME_LORA"] = "encoder"
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    env.setdefault("FLASHRT_ROBOT_ACTION_DIM", "16")
    env.setdefault("FLASHRT_PAD_STATE", "1")

    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--_worker_mode", mode,
        "--checkpoint", checkpoint,
        "--prompt", prompt,
        "--out", out_path,
        "--seed", str(seed),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    print(f"[orchestrator] mode={mode} returned {proc.returncode} "
          f"in {time.perf_counter() - t0:.1f}s")
    return proc.returncode


def _worker_main(args) -> int:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import flash_rt

    runtime_lora = os.environ.get("FLASHRT_RUNTIME_LORA", "")
    force_bf16 = os.environ.get("FVK_PI05_RTX_FORCE_BF16", "")
    print(f"[worker:{args._worker_mode}] "
          f"FVK_PI05_RTX_FORCE_BF16={force_bf16!r}, "
          f"FLASHRT_RUNTIME_LORA={runtime_lora!r}")

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="jax",
        num_views=3,
        autotune=1,
    )
    print(f"[worker:{args._worker_mode}] load_model: "
          f"{time.perf_counter() - t0:.1f}s")

    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    state = rng.standard_normal(16).astype(np.float32)

    # Warm-up: triggers graph capture + dynamic FP8 calibration.
    torch.manual_seed(args.seed)
    _ = model.predict(images=images, prompt=args.prompt, state=state)

    torch.manual_seed(args.seed)
    t0 = time.perf_counter()
    actions = model.predict(images=images, prompt=args.prompt, state=state)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    actions = np.asarray(actions)
    np.savez(
        args.out,
        actions=actions,
        infer_ms=infer_ms,
        mode=args._worker_mode,
        runtime_lora=runtime_lora,
        force_bf16=force_bf16,
    )
    print(f"[worker:{args._worker_mode}] saved actions shape={actions.shape} "
          f"dtype={actions.dtype} to {args.out} (infer {infer_ms:.1f}ms)")
    print(f"[worker:{args._worker_mode}] actions[0] = {actions[0].tolist()}")
    return 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _orchestrate(args) -> int:
    if not pathlib.Path(args.checkpoint).is_dir():
        print(f"FAIL: checkpoint {args.checkpoint} not a directory")
        return 1

    modes_to_run = ["bf16_merge", "fp8_merge", "fp8_runtime_enc"]

    with tempfile.TemporaryDirectory() as tdir:
        out_paths = {m: str(pathlib.Path(tdir) / f"{m}.npz")
                     for m in modes_to_run}
        for mode in modes_to_run:
            print(f"\n=== Running mode={mode} ===")
            rc = _run_one_mode(mode, args.checkpoint, args.prompt,
                               out_paths[mode], args.seed)
            if rc != 0:
                print(f"FAIL: worker for mode={mode} returned {rc}")
                return rc

        loaded = {m: np.load(out_paths[m]) for m in modes_to_run}
        actions = {m: loaded[m]["actions"] for m in modes_to_run}

        a_b = actions["bf16_merge"]
        a_fm = actions["fp8_merge"]
        a_fr = actions["fp8_runtime_enc"]

        if not (a_b.shape == a_fm.shape == a_fr.shape):
            print(f"FAIL: shape mismatch")
            return 1

        def _norm(x): return float(np.linalg.norm(x))

        norm_b = _norm(a_b)
        norm_fm = _norm(a_fm)
        norm_fr = _norm(a_fr)
        cos_b_fm = _cosine(a_b, a_fm)
        cos_b_fr = _cosine(a_b, a_fr)
        max_b_fm = float(np.abs(a_b - a_fm).max())
        max_b_fr = float(np.abs(a_b - a_fr).max())

        print("\n=== G2 comparison ===")
        print(f"  shape:                          {a_b.shape}")
        print(f"  norm bf16_merge:                {norm_b:.4f}  (ground truth)")
        print(f"  norm fp8_merge:                 {norm_fm:.4f}  "
              f"(ratio {norm_fm/norm_b:.3f})")
        print(f"  norm fp8_runtime_enc:           {norm_fr:.4f}  "
              f"(ratio {norm_fr/norm_b:.3f})")
        print(f"  cos(bf16_merge, fp8_merge):     {cos_b_fm:.6f}  "
              f"(catastrophe baseline)")
        print(f"  cos(bf16_merge, fp8_runtime):   {cos_b_fr:.6f}  "
              f"(candidate fix)")
        print(f"  max|bf16 - fp8_merge|:          {max_b_fm:.6f}")
        print(f"  max|bf16 - fp8_runtime|:        {max_b_fr:.6f}")
        cos_gap_recovered = (
            (cos_b_fr - cos_b_fm) / max(1.0 - cos_b_fm, 1e-6) * 100.0)
        print(f"  cos recovery vs fp8_merge:      {cos_gap_recovered:.2f}%")
        print(f"  infer bf16_merge:               "
              f"{float(loaded['bf16_merge']['infer_ms']):.1f} ms")
        print(f"  infer fp8_merge:                "
              f"{float(loaded['fp8_merge']['infer_ms']):.1f} ms")
        print(f"  infer fp8_runtime:              "
              f"{float(loaded['fp8_runtime_enc']['infer_ms']):.1f} ms")

        print()
        if cos_b_fr >= 0.95:
            print(f"  G2 GREAT  (cos {cos_b_fr:.4f} >= 0.95) — "
                  f"runtime LoRA fully rescues FP8 quality")
            return 0
        elif cos_b_fr >= 0.85:
            print(f"  G2 PASS   (cos {cos_b_fr:.4f} >= 0.85, < 0.95) — "
                  f"runtime LoRA partially rescues FP8. "
                  f"Next: extend to decoder, then re-calibrate FP8 "
                  f"with runtime LoRA on.")
            return 0
        else:
            print(f"  G2 FAIL   (cos {cos_b_fr:.4f} < 0.85)")
            print(f"    The cos 0.617 catastrophe baseline (fp8_merge) is "
                  f"{cos_b_fm:.4f} here.")
            if cos_b_fr > cos_b_fm:
                print(f"    Runtime LoRA helps ({cos_b_fr:.4f} > {cos_b_fm:.4f}) "
                      f"but not enough — check per-layer activation amax "
                      f"with calibration script, expect layer-16 amax to "
                      f"have dropped from 27.4.")
            else:
                print(f"    Runtime LoRA DOES NOT improve cos vs fp8_merge "
                      f"({cos_b_fr:.4f} <= {cos_b_fm:.4f}). Wiring bug; "
                      f"check that _apply_enc_lora is reached in the FP8 "
                      f"branches of _encoder_layer.")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="pick up the chocolate bar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--_worker_mode", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker_mode is not None:
        if args.out is None:
            print("worker requires --out", file=sys.stderr)
            return 2
        return _worker_main(args)
    return _orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
