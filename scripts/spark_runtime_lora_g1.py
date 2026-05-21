#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""G1 gate: BF16 runtime-LoRA arithmetic correctness.

Loads the OpenArm Pi0.5 LoRA checkpoint THREE times — fp32-merge,
``FLASHRT_LORA_SCALING=0`` (base only, LoRA disabled entirely), and
``FLASHRT_RUNTIME_LORA=encoder_ffn`` (extract encoder FFN LoRA + apply
as two bf16 matmuls per layer per GEMM) — runs a single inference frame
on fixed seeds in each, and reports both the raw cosines AND the
fraction of the merge-vs-no-LoRA gap that runtime LoRA closes.

Why this is the first thing to do per the Path A plan:
    The runtime-LoRA arithmetic ``out += (x @ la) @ lb`` should
    converge to the same result as the pre-merged
    ``out = x @ (W + la @ lb)`` in fp32, with small differences in
    bf16 from rounding order (this is exactly openpi's Bug #2 / fix:
    the runtime form is the JAX-equivalent one, the merged form is
    not — see PYTORCH_PARITY_DEBUG.md "★ 2026-05-19 RESOLVED").

What G1 actually checks:
    Because runtime LoRA and merge are intentionally NOT bit-equivalent
    (that's the whole point — merge is the wrong one), a naive
    cos(merge, runtime) > 0.999 check is unattainable by construction.
    The right check is whether the runtime LoRA is doing meaningful
    work in the right direction:

      * runtime LoRA must produce action_norm > merge action_norm
        (recovers the magnitude that the merge form loses; matches
        openpi's PT-merge ratio 0.918 vs JAX ratio 1.0)
      * runtime LoRA must close most of the gap between merge and
        "no LoRA at all" — quantifies that the LoRA tensors are
        actually wired and applied through the right activations
      * no NaN/Inf

Both modes run with ``FVK_PI05_RTX_FORCE_BF16=1`` so FP8 noise can't
contaminate the G1 signal.

Usage::

    .venv/bin/python3 scripts/spark_runtime_lora_g1.py \\
        --checkpoint /home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05_h10/29999 \\
        --prompt "pick up the chocolate bar"

The orchestrator runs two subprocesses (one per mode) to keep CUDA
state isolated, saves each mode's actions to /tmp, then compares.
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
from typing import Optional

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
    # Force BF16 throughout so FP8 noise can't contaminate the G1 signal.
    env["FVK_PI05_RTX_FORCE_BF16"] = "1"
    env.pop("FLASHRT_RUNTIME_LORA", None)
    env.pop("FLASHRT_LORA_SCALING", None)
    if mode == "merge":
        pass  # default behaviour: merge all LoRA in fp32
    elif mode == "no_lora":
        # Same as merge but force-zero the LoRA contribution. Equivalent
        # to "what would the base Gemma do without any fine-tuning".
        # Used as the "LoRA-missing" baseline for the recovery metric.
        env["FLASHRT_LORA_SCALING"] = "0.0"
    elif mode in ("runtime_lora_encoder_ffn", "runtime_lora"):
        env["FLASHRT_RUNTIME_LORA"] = "encoder_ffn"
    elif mode == "runtime_lora_encoder":
        # Encoder FFN + encoder attention (q/kv/o). Closes the residual
        # ~5 % cosine gap that encoder-FFN-only leaves.
        env["FLASHRT_RUNTIME_LORA"] = "encoder"
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    # OpenArm is bimanual 16-DOF; the JAX frontend reads this env var.
    env.setdefault("FLASHRT_ROBOT_ACTION_DIM", "16")
    # Pad-state on so prompt length is fixed (no rebuild jitter to confuse
    # the cosine measurement).
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
    """Inner worker: load one mode, run one frame, save actions."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import flash_rt
    from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx

    runtime_lora = os.environ.get("FLASHRT_RUNTIME_LORA", "")
    print(f"[worker:{args._worker_mode}] "
          f"FLASHRT_RUNTIME_LORA={runtime_lora!r}, "
          f"FVK_PI05_RTX_FORCE_BF16={os.environ.get('FVK_PI05_RTX_FORCE_BF16')!r}, "
          f"FLASHRT_ROBOT_ACTION_DIM={os.environ.get('FLASHRT_ROBOT_ACTION_DIM')!r}")

    t0 = time.perf_counter()
    # 3 cams to match openarm v4 setup. autotune=1 keeps load fast.
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="jax",
        num_views=3,
        autotune=1,
    )
    load_s = time.perf_counter() - t0
    print(f"[worker:{args._worker_mode}] load_model: {load_s:.1f}s")

    # Synthetic but deterministic 3-cam observation. Same seed = same images
    # so observation parity is held constant across the two modes.
    rng = np.random.default_rng(0)
    images = [
        rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    state = rng.standard_normal(16).astype(np.float32)

    # Warm-up: first call triggers FP8 calibration / graph capture; we don't
    # want that work mixed into the cosine measurement.
    torch.manual_seed(args.seed)
    _ = model.predict(images=images, prompt=args.prompt, state=state)

    # Measured call: fresh seed so diffusion noise is deterministic.
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

    modes_to_run = ["merge", "no_lora", "runtime_lora_encoder_ffn"]
    if args.encoder:
        modes_to_run.append("runtime_lora_encoder")

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
        a_m = actions["merge"]
        a_n = actions["no_lora"]
        a_r = actions["runtime_lora_encoder_ffn"]
        if not (a_m.shape == a_n.shape == a_r.shape):
            print(f"FAIL: shape mismatch: merge={a_m.shape} "
                  f"no_lora={a_n.shape} rl={a_r.shape}")
            return 1

        def _norm(x): return float(np.linalg.norm(x))
        def _ratio(x, base): return _norm(x) / _norm(base) if _norm(base) > 0 else float("nan")

        cos_mr = _cosine(a_m, a_r)
        cos_mn = _cosine(a_m, a_n)
        norm_m, norm_n, norm_r = _norm(a_m), _norm(a_n), _norm(a_r)
        ratio_r = norm_r / norm_m if norm_m > 0 else float("nan")
        ratio_n = norm_n / norm_m if norm_m > 0 else float("nan")
        max_abs_diff_r = float(np.abs(a_m - a_r).max())
        per_joint_max = np.abs(a_m - a_r).max(axis=0)

        gap_n = 1.0 - cos_mn  # cos gap when LoRA is missing entirely
        gap_r = 1.0 - cos_mr  # cos gap with runtime LoRA on
        recovery_pct = (
            (1.0 - gap_r / gap_n) * 100.0 if gap_n > 1e-6 else float("nan"))

        m, n, r = loaded["merge"], loaded["no_lora"], loaded["runtime_lora_encoder_ffn"]
        print("\n=== G1 comparison ===")
        print(f"  shape:                          {a_m.shape}")
        print(f"  action norm (merge):            {norm_m:.4f}  (ground truth)")
        print(f"  action norm (no_lora):          {norm_n:.4f}  (ratio {ratio_n:.3f})")
        print(f"  action norm (runtime ffn):      {norm_r:.4f}  (ratio {ratio_r:.3f})")
        print(f"  cos(merge, no_lora):            {cos_mn:.6f}  "
              f"(gap {gap_n:.6f})")
        print(f"  cos(merge, runtime ffn):        {cos_mr:.6f}  "
              f"(gap {gap_r:.6f})")
        print(f"  recovery (encoder FFN only):    {recovery_pct:.2f}% of LoRA gap")
        if "runtime_lora_encoder" in actions:
            a_re = actions["runtime_lora_encoder"]
            norm_re = _norm(a_re)
            ratio_re = norm_re / norm_m if norm_m > 0 else float("nan")
            cos_mre = _cosine(a_m, a_re)
            gap_re = 1.0 - cos_mre
            recovery_re = (1.0 - gap_re / gap_n) * 100.0 if gap_n > 1e-6 else float("nan")
            max_abs_diff_re = float(np.abs(a_m - a_re).max())
            print(f"  action norm (runtime encoder):  {norm_re:.4f}  (ratio {ratio_re:.3f})")
            print(f"  cos(merge, runtime encoder):    {cos_mre:.6f}  "
                  f"(gap {gap_re:.6f})")
            print(f"  recovery (encoder full):        {recovery_re:.2f}% of LoRA gap")
            print(f"  max|merge - runtime encoder|:   {max_abs_diff_re:.6f}")
            print(f"  infer runtime encoder:          "
                  f"{float(loaded['runtime_lora_encoder']['infer_ms']):.1f} ms")
        print(f"  max|merge - runtime ffn|:       {max_abs_diff_r:.6f}")
        print(f"  infer merge:                    {float(m['infer_ms']):.1f} ms")
        print(f"  infer no_lora:                  {float(n['infer_ms']):.1f} ms")
        print(f"  infer runtime ffn:              {float(r['infer_ms']):.1f} ms")
        print(f"  per-joint max |merge - runtime|:")
        for j, d in enumerate(per_joint_max):
            marker = "  *" if d > 0.05 else ""
            print(f"    joint {j:2d}: {d:.4f}{marker}")

        # G1 gate: runtime LoRA must (a) recover most of the LoRA
        # contribution gap and (b) increase action magnitude vs merge
        # (matches openpi's runtime-LoRA-recovers-magnitude finding).
        # Threshold of 90% gap recovery is generous because we only
        # patch encoder FFN (54/252 modules); full coverage extends to
        # encoder/decoder attn + decoder FFN.
        ok_recovery = recovery_pct >= 90.0
        ok_direction = norm_r > norm_m
        ok_finite = bool(np.isfinite(a_r).all())

        print()
        if ok_recovery and ok_direction and ok_finite:
            print(f"  G1 PASS  (recovered {recovery_pct:.1f}% of LoRA gap, "
                  f"runtime norm {ratio_r:.3f}× merge — matches openpi's "
                  f"runtime-recovers-magnitude finding)")
            print(f"  Next: G2 — wire runtime LoRA into the FP8 path "
                  f"and re-calibrate; measure cos(BF16 merge, FP8 runtime LoRA).")
            return 0
        else:
            print(f"  G1 FAIL")
            if not ok_recovery:
                print(f"    recovery {recovery_pct:.1f}% < 90% — wiring "
                      f"is not picking up the full LoRA contribution")
            if not ok_direction:
                print(f"    runtime norm {norm_r:.3f} <= merge norm "
                      f"{norm_m:.3f} — runtime LoRA is not recovering "
                      f"magnitude; check sign/scale of the delta")
            if not ok_finite:
                print(f"    NaN/Inf in runtime_lora actions")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="pick up the chocolate bar")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--encoder", action="store_true",
        help="Also run FLASHRT_RUNTIME_LORA=encoder mode (full encoder "
             "coverage: FFN + attention q/kv/o). Adds one model load.")
    # Internal worker arguments (not for direct CLI use).
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
