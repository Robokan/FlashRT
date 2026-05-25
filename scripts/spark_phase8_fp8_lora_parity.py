"""Phase 8 — FP8 + runtime-LoRA parity gate.

Validates the G12 fix (``flash_rt/models/pi05/pipeline_rtx.py``
decoder LoRA on FP8 paths) by comparing two **in-process** FlashRT
runs on the same checkpoint and the same observations:

    Run A  (reference, slow):  use_fp8=False  (pure BF16)
    Run B  (system under test): use_fp8=True   (FP8 enc + FP8 dec)

Both runs use the same runtime-LoRA mode (``FLASHRT_RUNTIME_LORA=all``
by default — set by ``--runtime-lora``). With deterministic noise
seeded per-sample the two runs should produce numerically near-identical
action chunks. Before G12, the FP8 run silently dropped the entire
decoder LoRA contribution and cosine collapsed to ~0.6 on
LoRA-finetuned checkpoints (see ``docs/spark_phase8_fp8_lora.md`` for
the root cause).

Usage
-----

Run twice on the Spark workstation (one model at a time to avoid GPU
RAM pressure):

::

    cd ~/sparkpack/FlashRT && source .venv/bin/activate

    # Pass 1 — BF16 reference. Saves actions to /tmp/fp8parity_bf16.npz
    PYTHONPATH=~/sparkpack/openpi/src:~/sparkpack/openpi/packages/openpi-client/src \\
    FLASHRT_ROBOT_ACTION_DIM=16 \\
    python scripts/spark_phase8_fp8_lora_parity.py \\
        --checkpoint ~/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-samples 20 \\
        --mode bf16 \\
        --output /tmp/fp8parity_bf16.npz

    # Pass 2 — FP8 SUT. Saves actions to /tmp/fp8parity_fp8.npz
    PYTHONPATH=~/sparkpack/openpi/src:~/sparkpack/openpi/packages/openpi-client/src \\
    FLASHRT_ROBOT_ACTION_DIM=16 \\
    python scripts/spark_phase8_fp8_lora_parity.py \\
        --checkpoint ~/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-samples 20 \\
        --mode fp8 \\
        --output /tmp/fp8parity_fp8.npz

    # Pass 3 — compare (no GPU needed)
    python scripts/spark_phase8_fp8_lora_parity.py \\
        --compare /tmp/fp8parity_bf16.npz /tmp/fp8parity_fp8.npz

Acceptance gates (the G12 fix should pass all three on the OpenArm v4
LoRA checkpoint):

    * per-sample cosine min      >= 0.99
    * per-sample L2 ratio        in [0.95, 1.05]
    * max abs joint-step diff    <= 0.10 rad on >= 90% of samples

Before G12 the cosine would sit around 0.55-0.65 and the ratio would
swing 0.4-1.6 because the FP8 decoder was running with LoRA-stripped
base weights.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

GATE_COSINE_MIN = 0.99
GATE_RATIO_LO = 0.95
GATE_RATIO_HI = 1.05
GATE_MAX_STEP_RAD = 0.10
GATE_PASS_FRACTION = 0.90


logger = logging.getLogger("phase8")


# ---------------------------------------------------------------------
# Math helpers (no GPU needed)
# ---------------------------------------------------------------------


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((af @ bf) / (na * nb))


def _l2_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """|a| / |b|."""
    na = float(np.linalg.norm(a.astype(np.float64)))
    nb = float(np.linalg.norm(b.astype(np.float64)))
    if nb == 0.0:
        return float("nan")
    return na / nb


def _chw(hwc: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.transpose(hwc.astype(np.uint8), (2, 0, 1)))


def _build_obs(data: dict, idx: int) -> dict:
    return {
        "state": np.asarray(data["state"][idx], dtype=np.float32),
        "images": {
            "cam_high":         _chw(data["images_ego"][idx]),
            "cam_left_wrist":   _chw(data["images_left"][idx]),
            "cam_right_wrist":  _chw(data["images_right"][idx]),
        },
        "prompt": str(data["prompts"][idx]),
    }


# ---------------------------------------------------------------------
# Inference run — one mode at a time
# ---------------------------------------------------------------------


def _run_one_mode(args: argparse.Namespace) -> int:
    if args.mode not in ("bf16", "fp8"):
        print(f"{RED}FAIL{RESET}  --mode must be bf16 or fp8")
        return 1
    if not args.calib_data or not args.calib_data.is_file():
        print(f"{RED}FAIL{RESET}  --calib-data {args.calib_data} missing")
        return 1
    if not args.output:
        print(f"{RED}FAIL{RESET}  --output required for inference run")
        return 1

    data = np.load(args.calib_data, allow_pickle=True)
    n_total = len(data["state"])
    if args.num_samples > n_total:
        print(f"{YELLOW}WARN{RESET}  --num-samples {args.num_samples} > "
              f"{n_total} available; using {n_total}")
        args.num_samples = n_total

    rng = np.random.default_rng(args.sample_seed)
    order = rng.permutation(n_total)[: args.num_samples].tolist()

    # Set runtime-LoRA mode BEFORE importing flash_rt.
    env_lora = os.environ.get("FLASHRT_RUNTIME_LORA")
    if env_lora is None:
        os.environ["FLASHRT_RUNTIME_LORA"] = args.runtime_lora
        print(f"{DIM}[env] FLASHRT_RUNTIME_LORA = {args.runtime_lora} "
              f"(from --runtime-lora){RESET}")
    else:
        print(f"{DIM}[env] FLASHRT_RUNTIME_LORA = {env_lora} (from env, "
              f"--runtime-lora={args.runtime_lora} ignored){RESET}")

    print(f"{BOLD}Loading FlashRT model: {args.mode} on {args.checkpoint}{RESET}")
    import torch  # noqa: F401  (force CUDA init before flash_rt)
    import flash_rt

    t0 = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework="jax",
        num_views=args.num_views,
        autotune=args.autotune,
        robot_action_dim=args.robot_action_dim,
        use_fp8=(args.mode == "fp8"),
        max_prompt_len=args.max_prompt_len,
        chunk_size=args.chunk_size,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    print(f"  loaded in {load_ms / 1000.0:.1f} s")

    # If FP8, run calibration first so the same calib set is used by
    # both runs. BF16 path is a no-op.
    if args.mode == "fp8":
        try:
            model.calibrate(data=str(args.calib_data),
                            num_samples=args.num_samples)
        except AttributeError:
            print(f"{YELLOW}WARN{RESET}  model.calibrate() not exposed by api; "
                  f"FP8 will lazy-calibrate on the first inference using a "
                  f"single observation (less stable).")

    # Force deterministic noise per-sample so the only difference
    # between BF16 and FP8 runs is the precision of the matmuls.
    actions: list[np.ndarray] = []
    rtc_chunks: list[np.ndarray] = []
    latencies: list[float] = []
    prompts: list[str] = []

    print(f"{BOLD}per-sample inference ({args.mode}, "
          f"runtime_lora={os.environ['FLASHRT_RUNTIME_LORA']}):{RESET}")
    for n, idx in enumerate(order):
        obs = _build_obs(data, int(idx))
        if args.prompt_override:
            obs["prompt"] = args.prompt_override
        seed = args.noise_seed + int(idx)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        t0 = time.perf_counter()
        try:
            res = model.predict(
                images=obs["images"],
                prompt=obs["prompt"],
                state=obs["state"],
            )
        except Exception as e:
            print(f"  [{n:3d}] {RED}crash:{RESET} {type(e).__name__}: {e}")
            return 1
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(res, dict):
            act = np.asarray(res["actions"], dtype=np.float32)
            rtc = res.get("_rtc_chunk_model_space")
            rtc = np.asarray(rtc, dtype=np.float32) if rtc is not None else None
        else:
            act = np.asarray(res, dtype=np.float32)
            rtc = None
        actions.append(act)
        if rtc is not None:
            rtc_chunks.append(rtc)
        latencies.append(latency_ms)
        prompts.append(obs["prompt"])
        print(f"  [{n:3d}] idx={int(idx):4d} seed={seed} latency={latency_ms:7.1f} ms "
              f"act_shape={act.shape}  |act|={np.linalg.norm(act):.4f}")

    actions_arr = np.stack(actions, axis=0)
    rtc_arr = np.stack(rtc_chunks, axis=0) if rtc_chunks else None
    out_dict = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "runtime_lora": os.environ["FLASHRT_RUNTIME_LORA"],
        "num_samples": args.num_samples,
        "sample_order": np.asarray(order, dtype=np.int64),
        "noise_seed_base": args.noise_seed,
        "actions": actions_arr,
        "latencies_ms": np.asarray(latencies, dtype=np.float32),
        "prompts": np.asarray(prompts, dtype=object),
    }
    if rtc_arr is not None:
        out_dict["rtc_chunk_model_space"] = rtc_arr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **out_dict)
    print(f"{GREEN}wrote{RESET}  {args.output}  "
          f"(actions {actions_arr.shape}, mean latency "
          f"{np.mean(latencies):.1f} ms)")
    return 0


# ---------------------------------------------------------------------
# Comparison — no GPU needed
# ---------------------------------------------------------------------


def _compare(args: argparse.Namespace) -> int:
    if len(args.compare) != 2:
        print(f"{RED}FAIL{RESET}  --compare needs exactly two paths "
              f"(reference, sut)")
        return 1
    ref_p, sut_p = args.compare
    ref = np.load(ref_p, allow_pickle=True)
    sut = np.load(sut_p, allow_pickle=True)

    def _modetag(d) -> str:
        return str(d["mode"]) if "mode" in d.files else "?"

    print(f"{BOLD}reference{RESET}  {ref_p}   mode={_modetag(ref)}  "
          f"lora={str(ref['runtime_lora']) if 'runtime_lora' in ref.files else '?'}")
    print(f"{BOLD}sut      {RESET}  {sut_p}   mode={_modetag(sut)}  "
          f"lora={str(sut['runtime_lora']) if 'runtime_lora' in sut.files else '?'}")

    if not np.array_equal(ref["sample_order"], sut["sample_order"]):
        print(f"{YELLOW}WARN{RESET}  sample orders differ — will compare "
              f"in saved order, which may pair unrelated samples.")
    n = min(len(ref["actions"]), len(sut["actions"]))
    if n == 0:
        print(f"{RED}FAIL{RESET}  no samples to compare")
        return 1

    cos_vals: list[float] = []
    ratio_vals: list[float] = []
    max_step_vals: list[float] = []
    print()
    print(f"{BOLD}per-sample parity:{RESET}")
    for i in range(n):
        a_ref = ref["actions"][i]
        a_sut = sut["actions"][i]
        steps = min(a_ref.shape[0], a_sut.shape[0])
        dims = min(a_ref.shape[1], a_sut.shape[1])
        r = a_ref[:steps, :dims]
        s = a_sut[:steps, :dims]
        cos = _cosine(r, s)
        ratio = _l2_ratio(s, r)
        max_step = float(np.max(np.abs(s - r)))
        cos_vals.append(cos)
        ratio_vals.append(ratio)
        max_step_vals.append(max_step)
        ok_cos = cos >= GATE_COSINE_MIN
        ok_rat = GATE_RATIO_LO <= ratio <= GATE_RATIO_HI
        ok_stp = max_step <= GATE_MAX_STEP_RAD
        verdict = (f"{GREEN}PASS{RESET}" if (ok_cos and ok_rat and ok_stp)
                   else f"{RED}FAIL{RESET}")
        print(f"  [{i:3d}] cos={cos:7.4f}  ratio={ratio:6.3f}  "
              f"max_step={max_step:6.3f} rad   {verdict}")

    cos_arr = np.asarray(cos_vals)
    ratio_arr = np.asarray(ratio_vals)
    step_arr = np.asarray(max_step_vals)

    n_pass_cos = int(np.sum(cos_arr >= GATE_COSINE_MIN))
    n_pass_rat = int(np.sum((ratio_arr >= GATE_RATIO_LO)
                            & (ratio_arr <= GATE_RATIO_HI)))
    n_pass_stp = int(np.sum(step_arr <= GATE_MAX_STEP_RAD))
    n_pass_all = int(np.sum(
        (cos_arr >= GATE_COSINE_MIN)
        & (ratio_arr >= GATE_RATIO_LO)
        & (ratio_arr <= GATE_RATIO_HI)
        & (step_arr <= GATE_MAX_STEP_RAD)))

    pct_pass = n_pass_all / n
    gate_pass = pct_pass >= GATE_PASS_FRACTION

    print()
    print(f"{BOLD}summary (n={n}):{RESET}")
    print(f"  cosine  min={cos_arr.min():.4f}  med={np.median(cos_arr):.4f}  "
          f"max={cos_arr.max():.4f}  pass={n_pass_cos}/{n}")
    print(f"  ratio   min={ratio_arr.min():.4f}  med={np.median(ratio_arr):.4f}  "
          f"max={ratio_arr.max():.4f}  pass={n_pass_rat}/{n}")
    print(f"  step    min={step_arr.min():.4f}  med={np.median(step_arr):.4f}  "
          f"max={step_arr.max():.4f} rad  pass={n_pass_stp}/{n}")
    print()
    print(f"  gate: cos>=0.99 AND ratio in [0.95, 1.05] AND max_step<=0.10")
    print(f"        {n_pass_all}/{n} samples pass ({pct_pass*100:.1f}%)  "
          f"(threshold {GATE_PASS_FRACTION*100:.0f}%)")
    if gate_pass:
        print(f"  {GREEN}{BOLD}GATE PASS{RESET}")
    else:
        print(f"  {RED}{BOLD}GATE FAIL{RESET}  — see docs/spark_phase8_fp8_lora.md")
    if args.compare_report:
        report = {
            "ref": str(ref_p), "sut": str(sut_p),
            "n": n,
            "cos": cos_arr.tolist(),
            "ratio": ratio_arr.tolist(),
            "max_step": step_arr.tolist(),
            "gate_pass": bool(gate_pass),
            "pct_pass": float(pct_pass),
        }
        args.compare_report.write_text(json.dumps(report, indent=2))
        print(f"  report → {args.compare_report}")
    return 0 if gate_pass else 1


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", choices=["bf16", "fp8"], default=None,
                   help="Inference mode for an inference run. Required "
                        "unless --compare is used.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Path to the Orbax JAX checkpoint directory.")
    p.add_argument("--calib-data", type=Path, default=None,
                   help="Phase 3 npz with stratified observations.")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--sample-seed", type=int, default=0,
                   help="RNG seed for sample-order permutation. Must be "
                        "the same for both bf16 and fp8 runs so the same "
                        "samples are compared.")
    p.add_argument("--noise-seed", type=int, default=42,
                   help="Base seed for per-sample noise. Each sample uses "
                        "noise_seed + idx so seeds are stable across "
                        "subsets / different --num-samples.")
    p.add_argument("--robot-action-dim", type=int, default=16)
    p.add_argument("--num-views", type=int, default=3)
    p.add_argument("--autotune", type=int, default=3)
    p.add_argument("--max-prompt-len", type=int, default=128)
    p.add_argument("--chunk-size", type=int, default=None)
    p.add_argument("--runtime-lora", default="all",
                   choices=["0", "all", "encoder", "encoder_ffn"],
                   help="FLASHRT_RUNTIME_LORA value. Default 'all' matches "
                        "the production server. Use '0' to compare with "
                        "merged-LoRA (which is the pre-G6 baseline).")
    p.add_argument("--prompt-override", default=None)
    p.add_argument("--output", type=Path, default=None,
                   help="Where to save the per-sample actions NPZ.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    p.add_argument("--compare", nargs=2, metavar=("BF16_NPZ", "FP8_NPZ"),
                   default=None,
                   help="Compare two NPZs from prior inference runs. "
                        "No GPU needed. Prints per-sample cos / ratio / "
                        "max-step and returns 0 if the gate passes.")
    p.add_argument("--compare-report", type=Path, default=None,
                   help="Optional JSON output path for the comparison.")

    args = p.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    if args.compare is not None:
        args.compare = [Path(x) for x in args.compare]
        return _compare(args)

    if args.mode is None:
        p.error("--mode required (bf16 or fp8) unless --compare is used")
    if args.checkpoint is None:
        p.error("--checkpoint required for inference runs")
    if args.calib_data is None:
        p.error("--calib-data required for inference runs")
    return _run_one_mode(args)


if __name__ == "__main__":
    sys.exit(main())
