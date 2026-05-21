"""Phase 4 — parity check, final-product topology.

This script hits two live websocket servers from the canonical openpi
client and compares the actions they return on identical observations:

    Server A  (reference, slow, ground truth)
        openpi JAX BF16 policy on port 8000
        served via openpi/scripts/serve_policy.py inside the
        openpi_server_ngc Docker container (NGC JAX 25.04 base)

    Server B  (system under test, fast, the future production server)
        FlashRT JAX FP8 policy on port 8002
        served via scripts/serve_policy_flashrt.py natively on Spark
        + the openpi WebsocketPolicyServer + the FlashRTPolicyAdapter

Both observations are formed the way openpi's own diag scripts do
(see openpi/scripts/diag_quant_parity.py and diag_live_server_parity.py):

    {
        "state":   (16,) float32,            # robot proprio
        "images":  {"cam_high": (3,224,224) u8,
                    "cam_left_wrist": ...,
                    "cam_right_wrist": ...},
        "prompt":  str,
    }

with images as **(C, H, W) uint8** (the OpenArmInputs transform on the
server side rearranges them to HWC). Observations are read from the
Phase 3 calibration npz so the scenes are realistic teleop frames, not
random pixels.

Architectural mismatches we already know about, accounted for in this
comparison:

  1. Action horizon: openpi pi05 trains and serves with
     action_horizon=50; FlashRT's pi05 pipeline hard-codes chunk_size=10
     for latency reasons. We compare on the overlapping first 10 steps.
  2. Unnormalization: openpi's serve_policy.py applies the OpenArm
     output transform (joint-radians). FlashRT's Pi05TorchFrontendRtx
     (which the JAX frontend inherits) calls unnormalize_actions in
     its infer path, so FlashRT actions should also be in joint-radian
     space. If the per-axis magnitudes look very different, we report
     that as a structural finding rather than a quantization gap.

Acceptance gate (matches the numbers from openpi/PYTORCH_PARITY_DEBUG.md
runtime-LoRA fp32 row, with looser ratio bounds for FP8 vs BF16):

  * post-unnorm cosine min  >= 0.99    (FP8 vs BF16 at 10-step diffusion)
  * post-unnorm ratio       in [0.95, 1.05]

If both servers are within those bounds on >= 90% of samples the gate
passes. Per-sample breakdown is printed to stdout and saved to a JSON
report.

Setup:

    # Shell 1 — openpi JAX reference server (Docker)
    cd ~/sparkpack/openpi
    docker compose -f scripts/docker/compose_ngc.yml run --rm -T \\
        --name openpi_jax_server openpi_serve \\
        python scripts/serve_policy.py policy:checkpoint \\
            --policy.config=pi05_openarm_ngc_lora_v4 \\
            --policy.dir=/app/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999

    # Shell 2 — FlashRT FP8 SUT server (native Spark venv)
    cd ~/sparkpack/FlashRT && source .venv/bin/activate
    PYTHONPATH=~/sparkpack/openpi/src:~/sparkpack/openpi/packages/openpi-client/src \\
    FLASHRT_ROBOT_ACTION_DIM=16 \\
    python scripts/serve_policy_flashrt.py \\
        --checkpoint ~/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999 \\
        --robot-action-dim 16 --num-views 3 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --default-prompt "put the chocolate bars in the container" \\
        --port 8002

    # Shell 3 — parity run
    PYTHONPATH=~/sparkpack/openpi/packages/openpi-client/src \\
    python scripts/spark_phase4_parity.py \\
        --reference-server localhost:8000 \\
        --flashrt-server   localhost:8002 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-samples 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

GATE_COSINE = 0.99
GATE_RATIO_LO = 0.95
GATE_RATIO_HI = 1.05
GATE_PASS_FRACTION = 0.90


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
    """Convert HWC uint8 -> CHW uint8 (openpi server input convention)."""
    return np.ascontiguousarray(np.transpose(hwc.astype(np.uint8), (2, 0, 1)))


def _build_obs(data: dict, idx: int) -> dict:
    """Build a single openpi-style observation dict from the Phase 3 npz row."""
    obs: dict = {
        "state": np.asarray(data["state"][idx], dtype=np.float32),
        "images": {
            "cam_high":         _chw(data["images_ego"][idx]),
            "cam_left_wrist":   _chw(data["images_left"][idx]),
            "cam_right_wrist":  _chw(data["images_right"][idx]),
        },
        "prompt": str(data["prompts"][idx]),
    }
    return obs


def _connect(host_port: str, default_port: int):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    host, _, port = host_port.partition(":")
    port_num = int(port) if port else default_port
    return WebsocketClientPolicy(host=host or "localhost", port=port_num), host, port_num


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-server", default="localhost:8000",
                        help="openpi JAX server (the slow ground truth)")
    parser.add_argument("--flashrt-server", default="localhost:8002",
                        help="FlashRT-served websocket policy (the SUT)")
    parser.add_argument("--calib-data", required=True, type=Path,
                        help="Phase 3 npz (we reuse its stratified obs so "
                             "both servers see realistic OpenArm teleop "
                             "scenes, not random pixels)")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Parity samples. 20 = ~10 sec each on JAX "
                             "BF16 first-call + 100 ms FP8 = ~3 min wall.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sample ordering RNG seed")
    parser.add_argument("--robot-action-dim", type=int, default=16,
                        help="OpenArm = 16. Slice both servers' outputs "
                             "to this many dims for comparison.")
    parser.add_argument("--prompt-override", default=None,
                        help="If set, send this prompt with every "
                             "observation, ignoring the per-sample prompt "
                             "in the npz. Without this, the FlashRT server "
                             "rebuilds its pipeline (and re-runs FP8 "
                             "calibration on the lazy path) every time the "
                             "tokenised prompt length changes, which "
                             "happens between calib samples with minor "
                             "casing/punctuation differences and inflates "
                             "FlashRT latency 10x for the affected calls. "
                             "Suggested for v4: 'put the chocolate bars "
                             "in the container'.")
    parser.add_argument("--output", default=None,
                        help="JSON report path (default phase4_parity_<ts>.json)")
    args = parser.parse_args()

    if not args.calib_data.is_file():
        print(f"{RED}FAIL{RESET}  --calib-data {args.calib_data} missing")
        return 1
    data = np.load(args.calib_data, allow_pickle=True)
    n_total = len(data["state"])
    if args.num_samples > n_total:
        print(f"{YELLOW}WARN{RESET}  --num-samples {args.num_samples} > "
              f"{n_total} available; using {n_total}")
        args.num_samples = n_total

    # Connect both servers.
    try:
        ref_client, ref_host, ref_port = _connect(args.reference_server, 8000)
        sut_client, sut_host, sut_port = _connect(args.flashrt_server, 8002)
    except Exception as e:
        print(f"{RED}FAIL{RESET}  websocket client import: "
              f"{type(e).__name__}: {e}")
        print(f"{DIM}Hint: PYTHONPATH=~/sparkpack/openpi/packages/openpi-client/src{RESET}")
        return 1
    try:
        ref_meta = ref_client.get_server_metadata()
        sut_meta = sut_client.get_server_metadata()
    except Exception as e:
        print(f"{RED}FAIL{RESET}  handshake: {type(e).__name__}: {e}")
        print(f"{DIM}Hint: is the openpi JAX server (port 8000) and the "
              f"FlashRT server (port 8002) actually running? See script "
              f"header for the launch commands.{RESET}")
        return 1
    print(f"{GREEN}reference{RESET}  ws://{ref_host}:{ref_port}  metadata={ref_meta}")
    print(f"{GREEN}flashrt  {RESET}  ws://{sut_host}:{sut_port}  metadata={sut_meta}")
    print()

    # Pick sample order. Random shuffle so any per-shape biases aren't
    # all concentrated at the start.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n_total)[: args.num_samples].tolist()

    records: list[dict] = []
    cos_vals: list[float] = []
    ratio_vals: list[float] = []
    raw_diff_norms: list[float] = []
    ref_first_call_dt = None
    sut_first_call_dt = None

    print(f"{BOLD}per-sample parity (first {args.robot_action_dim} dims, "
          f"first 10 chunk steps):{RESET}")
    for n, idx in enumerate(order):
        obs = _build_obs(data, int(idx))
        if args.prompt_override:
            obs["prompt"] = args.prompt_override

        # Reference server.
        try:
            t0 = time.perf_counter()
            ref_res = ref_client.infer(obs)
            ref_dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"  [{n:3d}] {RED}ref crash:{RESET} "
                  f"{type(e).__name__}: {e}")
            continue
        if ref_first_call_dt is None:
            ref_first_call_dt = ref_dt_ms

        # SUT.
        try:
            t0 = time.perf_counter()
            sut_res = sut_client.infer(obs)
            sut_dt_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as e:
            print(f"  [{n:3d}] {RED}flashrt crash:{RESET} "
                  f"{type(e).__name__}: {e}")
            continue
        if sut_first_call_dt is None:
            sut_first_call_dt = sut_dt_ms

        ref_act = np.asarray(ref_res["actions"], dtype=np.float64)
        sut_act = np.asarray(sut_res["actions"], dtype=np.float64)

        # Slice both to (min_chunk, robot_action_dim) for an
        # apples-to-apples comparison. min_chunk handles the
        # FlashRT-uses-10 vs openpi-uses-50 mismatch.
        steps = min(ref_act.shape[0], sut_act.shape[0])
        dims = min(ref_act.shape[1], sut_act.shape[1], args.robot_action_dim)
        ref_slice = ref_act[:steps, :dims]
        sut_slice = sut_act[:steps, :dims]

        cos = _cosine(ref_slice, sut_slice)
        ratio = _l2_ratio(sut_slice, ref_slice)  # |SUT| / |REF|
        diff_norm = float(np.linalg.norm(ref_slice - sut_slice))

        cos_vals.append(cos)
        ratio_vals.append(ratio)
        raw_diff_norms.append(diff_norm)

        cos_ok = cos >= GATE_COSINE
        ratio_ok = GATE_RATIO_LO <= ratio <= GATE_RATIO_HI
        color = GREEN if (cos_ok and ratio_ok) else (
            YELLOW if cos_ok or ratio_ok else RED)
        ref_first_row = ref_slice[0]
        sut_first_row = sut_slice[0]
        max_abs_diff = float(np.max(np.abs(ref_first_row - sut_first_row)))

        # On the first sample, also print the actual first-row vectors
        # side by side so structural mismatches (unnormalization,
        # axis order) jump out immediately.
        if n == 0:
            print(f"  [first-sample shape] ref={ref_act.shape} "
                  f"sut={sut_act.shape} -> compare slice {ref_slice.shape}")
            print(f"  [first-step ref]  {np.array2string(ref_first_row, precision=3, suppress_small=True)}")
            print(f"  [first-step sut]  {np.array2string(sut_first_row, precision=3, suppress_small=True)}")
            print(f"  [first-step diff] {np.array2string(ref_first_row - sut_first_row, precision=3, suppress_small=True)}")

        print(f"  [{n:3d}] idx={idx:3d}  cos={color}{cos:+.4f}{RESET}  "
              f"ratio={color}{ratio:.3f}{RESET}  "
              f"|diff|={diff_norm:.3f}  max|d|@t0={max_abs_diff:.3f}  "
              f"ref={ref_dt_ms:6.0f}ms  sut={sut_dt_ms:5.0f}ms  "
              f"'{obs['prompt'][:36]}'")

        records.append({
            "n": n, "sample_idx": int(idx),
            "prompt": obs["prompt"][:80],
            "ref_shape": list(ref_act.shape),
            "sut_shape": list(sut_act.shape),
            "compare_shape": list(ref_slice.shape),
            "cosine": cos, "ratio": ratio, "diff_norm": diff_norm,
            "max_abs_diff_first_step": max_abs_diff,
            "ref_dt_ms": ref_dt_ms, "sut_dt_ms": sut_dt_ms,
        })

    if not records:
        print(f"{RED}FAIL{RESET}  no successful parity samples")
        return 1

    cos_arr = np.asarray(cos_vals)
    ratio_arr = np.asarray(ratio_vals)
    cos_pass = (cos_arr >= GATE_COSINE)
    ratio_pass = (ratio_arr >= GATE_RATIO_LO) & (ratio_arr <= GATE_RATIO_HI)
    overall_pass_per_sample = cos_pass & ratio_pass
    pass_fraction = float(overall_pass_per_sample.mean())

    print()
    print(f"{BOLD}=== Phase 4 parity summary (n={len(records)}) ==={RESET}")
    print(f"  cosine: min={cos_arr.min():+.4f}  median={float(np.median(cos_arr)):+.4f}  "
          f"mean={float(cos_arr.mean()):+.4f}")
    print(f"  ratio : min={ratio_arr.min():.3f}   median={float(np.median(ratio_arr)):.3f}    "
          f"max={ratio_arr.max():.3f}")
    print(f"  per-sample PASS: {int(overall_pass_per_sample.sum())}/{len(records)} "
          f"({pass_fraction*100:.0f}%)  "
          f"(gate: cos>={GATE_COSINE}, ratio in [{GATE_RATIO_LO}, {GATE_RATIO_HI}])")
    print(f"  ref-server first-call latency: {ref_first_call_dt:.0f}ms "
          f"(JAX JIT)")
    print(f"  sut-server first-call latency: {sut_first_call_dt:.0f}ms "
          f"(FlashRT CUDA graph replay)")
    sut_steady = [r["sut_dt_ms"] for r in records[1:]] if len(records) > 1 else []
    if sut_steady:
        print(f"  sut-server steady p50: {float(np.median(sut_steady)):.0f}ms  "
              f"p99: {float(np.percentile(sut_steady, 99)):.0f}ms")

    gate_overall = pass_fraction >= GATE_PASS_FRACTION
    out_path = (Path(args.output) if args.output else
                Path(f"phase4_parity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    report = {
        "timestamp": datetime.now().isoformat(),
        "reference_server": args.reference_server,
        "flashrt_server": args.flashrt_server,
        "calib_data": str(args.calib_data),
        "ref_metadata": ref_meta,
        "sut_metadata": sut_meta,
        "robot_action_dim": args.robot_action_dim,
        "num_samples": len(records),
        "gate": {
            "cosine_min": GATE_COSINE,
            "ratio_bounds": [GATE_RATIO_LO, GATE_RATIO_HI],
            "pass_fraction": GATE_PASS_FRACTION,
            "achieved_pass_fraction": pass_fraction,
            "overall_pass": gate_overall,
        },
        "aggregate": {
            "cosine_min": float(cos_arr.min()),
            "cosine_median": float(np.median(cos_arr)),
            "cosine_mean": float(cos_arr.mean()),
            "ratio_min": float(ratio_arr.min()),
            "ratio_median": float(np.median(ratio_arr)),
            "ratio_max": float(ratio_arr.max()),
        },
        "records": records,
    }
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_path}")

    if gate_overall:
        print(f"\n{BOLD}{GREEN}Phase 4 PASSED{RESET}  — FlashRT FP8 matches "
              f"openpi JAX BF16 to within tolerance on "
              f"{int(pass_fraction*100)}% of samples.")
        return 0
    print(f"\n{BOLD}{RED}Phase 4 FAILED{RESET}  — "
          f"{int((1.0 - pass_fraction)*100)}% of samples outside the "
          f"tolerance gate.")
    print(f"{DIM}Likely culprits if cosine is low:")
    print(f"  - unnormalization mismatch (FlashRT JAX path skips it)")
    print(f"  - LoRA merge incomplete (Phase 2 reported merging only N pairs)")
    print(f"  - FP8 calibration set too narrow (encoder_ffn_down_w_16 saturating)")
    print(f"  - chunk_size=10 vs 50 attention-mask difference")
    print(f"  - robot_action_dim slicing mismatch (--robot-action-dim {args.robot_action_dim})"
          f"{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
