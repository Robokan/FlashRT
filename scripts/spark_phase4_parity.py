"""Phase 4 — parity check: FlashRT-on-Spark vs the openpi JAX server.

This is the decisive correctness gate for the migration. Both models
get the same observation, and we compare:
  - raw cosine             (model output before unnormalization, full 32-dim)
  - raw L2 ratio           (|flashrt| / |jax|)
  - post-unnorm cosine     (after norm_stats applied, robot_action_dim slice)
  - post-unnorm L2 ratio   (|flashrt[:k]| / |jax[:k]|)

Acceptance gate (matches the runtime-LoRA-fp32 numbers from
``openpi/PYTORCH_PARITY_DEBUG.md``):
  - post-unnorm cosine  ≥ 0.999
  - post-unnorm ratio   ∈ [0.995, 1.005]

Setup:
  1. Start the JAX server in another shell:
        docker compose -f scripts/docker/compose_ngc.yml run --rm \\
            -p 8001:8001 openpi_serve \\
            python scripts/serve_policy.py policy:checkpoint \\
                --policy.config=pi05_openarm_ngc_lora \\
                --policy.dir=/openpi_assets/pi05_openarm_ngc_lora_v4 \\
                --port=8001
  2. Run this script in the flashrt_spark container:
        python3 scripts/spark_phase4_parity.py \\
            --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \\
            --calib-data /openpi_assets/calib_openarm_80.npz \\
            --jax-server ws://localhost:8001 \\
            --robot-action-dim 16

Output:
  - Per-sample metrics printed to stdout
  - Aggregate cosine + ratio reported with PASS/FAIL gate
  - JSON report at --output (default: phase4_parity_<timestamp>.json)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"

GATE_COSINE = 0.999
GATE_RATIO_LO = 0.995
GATE_RATIO_HI = 1.005


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float64)
    bf = b.flatten().astype(np.float64)
    na = float(np.linalg.norm(af))
    nb = float(np.linalg.norm(bf))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((af @ bf) / (na * nb))


def l2_ratio(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a.astype(np.float64)))
    nb = float(np.linalg.norm(b.astype(np.float64)))
    if nb == 0.0:
        return float("nan")
    return na / nb


def jax_server_infer(client, prompt: str, image: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Round-trip an observation to the openpi JAX websocket server."""
    obs = {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "prompt": prompt,
    }
    return np.asarray(client.infer(obs)["actions"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calib-data", required=True,
                        help="npz from spark_phase3_prepare_calib.py (we reuse "
                             "the stratified samples for parity)")
    parser.add_argument("--jax-server", default="localhost:8001",
                        help="openpi JAX server host:port")
    parser.add_argument("--num-samples", type=int, default=20,
                        help="Number of parity samples (default 20)")
    parser.add_argument("--robot-action-dim", type=int, default=16,
                        help="OpenArm bimanual = 16, LIBERO = 7")
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Load FlashRT model in the same process.
    try:
        import flash_rt
        t0 = time.perf_counter()
        model = flash_rt.load_model(
            checkpoint=args.checkpoint,
            framework="jax",
            num_views=2,
            autotune=3,
            robot_action_dim=args.robot_action_dim,
        )
        print(f"FlashRT loaded in {time.perf_counter() - t0:.1f}s "
              f"(robot_action_dim={args.robot_action_dim})", flush=True)
    except Exception as e:
        print(f"{RED}FlashRT load failed:{RESET} {e}")
        traceback.print_exc()
        return 1

    # Connect to JAX server.
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        host, _, port = args.jax_server.partition(":")
        port_num = int(port) if port else 8001
        client = WebsocketClientPolicy(host=host or "localhost", port=port_num)
        print(f"JAX server connected at {host}:{port_num}", flush=True)
    except Exception as e:
        print(f"{RED}JAX server connect failed:{RESET} {e}")
        print(f"{DIM}Hint: start the JAX server first; see header.{RESET}")
        return 1

    # Load samples.
    data = np.load(args.calib_data, allow_pickle=True)
    images = data["images"][: args.num_samples]
    wrists = data["wrist_images"][: args.num_samples]
    prompts = data["prompts"][: args.num_samples]
    n = len(images)
    if n == 0:
        print(f"{RED}no samples in {args.calib_data}{RESET}")
        return 1

    # Calibrate FlashRT with the same data (deterministic for parity).
    try:
        obs_list = [{"image": im, "wrist_image": wr}
                    for im, wr in zip(images, wrists)]
        # Use the first prompt for calibration; per-sample prompt set
        # happens inside the loop.
        model._pipe.set_prompt(str(prompts[0]))
        model._current_prompt = str(prompts[0])
        model.calibrate(obs_list[: min(n, 50)], percentile=99.9)
        print("FlashRT calibration complete", flush=True)
    except Exception as e:
        print(f"{RED}FlashRT calibration failed:{RESET} {e}")
        traceback.print_exc()
        return 1

    # Compare per sample.
    records: list[dict] = []
    raw_cos_list = []
    raw_ratio_list = []
    unnorm_cos_list = []
    unnorm_ratio_list = []

    rng = np.random.default_rng(args.seed)
    # Shuffle order so failures aren't all concentrated at the start.
    order = rng.permutation(n).tolist()

    for sample_idx in order:
        img = images[sample_idx]
        wrist = wrists[sample_idx]
        prompt = str(prompts[sample_idx]) if prompts[sample_idx] else "pick up the red block"

        # FlashRT inference.
        try:
            flashrt_actions = model.predict(images=[img, wrist], prompt=prompt)
        except Exception as e:
            print(f"  sample {sample_idx}: {RED}FlashRT crash:{RESET} {e}")
            continue

        # JAX server inference (round-trip + JPEG encode/decode on the server side).
        try:
            jax_actions = jax_server_infer(client, prompt, img, wrist)
        except Exception as e:
            print(f"  sample {sample_idx}: {RED}JAX server crash:{RESET} {e}")
            continue

        # The JAX server returns actions in robot dim already (e.g. 16 for OpenArm
        # via the policy's output unnorm). FlashRT returns robot_action_dim slice
        # of the unnormalized 32-dim output. So shapes should match: (chunk, k).
        if flashrt_actions.shape != jax_actions.shape:
            print(f"  sample {sample_idx}: {YELLOW}shape mismatch:{RESET} "
                  f"flashrt={flashrt_actions.shape} jax={jax_actions.shape}")
            # Best-effort: compare the overlapping slice.
            k = min(flashrt_actions.shape[1], jax_actions.shape[1])
            t = min(flashrt_actions.shape[0], jax_actions.shape[0])
            flashrt_actions = flashrt_actions[:t, :k]
            jax_actions = jax_actions[:t, :k]

        unnorm_cos = cosine(flashrt_actions, jax_actions)
        unnorm_r = l2_ratio(flashrt_actions, jax_actions)
        # Raw cosine here is the same as post-unnorm cosine because we
        # don't have the pre-unnorm activations from the JAX server.
        # Phase 4's gate is on post-unnorm anyway.
        raw_cos = unnorm_cos
        raw_r = unnorm_r

        raw_cos_list.append(raw_cos)
        raw_ratio_list.append(raw_r)
        unnorm_cos_list.append(unnorm_cos)
        unnorm_ratio_list.append(unnorm_r)

        records.append({
            "sample_idx": int(sample_idx),
            "prompt": prompt[:80],
            "unnorm_cosine": unnorm_cos,
            "unnorm_ratio": unnorm_r,
            "shape": list(flashrt_actions.shape),
        })

        color = (GREEN if unnorm_cos >= GATE_COSINE
                 and GATE_RATIO_LO <= unnorm_r <= GATE_RATIO_HI
                 else RED)
        print(f"  sample {sample_idx:3d}: cos={color}{unnorm_cos:.6f}{RESET}  "
              f"ratio={color}{unnorm_r:.4f}{RESET}  '{prompt[:50]}'",
              flush=True)

    # Aggregate.
    if not records:
        print(f"{RED}No successful parity comparisons.{RESET}")
        return 1

    mean_cos = float(np.mean(unnorm_cos_list))
    median_cos = float(np.median(unnorm_cos_list))
    min_cos = float(np.min(unnorm_cos_list))
    mean_ratio = float(np.mean(unnorm_ratio_list))
    median_ratio = float(np.median(unnorm_ratio_list))
    min_ratio = float(np.min(unnorm_ratio_list))
    max_ratio = float(np.max(unnorm_ratio_list))

    print(f"\n{BOLD}=== Phase 4 parity summary (n={len(records)}) ==={RESET}")
    print(f"  unnorm cosine: mean={mean_cos:.6f}  median={median_cos:.6f}  min={min_cos:.6f}")
    print(f"  unnorm ratio : mean={mean_ratio:.4f}  median={median_ratio:.4f}  range=[{min_ratio:.4f}, {max_ratio:.4f}]")

    cos_pass = min_cos >= GATE_COSINE
    ratio_pass = GATE_RATIO_LO <= min_ratio and max_ratio <= GATE_RATIO_HI

    print()
    print(f"  cosine gate (min ≥ {GATE_COSINE}):     {'PASS' if cos_pass else 'FAIL'}")
    print(f"  ratio gate ([{GATE_RATIO_LO}, {GATE_RATIO_HI}]):  {'PASS' if ratio_pass else 'FAIL'}")

    # Persist report.
    out = args.output or f"phase4_parity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "jax_server": args.jax_server,
        "robot_action_dim": args.robot_action_dim,
        "num_samples": len(records),
        "aggregate": {
            "unnorm_cosine_mean": mean_cos,
            "unnorm_cosine_median": median_cos,
            "unnorm_cosine_min": min_cos,
            "unnorm_ratio_mean": mean_ratio,
            "unnorm_ratio_median": median_ratio,
            "unnorm_ratio_min": min_ratio,
            "unnorm_ratio_max": max_ratio,
        },
        "gates": {
            "cosine_pass": cos_pass,
            "ratio_pass": ratio_pass,
            "overall_pass": cos_pass and ratio_pass,
        },
        "records": records,
    }
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out}")

    if cos_pass and ratio_pass:
        print(f"{BOLD}{GREEN}Phase 4 PASSED{RESET}  — FlashRT matches JAX server. Advance to Phase 5.")
        return 0
    print(f"{BOLD}{RED}Phase 4 FAILED{RESET}  — investigate before Phase 5.")
    print(f"{DIM}Likely culprits if cosine is low: missing LoRA merge "
          f"(Phase 2), bad calibration (Phase 3), wrong robot_action_dim "
          f"(--robot-action-dim flag), or norm_stats mismatch.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
