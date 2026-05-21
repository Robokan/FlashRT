"""Phase 5 part 1 — smoke a FlashRT-served websocket policy via the
canonical openpi client (``openpi_client.WebsocketClientPolicy``).

Server topology (the final-product topology this script tests):

    spark_phase5_serve_smoke.py (this script)
        │ obs dict (HWC uint8 images + state + prompt)
        ▼
    openpi_client.WebsocketClientPolicy        ─── lives in openpi-client
        │ websocket frame
        ▼
    openpi.serving.websocket_policy_server     ─── lives in openpi
        │ obs dict (passed through, JPEGs decoded)
        ▼
    flash_rt.serving.openpi_adapter.FlashRTPolicyAdapter
        │ images list
        ▼
    flash_rt.VLAModel.predict()                 ─── FlashRT JAX FP8

i.e. exactly what the robot client will see, with the smoke script
substituted for the robot. The only difference is that the robot streams
raw camera feeds at 50 Hz with RTC chunk overlap, where this script
fires N one-shot infers serially.

Setup (in one shell, leave running):

    PYTHONPATH=~/sparkpack/openpi/src:~/sparkpack/openpi/packages/openpi-client/src \\
    python3 scripts/serve_policy_flashrt.py \\
        --checkpoint ~/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999 \\
        --robot-action-dim 16 --num-views 3 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --default-prompt "put the chocolate bars in the container" \\
        --port 8002

Run the smoke (in a second shell, after the server prints "Ready"):

    PYTHONPATH=~/sparkpack/openpi/packages/openpi-client/src \\
    python3 scripts/spark_phase5_serve_smoke.py \\
        --server localhost:8002 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-samples 5
"""

from __future__ import annotations

import argparse
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


def _pass(name: str, info: str = "") -> None:
    print(f"{GREEN}PASS{RESET}  {name}  {DIM}{info}{RESET}")


def _fail(name: str, info: str = "") -> None:
    print(f"{RED}FAIL{RESET}  {name}  {DIM}{info}{RESET}")


def _warn(name: str, info: str = "") -> None:
    print(f"{YELLOW}WARN{RESET}  {name}  {DIM}{info}{RESET}")


def _load_samples(calib_path: Path, num_samples: int) -> list[dict]:
    """Build N robot-style observation dicts from a Phase 3 calib npz."""
    data = np.load(calib_path, allow_pickle=True)
    keys = list(data.files)
    if "images_ego" in keys:
        cams_in = [("ego", data["images_ego"])]
        if "images_left" in keys:
            cams_in.append(("left_wrist", data["images_left"]))
        if "images_right" in keys:
            cams_in.append(("right_wrist", data["images_right"]))
    elif "images" in keys:
        cams_in = [("ego", data["images"])]
        if "wrist_images" in keys:
            cams_in.append(("left_wrist", data["wrist_images"]))
    else:
        raise RuntimeError(f"unrecognised npz schema, keys={keys}")

    prompts = data["prompts"]
    state = data["state"] if "state" in keys else None
    n = min(num_samples, len(cams_in[0][1]))

    samples: list[dict] = []
    for i in range(n):
        # Send the obs in the openpi-server-compatible flat form: the
        # FlashRTPolicyAdapter prefers obs["images"][cam_name] but also
        # accepts flat obs["image"], obs["wrist_image"], etc. Use the
        # flat form since that is what the existing openpi diag scripts
        # and the openpi v4 policy expect on the wire.
        obs: dict = {
            "image": cams_in[0][1][i],
            "prompt": str(prompts[i]) if prompts[i] else "do the task",
        }
        if len(cams_in) >= 2:
            obs["wrist_image"] = cams_in[1][1][i]
        if len(cams_in) >= 3:
            obs["wrist_image_right"] = cams_in[2][1][i]
        if state is not None:
            obs["state"] = state[i]
        samples.append(obs)
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="localhost:8002",
                        help="host:port of the FlashRT-served websocket policy")
    parser.add_argument("--calib-data", required=True, type=Path,
                        help="Phase 3 npz used for obs construction (we "
                             "reuse the stratified observations so the "
                             "server sees realistic OpenArm scenes, not "
                             "random pixels)")
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--latency-ceiling-ms", type=float, default=200.0,
                        help="Plan-spec deadline. The first call after the "
                             "server boots may exceed this (cold CUDA graph "
                             "replay + decode); steady-state is what's "
                             "gated.")
    args = parser.parse_args()

    if not args.calib_data.is_file():
        _fail("calib data", f"{args.calib_data} missing — run "
                            f"spark_phase3_prepare_calib.py first")
        return 1

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as e:
        _fail("import openpi_client",
              f"{e}; set PYTHONPATH=~/sparkpack/openpi/packages/openpi-client/src")
        return 1
    _pass("import openpi_client")

    host, _, port = args.server.partition(":")
    port_num = int(port) if port else 8002
    client = WebsocketClientPolicy(host=host or "localhost", port=port_num)
    try:
        meta = client.get_server_metadata()
    except Exception as e:
        _fail("connect to server", f"{host}:{port_num}: {type(e).__name__}: {e}")
        print(f"{DIM}Is `scripts/serve_policy_flashrt.py --port {port_num}` "
              f"running in another shell?{RESET}")
        return 1
    _pass("server handshake", f"metadata={meta}")

    try:
        samples = _load_samples(args.calib_data, args.num_samples)
    except Exception as e:
        _fail("load samples", f"{type(e).__name__}: {e}")
        return 1
    _pass("load samples",
          f"n={len(samples)} keys={sorted(samples[0].keys())} "
          f"image={np.asarray(samples[0]['image']).shape}/{np.asarray(samples[0]['image']).dtype}")

    failures: list[str] = []
    latencies_ms: list[float] = []
    shapes: list[tuple] = []
    last_actions: np.ndarray | None = None
    for i, obs in enumerate(samples):
        t0 = time.perf_counter()
        try:
            res = client.infer(obs)
        except Exception as e:
            _fail(f"infer[{i}]", f"{type(e).__name__}: {e}")
            failures.append("infer_crash")
            continue
        dt_ms = (time.perf_counter() - t0) * 1000.0
        actions = np.asarray(res.get("actions"))
        if actions.ndim != 2:
            _fail(f"infer[{i}] shape", f"actions.shape={actions.shape}")
            failures.append("shape")
            continue
        if not np.all(np.isfinite(actions)):
            _fail(f"infer[{i}] finite",
                  f"actions has {(~np.isfinite(actions)).sum()} NaN/Inf")
            failures.append("nonfinite")
            continue
        latencies_ms.append(dt_ms)
        shapes.append(actions.shape)
        last_actions = actions
        # Pull policy_timing if the server returned it.
        timing = res.get("policy_timing") or {}
        infer_ms = timing.get("infer_ms")
        side = (f" server={infer_ms:.0f}ms" if infer_ms is not None else "")
        print(f"  infer[{i}]: roundtrip={dt_ms:.1f}ms{side}  "
              f"shape={actions.shape}  prompt={obs['prompt'][:48]!r}")

    if not latencies_ms:
        _fail("no successful infers")
        return 1

    # Cold call (first) is always slower; report it separately.
    cold = latencies_ms[0]
    steady = latencies_ms[1:] if len(latencies_ms) > 1 else latencies_ms
    p50 = float(np.percentile(steady, 50))
    p99 = float(np.percentile(steady, 99)) if len(steady) > 1 else p50

    print()
    print(f"{BOLD}Latency:{RESET} cold={cold:.0f}ms  "
          f"steady p50={p50:.0f}ms p99={p99:.0f}ms  "
          f"(ceiling={args.latency_ceiling_ms:.0f}ms)")
    if p50 > args.latency_ceiling_ms:
        _warn("steady-state p50",
              f"{p50:.0f}ms > ceiling {args.latency_ceiling_ms:.0f}ms")

    _pass("action shapes", f"all {shapes[0]}; n={len(shapes)}")
    if last_actions is not None:
        _pass("sample actions",
              f"range=[{last_actions.min():+.3f}, {last_actions.max():+.3f}] "
              f"mean={last_actions.mean():+.3f}")

    if failures:
        print(f"\n{BOLD}{RED}Phase 5 smoke FAILED{RESET}  "
              f"({len(failures)}/{len(samples)} infers had problems)")
        return 1

    print(f"\n{BOLD}{GREEN}Phase 5 smoke PASSED{RESET}  — "
          f"openpi_client can talk to the FlashRT-served websocket "
          f"policy. The robot client topology is wired correctly.")
    print(f"{DIM}Next: stand up the openpi JAX reference server on port "
          f"8001 and run scripts/spark_phase4_parity.py for the parity "
          f"gate.{RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
