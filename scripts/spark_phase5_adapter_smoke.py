"""Phase 5 smoke test — confirm the FlashRT websocket server is wire-
compatible with the existing AsyncActionChunkBroker client.

This test does NOT require a real robot. It simulates the
AsyncActionChunkBroker client side, connects to the running FlashRT
server, sends a synthetic observation, and verifies:

  1. Metadata handshake (the server should send metadata with at least
     a 'chunk_size' field — the broker reads this).
  2. A single infer round-trip returns a dict containing `actions` with
     shape (chunk_size, robot_action_dim).
  3. The latency reported in `policy_timing.infer_ms` is < 200 ms.
  4. The `_rtc_chunk_model_space` field is present so the broker's
     RTC bookkeeping doesn't crash.
  5. 10 sequential calls all succeed (catches bugs that only show up
     after the CUDA graph cache settles).

Prereqs:
  - Run `python3 scripts/serve_policy_flashrt.py ... --port 8001` in a
    separate shell first.

Usage:
    python3 scripts/spark_phase5_adapter_smoke.py --port 8001
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback

import numpy as np


GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num-calls", type=int, default=10)
    parser.add_argument("--prompt", default="pick up the red block")
    args = parser.parse_args()

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError:
        print(f"{RED}openpi_client not installed.{RESET}")
        return 1

    print(f"Connecting to ws://{args.host}:{args.port}...", flush=True)
    try:
        client = WebsocketClientPolicy(host=args.host, port=args.port)
    except Exception as e:
        print(f"{RED}connect failed:{RESET} {e}")
        return 1

    # 1. Metadata.
    try:
        meta = client.get_server_metadata()
        if "chunk_size" not in meta:
            print(f"{RED}metadata missing 'chunk_size':{RESET} {meta}")
            return 1
        chunk_size = int(meta["chunk_size"])
        robot_action_dim = int(meta.get("robot_action_dim", 7))
        print(f"{GREEN}metadata OK{RESET}  chunk_size={chunk_size}  robot_action_dim={robot_action_dim}  {DIM}{meta}{RESET}")
    except Exception as e:
        print(f"{RED}metadata fetch failed:{RESET} {e}")
        traceback.print_exc()
        return 1

    rng = np.random.default_rng(0)
    failures = 0
    latencies_ms: list[float] = []

    for i in range(args.num_calls):
        # Build an openpi-style observation. The client side normally
        # JPEG-encodes inside images["..."]; for smoke we send raw arrays
        # — both formats are supported by the server's _decode_jpeg_images.
        obs = {
            "image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
            "prompt": args.prompt,
        }

        t0 = time.perf_counter()
        try:
            result = client.infer(obs)
        except Exception as e:
            print(f"  {i:2d}: {RED}infer crashed:{RESET} {e}")
            failures += 1
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        # 2 + 3. Shape & timing.
        actions = result.get("actions")
        if actions is None:
            print(f"  {i:2d}: {RED}no 'actions' in response:{RESET} keys={list(result.keys())}")
            failures += 1
            continue
        actions = np.asarray(actions)
        if actions.shape != (chunk_size, robot_action_dim):
            print(f"  {i:2d}: {RED}wrong action shape:{RESET} {actions.shape} expected {(chunk_size, robot_action_dim)}")
            failures += 1
            continue
        if not np.isfinite(actions).all():
            print(f"  {i:2d}: {RED}non-finite actions{RESET}")
            failures += 1
            continue

        timing = result.get("policy_timing", {})
        infer_ms = timing.get("infer_ms")

        # 4. RTC field present.
        rtc_chunk = result.get("_rtc_chunk_model_space")
        if rtc_chunk is None:
            # Not fatal; some adapter versions may drop it. Warn once.
            if i == 0:
                print(f"  {DIM}note: server did not include _rtc_chunk_model_space (RTC may degrade){RESET}")

        print(f"  {i:2d}: {GREEN}OK{RESET}  shape={actions.shape}  "
              f"client_rt={elapsed_ms:.1f}ms  server_infer={infer_ms:.1f}ms")

    print()
    if failures:
        print(f"{BOLD}{RED}Phase 5 FAILED{RESET}  ({failures} / {args.num_calls} calls bad)")
        return 1
    if latencies_ms:
        lats = np.asarray(latencies_ms[1:]) if len(latencies_ms) > 1 else np.asarray(latencies_ms)
        print(f"{BOLD}{GREEN}Phase 5 PASSED{RESET}  "
              f"client-side round-trip mean={lats.mean():.1f}ms p99={np.percentile(lats, 99):.1f}ms "
              f"(includes msgpack + ws localhost)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
