"""Phase 7 — measure latency components on DGX Spark so the colocation
decision is data-driven instead of intuition-driven.

Components measured:
  1. JPEG encode (on the client) — what we'd save by going to raw arrays.
  2. JPEG decode (on the server) — what we'd save by going to raw arrays.
  3. msgpack pack/unpack — how much serialisation costs.
  4. Websocket round-trip on localhost (with a no-op server).
  5. FlashRT inference (steady-state, post-calibration).
  6. End-to-end client round-trip via the real FlashRT server.

These run independently of the robot; everything is synthetic data.

Decision rules (from the plan):
  - If JPEG decode > 10% of inference → recommend raw-array transport
    (one-file change in websocket_policy_server.py).
  - If JPEG-fixed loop latency budget is still tight → consider
    colocation.
  - Otherwise → keep the websocket boundary, no architectural change.

Usage (Spark host with FlashRT server already running):
    python3 scripts/spark_phase7_latency_breakdown.py \\
        --port 8001 \\
        --num-iter 50
"""

from __future__ import annotations

import argparse
import io
import socket
import sys
import time
from typing import Callable

import numpy as np


GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def time_op(fn: Callable[[], object], n: int = 50, warmup: int = 5) -> dict:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    arr = np.asarray(ts)
    return {
        "n": n,
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001,
                        help="FlashRT server port (must be running). 0 = skip the e2e bench.")
    parser.add_argument("--num-iter", type=int, default=50)
    parser.add_argument("--image-h", type=int, default=224)
    parser.add_argument("--image-w", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(args.image_h, args.image_w, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(args.image_h, args.image_w, 3), dtype=np.uint8)

    results: dict[str, dict] = {}

    # 1+2. JPEG encode + decode.
    try:
        import cv2
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality]

        # Pre-encode once for the decode test.
        ok, buf = cv2.imencode(".jpg", img, encode_params)
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        jpeg_bytes = buf.tobytes()

        results["jpeg_encode (one image)"] = time_op(
            lambda: cv2.imencode(".jpg", img, encode_params)[1], n=args.num_iter)
        results["jpeg_decode (one image)"] = time_op(
            lambda: cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR),
            n=args.num_iter)

        # The server-side _decode_jpeg_images does decode + BGR→RGB + transpose
        # (CHW). Bench the whole pipeline so the number reflects what the
        # server actually pays.
        def server_decode():
            arr = np.frombuffer(jpeg_bytes, np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return np.transpose(rgb, (2, 0, 1))

        results["server _decode_jpeg_images (1 cam)"] = time_op(server_decode, n=args.num_iter)
    except ImportError:
        print(f"{YELLOW}cv2 not available — skipping JPEG measurements{RESET}")

    # 3. msgpack pack/unpack.
    try:
        from openpi_client import msgpack_numpy
        packer = msgpack_numpy.Packer()
        obs_raw = {"image": img, "wrist_image": wrist, "prompt": "pick up the red block"}
        packed = packer.pack(obs_raw)

        results["msgpack pack (raw arrays)"] = time_op(
            lambda: packer.pack(obs_raw), n=args.num_iter)
        results["msgpack unpack (raw arrays)"] = time_op(
            lambda: msgpack_numpy.unpackb(packed), n=args.num_iter)

        # Same but with JPEG-encoded images (closer to the real transport).
        if "cv2" in sys.modules:
            import cv2
            _, jb = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            _, wb = cv2.imencode(".jpg", wrist, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            obs_jpeg = {
                "images": {"cam_high": jb.tobytes(), "cam_left_wrist": wb.tobytes()},
                "prompt": "pick up the red block",
            }
            packed_jpeg = packer.pack(obs_jpeg)
            results["msgpack pack (JPEG-encoded)"] = time_op(
                lambda: packer.pack(obs_jpeg), n=args.num_iter)
            results["msgpack unpack (JPEG-encoded)"] = time_op(
                lambda: msgpack_numpy.unpackb(packed_jpeg), n=args.num_iter)
            print(f"{DIM}Raw obs:   {len(packed)/1024:.1f} KB{RESET}")
            print(f"{DIM}JPEG obs:  {len(packed_jpeg)/1024:.1f} KB{RESET}")
    except ImportError:
        print(f"{YELLOW}openpi_client not available — skipping msgpack{RESET}")

    # 4. TCP loopback round-trip (lower bound on websocket RTT).
    try:
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.bind(("127.0.0.1", 0))
        port = server_sock.getsockname()[1]
        server_sock.listen(1)

        client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_sock.connect(("127.0.0.1", port))
        accepted, _ = server_sock.accept()

        payload = packed if "msgpack pack (raw arrays)" in results else b"x" * 1024

        def loopback():
            client_sock.sendall(payload)
            n = 0
            while n < len(payload):
                buf = accepted.recv(len(payload) - n)
                if not buf:
                    break
                n += len(buf)
                accepted.sendall(buf)
            n2 = 0
            while n2 < len(payload):
                buf = client_sock.recv(len(payload) - n2)
                if not buf:
                    break
                n2 += len(buf)

        results[f"TCP loopback echo ({len(payload)//1024} KB)"] = time_op(loopback, n=args.num_iter)
        client_sock.close()
        accepted.close()
        server_sock.close()
    except Exception as e:
        print(f"{YELLOW}TCP loopback skipped: {e}{RESET}")

    # 5+6. Full client round-trip via the actual FlashRT server.
    if args.port > 0:
        try:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client = WebsocketClientPolicy(host=args.host, port=args.port)
            obs_e2e = {
                "image": img,
                "wrist_image": wrist,
                "prompt": "pick up the red block",
            }
            # Warm up CUDA graph cache.
            for _ in range(3):
                client.infer(obs_e2e)

            client_ts = []
            server_ts = []
            for _ in range(args.num_iter):
                t0 = time.perf_counter()
                res = client.infer(obs_e2e)
                client_ts.append((time.perf_counter() - t0) * 1000.0)
                if "policy_timing" in res and "infer_ms" in res["policy_timing"]:
                    server_ts.append(float(res["policy_timing"]["infer_ms"]))

            arr = np.asarray(client_ts)
            results["e2e client round-trip"] = {
                "n": len(arr),
                "mean_ms": float(arr.mean()),
                "p50_ms": float(np.percentile(arr, 50)),
                "p99_ms": float(np.percentile(arr, 99)),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
            }
            if server_ts:
                arr_s = np.asarray(server_ts)
                results["e2e server infer (FlashRT predict)"] = {
                    "n": len(arr_s),
                    "mean_ms": float(arr_s.mean()),
                    "p50_ms": float(np.percentile(arr_s, 50)),
                    "p99_ms": float(np.percentile(arr_s, 99)),
                    "min_ms": float(arr_s.min()),
                    "max_ms": float(arr_s.max()),
                }
        except Exception as e:
            print(f"{YELLOW}e2e bench skipped:{RESET} {e}")

    # Print table.
    print()
    print(f"{BOLD}{'component':<42s}  {'mean':>8s}  {'p50':>8s}  {'p99':>8s}  {'min':>8s}  {'max':>8s}{RESET}")
    print("-" * 92)
    for name, r in results.items():
        print(f"  {name:<40s}  {r['mean_ms']:>6.2f}ms  {r['p50_ms']:>6.2f}ms  "
              f"{r['p99_ms']:>6.2f}ms  {r['min_ms']:>6.2f}ms  {r['max_ms']:>6.2f}ms")

    # Decision tree.
    print()
    print(f"{BOLD}=== Phase 7 colocation decision ==={RESET}")
    server_infer = results.get("e2e server infer (FlashRT predict)", {}).get("mean_ms")
    jpeg_decode = results.get("server _decode_jpeg_images (1 cam)", {}).get("mean_ms")
    e2e = results.get("e2e client round-trip", {}).get("mean_ms")

    if jpeg_decode is not None and server_infer is not None:
        # Two cameras' worth of decode.
        decode_pair = jpeg_decode * 2
        decode_fraction = decode_pair / server_infer
        print(f"  JPEG decode (2 cams):  {decode_pair:.2f} ms")
        print(f"  Server infer:          {server_infer:.2f} ms")
        print(f"  Decode/infer ratio:    {decode_fraction:.2%}")
        if decode_fraction > 0.10:
            print(f"  → {YELLOW}Recommend{RESET}: switch transport to raw arrays.")
            print(f"     Concrete: openpi_client send raw uint8 arrays (skip cv2.imencode); "
                  f"server passes through obs['image'] unchanged. ~30 minute change.")
        else:
            print(f"  → {GREEN}No action needed.{RESET} JPEG decode is < 10% of inference.")
    if e2e is not None and server_infer is not None:
        net_overhead = e2e - server_infer
        print(f"  Net+decode+pack overhead:  {net_overhead:.2f} ms  ({net_overhead/e2e:.2%} of e2e)")
        if net_overhead > 0.5 * server_infer:
            print(f"  → {YELLOW}Investigate{RESET}: client-side transport adds > 50% over inference.")
            print(f"     If JPEG fix doesn't close the gap, colocation may be worth the complexity.")
        else:
            print(f"  → {GREEN}Websocket transport is fine.{RESET} "
                  f"Colocation would shave at most {net_overhead:.1f} ms — not worth the "
                  f"GIL contention, dependency conflicts, and crash containment loss.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
