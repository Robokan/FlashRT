"""Standalone CLI launcher for the chunked websocket client.

Sister tool to ``flash_rt/serving/chunked_websocket_client.py``. The
wrapper itself is meant to be imported (by SparkJAX, or whatever the
robot stack is); this CLI lets you exercise it without that integration:

    * smoke-test that the wrapper actually connects + serves actions
    * sweep blending modes 1..4 against a backend, compare latency
      and chunk-swap behaviour
    * sanity-check a new FlashRT or openpi-JAX deployment

It is NOT a robot driver: there is no FIFO write, no safety check,
no CAN bus. Observations come from a calibration npz (the same Phase
4 calib_data file works) or from synthetic zeros/random.

Examples::

    # Mode 2 (default) against the FlashRT server, 50 steps, npz obs.
    python scripts/robot_client_chunked.py \\
        --server-url ws://localhost:8002 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-steps 50

    # Sweep all 4 modes against openpi-JAX, prints per-mode stats.
    python scripts/robot_client_chunked.py \\
        --server-url ws://localhost:8000 \\
        --calib-data /tmp/calib_openarm_v4_80.npz \\
        --num-steps 30 \\
        --sweep-modes

    # Synthetic-zeros obs (no calib data needed); useful for
    # connection / handshake smoke test.
    python scripts/robot_client_chunked.py \\
        --server-url ws://localhost:8002 \\
        --obs-source synthetic-zeros \\
        --prompt 'put the chocolate bars in the container' \\
        --num-steps 20
"""

from __future__ import annotations

import argparse
import logging
import statistics
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


def _build_synthetic_obs(
    prompt: str, state_dim: int, num_views: int, mode: str
) -> dict:
    """Synthetic observation matching the openpi server's expected shape.

    Mode ``synthetic-zeros`` returns all-zeros (deterministic,
    cheapest). Mode ``synthetic-random`` returns numpy random values
    (per-call new seed; not reproducible).
    """
    if mode == "synthetic-zeros":
        state = np.zeros(state_dim, dtype=np.float32)
        images = {f"cam_{i}": np.zeros((224, 224, 3), dtype=np.uint8)
                  for i in range(num_views)}
    elif mode == "synthetic-random":
        state = np.random.uniform(-1.0, 1.0, state_dim).astype(np.float32)
        images = {f"cam_{i}": np.random.randint(
            0, 255, (224, 224, 3), dtype=np.uint8)
                  for i in range(num_views)}
    else:
        raise ValueError(f"unknown synthetic mode: {mode}")
    canonical_cams = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
    out_images = {}
    for i, raw_cam in enumerate(images.values()):
        if i >= len(canonical_cams):
            break
        out_images[canonical_cams[i]] = raw_cam
    return {
        "state": state,
        "images": out_images,
        "prompt": prompt,
    }


def _chw(hwc: np.ndarray) -> np.ndarray:
    """Convert HWC uint8 -> CHW uint8 (openpi server input convention).

    Mirrors ``scripts/spark_phase4_parity.py:_chw``.
    """
    return np.ascontiguousarray(np.transpose(hwc.astype(np.uint8), (2, 0, 1)))


_CAM_KEY_ALIASES = (
    # Phase 3/4 npz convention -> server-side openpi cam name
    ("images_ego",   "cam_high"),
    ("images_left",  "cam_left_wrist"),
    ("images_right", "cam_right_wrist"),
    # Already-canonical fallback (some older calib npz files).
    ("cam_high",         "cam_high"),
    ("cam_left_wrist",   "cam_left_wrist"),
    ("cam_right_wrist",  "cam_right_wrist"),
)


def _load_calib_obs(path: Path) -> list[dict]:
    """Read the Phase 3/4 calib npz and reshape to a list of obs dicts.

    Schema observed: ``images_ego``/``images_left``/``images_right``
    HWC uint8 arrays ``(N, H, W, 3)``, ``state`` ``(N, state_dim)
    float32``, ``prompts`` ``(N,)`` of strings. Falls back to the
    canonical ``cam_high``/``cam_left_wrist``/``cam_right_wrist``
    names if the npz uses them directly.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    per_cam: list[tuple[str, np.ndarray]] = []
    seen_targets: set[str] = set()
    for src, dst in _CAM_KEY_ALIASES:
        if src in keys and dst not in seen_targets:
            per_cam.append((dst, data[src]))
            seen_targets.add(dst)
    if not per_cam:
        raise ValueError(
            f"calib npz {path} has no recognised camera arrays. "
            f"Keys: {keys}. Expected one of "
            f"{[a[0] for a in _CAM_KEY_ALIASES]}.")
    n = len(per_cam[0][1])
    states = data["state"] if "state" in keys else None
    prompts = data["prompts"] if "prompts" in keys else None
    obs_list: list[dict] = []
    for i in range(n):
        images_chw = {name: _chw(arr[i]) for name, arr in per_cam}
        obs: dict = {"images": images_chw}
        if states is not None:
            obs["state"] = np.asarray(states[i], dtype=np.float32)
        if prompts is not None and len(prompts) > i and prompts[i]:
            obs["prompt"] = str(prompts[i])
        obs_list.append(obs)
    return obs_list


def _iter_obs(args: argparse.Namespace, prompt: str) -> "list[dict]":
    if args.obs_source.startswith("synthetic-"):
        return [_build_synthetic_obs(prompt, args.state_dim, args.num_views,
                                     args.obs_source)
                for _ in range(args.num_steps)]
    if args.obs_source == "calib-npz":
        if args.calib_data is None:
            raise ValueError("--obs-source calib-npz requires --calib-data")
        all_obs = _load_calib_obs(Path(args.calib_data))
        if not all_obs:
            raise ValueError(f"calib npz at {args.calib_data} is empty")
        out = []
        for i in range(args.num_steps):
            obs = dict(all_obs[i % len(all_obs)])
            if "prompt" not in obs:
                obs["prompt"] = prompt
            out.append(obs)
        return out
    raise ValueError(f"unknown --obs-source: {args.obs_source}")


def _run_mode(
    client_factory,
    obs_iter: list[dict],
    mode: int,
    verbose: bool,
    target_hz: float,
) -> dict:
    """Run a single mode against the given obs iterable, return stats.

    The loop paces calls at ``target_hz`` to give the AsyncChunkRunner's
    background inference time to complete between consumer ticks. Without
    pacing the consumer races past every chunk boundary before the new
    chunk is ready and the runner falls back to ``miss_policy="hold_last"``
    on every step — which exercises the deadline path but doesn't
    represent real robot behaviour.
    """
    from flash_rt.serving.chunked_websocket_client import (
        ChunkedWebsocketClient, MODE_DESCRIPTIONS)
    print(f"\n{BOLD}=== Mode {mode}: {MODE_DESCRIPTIONS[mode]} ==={RESET}")
    client: ChunkedWebsocketClient = client_factory(mode)
    print(f"  chunk_len resolved to {client.chunk_len}")
    period_s = 1.0 / target_hz
    latencies_ms: list[float] = []
    first_latency_ms = None
    next_deadline = time.monotonic() + period_s
    try:
        for step, obs in enumerate(obs_iter):
            t0 = time.monotonic()
            action = client.next_action(obs)
            dt_ms = (time.monotonic() - t0) * 1000.0
            latencies_ms.append(dt_ms)
            if step == 0:
                first_latency_ms = dt_ms
            spike = " *SLOW" if dt_ms > 1.5 * period_s * 1000 else ""
            if verbose or step < 5 or step % 10 == 0:
                action_arr = np.asarray(action)
                preview = action_arr[:4] if action_arr.size > 4 else action_arr
                print(f"  step {step:3d}: dt={dt_ms:6.1f} ms  "
                      f"action[:4]={preview.tolist()!r:.45s}{spike}")
            # Pace to target_hz so background inference can complete
            # between calls. Drop the deadline (don't sleep negative)
            # if we already overran.
            now = time.monotonic()
            sleep_s = next_deadline - now
            if sleep_s > 0:
                time.sleep(sleep_s)
                next_deadline += period_s
            else:
                next_deadline = now + period_s
        stats = client.stats
        if len(latencies_ms) >= 2:
            p50 = statistics.median(latencies_ms)
            steady = latencies_ms[1:]
            steady_p50 = (statistics.median(steady) if steady else p50)
        else:
            p50 = latencies_ms[0]
            steady_p50 = p50
        return {
            "mode": mode,
            "first_latency_ms": first_latency_ms,
            "all_p50_ms": p50,
            "steady_p50_ms": steady_p50,
            "max_ms": max(latencies_ms),
            "n": len(latencies_ms),
            "chunks_started": stats.chunks_started,
            "chunks_completed": stats.chunks_completed,
            "actions_served": stats.actions_served,
            "deadline_misses": stats.deadline_misses,
            "swaps": stats.swaps,
        }
    finally:
        client.close()


def _print_sweep_table(rows: list[dict]) -> None:
    print(f"\n{BOLD}=== Per-mode summary ==={RESET}")
    header = (f"  {'mode':<6s} {'first_ms':>10s} {'steady_p50':>12s} "
              f"{'max_ms':>10s} {'served':>8s} {'swaps':>8s} {'misses':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(f"  {r['mode']:<6d} {r['first_latency_ms']:>10.1f} "
              f"{r['steady_p50_ms']:>12.1f} {r['max_ms']:>10.1f} "
              f"{r['actions_served']:>8d} {r['swaps']:>8d} "
              f"{r['deadline_misses']:>8d}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--server-url", default="ws://localhost:8002",
        help="Websocket URL of the policy server (default: ws://localhost:8002)")
    parser.add_argument(
        "--blending-mode", type=int, default=2, choices=(1, 2, 3, 4),
        help="Blending mode 1..4 (default 2 = async pipelined, no blend)")
    parser.add_argument(
        "--target-hz", type=float, default=25.0,
        help="Controller rate (default 25 Hz)")
    parser.add_argument(
        "--num-steps", type=int, default=50,
        help="How many actions to consume (default 50)")
    parser.add_argument(
        "--prompt", default="put the chocolate bars in the container",
        help="Default prompt for synthetic obs / missing per-sample prompts")
    parser.add_argument(
        "--obs-source", default="calib-npz",
        choices=("calib-npz", "synthetic-zeros", "synthetic-random"),
        help="Observation source. 'calib-npz' replays real frames "
             "from --calib-data; the synthetic modes need no data.")
    parser.add_argument(
        "--calib-data", default=None,
        help="Path to a Phase 3/4 calib npz with images + state + prompts. "
             "Required when --obs-source=calib-npz.")
    parser.add_argument(
        "--state-dim", type=int, default=16,
        help="State vector dimension for synthetic obs (default 16 = OpenArm)")
    parser.add_argument(
        "--num-views", type=int, default=3,
        help="Number of camera views for synthetic obs (default 3)")
    parser.add_argument(
        "--chunk-len-override", type=int, default=None,
        help="Override H instead of reading server metadata. Use when "
             "talking to a server that doesn't publish chunk_size.")
    parser.add_argument(
        "--sweep-modes", action="store_true",
        help="Run all 4 modes back-to-back and print a comparison table")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every step's action preview (default: every 10th)")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as e:
        print(f"{RED}openpi-client not installed: {e}{RESET}", file=sys.stderr)
        print("Install: uv pip install -e ~/sparkpack/openpi/packages/openpi-client",
              file=sys.stderr)
        return 1

    print(f"{BOLD}Server  :{RESET} {args.server_url}")
    print(f"{BOLD}Obs     :{RESET} {args.obs_source}"
          + (f" from {args.calib_data}" if args.calib_data else ""))
    print(f"{BOLD}Steps   :{RESET} {args.num_steps} @ {args.target_hz} Hz")

    if args.server_url.startswith("ws://"):
        host_port = args.server_url[5:]
    elif args.server_url.startswith("ws"):
        host_port = args.server_url[2:]
    else:
        host_port = args.server_url
    if ":" in host_port:
        host, port_str = host_port.split(":", 1)
        port = int(port_str)
    else:
        host, port = host_port, None

    print(f"{DIM}Connecting to {host}:{port} ...{RESET}")
    try:
        policy = WebsocketClientPolicy(host=host, port=port)
    except Exception as e:
        print(f"{RED}Failed to connect: {e}{RESET}", file=sys.stderr)
        return 2
    print(f"{GREEN}Connected.{RESET} server metadata: {policy.get_server_metadata()}")

    obs_iter = _iter_obs(args, args.prompt)
    if not obs_iter:
        print(f"{RED}No observations to feed; nothing to do.{RESET}",
              file=sys.stderr)
        return 3

    from flash_rt.serving.chunked_websocket_client import ChunkedWebsocketClient

    def client_factory(mode: int) -> ChunkedWebsocketClient:
        return ChunkedWebsocketClient(
            policy,
            blending_mode=mode,
            target_hz=args.target_hz,
            chunk_len_override=args.chunk_len_override)

    if args.sweep_modes:
        rows = []
        for mode in (1, 2, 3, 4):
            rows.append(_run_mode(client_factory, obs_iter, mode,
                                  args.verbose, args.target_hz))
        _print_sweep_table(rows)
    else:
        row = _run_mode(client_factory, obs_iter, args.blending_mode,
                        args.verbose, args.target_hz)
        _print_sweep_table([row])

    return 0


if __name__ == "__main__":
    sys.exit(main())
