"""Phase 6 — compare task success between FlashRT and JAX-server on a
real OpenArm robot.

Phase 6's acceptance gate is "≥ 95% of the JAX-server task success rate".
This script doesn't drive the robot directly — that code lives in the
user's robot client (uses ``AsyncActionChunkBroker``). It is the
*scaffold* around the robot loop that:

  1. Lets you record per-episode results to a JSON log.
  2. Compares two recorded logs (FlashRT vs JAX) and computes the
     success-rate ratio with a binomial confidence interval.

How to use:

A. Record JAX-server baseline (or look up existing numbers):
        # In the JAX server's shell:
        python scripts/serve_policy.py --policy.config=pi05_openarm_ngc_lora \\
            --policy.dir=/openpi_assets/pi05_openarm_ngc_lora_v4 --port=8001

        # In your robot client, run N episodes (e.g. 20 per task * 5 tasks = 100).
        # After each episode, append to a JSON file like:
        # {"server": "jax", "task": "pick_red_block", "success": true,
        #  "episode_steps": 87, "timestamp": "..."}

   Use the ``--append`` mode of this script to record:
        python3 scripts/spark_phase6_robot_compare.py --append \\
            --log jax_baseline.json --server jax --task pick_red_block --success true

B. Record FlashRT-on-Spark:
        # Same robot loop, FlashRT server instead of JAX server.
        python3 scripts/spark_phase6_robot_compare.py --append \\
            --log flashrt_spark.json --server flashrt --task pick_red_block --success false

C. Compare:
        python3 scripts/spark_phase6_robot_compare.py --compare \\
            --baseline jax_baseline.json --candidate flashrt_spark.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson-score 95% CI for a binomial proportion (better than Wald
    on small n / extremes)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def cmd_append(args: argparse.Namespace) -> int:
    log = Path(args.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict] = []
    if log.exists():
        try:
            entries = json.loads(log.read_text())
        except json.JSONDecodeError:
            print(f"{YELLOW}WARN: existing log {log} is not valid JSON; starting fresh{RESET}")
            entries = []
    entry = {
        "server": args.server,
        "task": args.task,
        "success": args.success.lower() in ("true", "1", "yes", "y", "t"),
        "episode_steps": args.episode_steps,
        "notes": args.notes,
        "timestamp": time.time(),
    }
    entries.append(entry)
    log.write_text(json.dumps(entries, indent=2))
    print(f"{GREEN}appended{RESET}  {log}  n={len(entries)}  {entry}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    base = Path(args.baseline)
    cand = Path(args.candidate)
    if not base.is_file():
        print(f"{RED}baseline log missing:{RESET} {base}")
        return 1
    if not cand.is_file():
        print(f"{RED}candidate log missing:{RESET} {cand}")
        return 1
    baseline = json.loads(base.read_text())
    candidate = json.loads(cand.read_text())

    def aggregate(entries: list[dict]) -> dict[str, tuple[int, int]]:
        """task → (successes, total)."""
        out: dict[str, tuple[int, int]] = defaultdict(lambda: (0, 0))
        for e in entries:
            t = e.get("task", "unknown")
            s, n = out[t]
            out[t] = (s + (1 if e.get("success") else 0), n + 1)
        return dict(out)

    b_agg = aggregate(baseline)
    c_agg = aggregate(candidate)

    all_tasks = sorted(set(b_agg) | set(c_agg))
    print(f"\n{BOLD}{'Task':<40s}  {'JAX':<18s}  {'FlashRT':<18s}  {'ratio':<10s}  gate{RESET}")
    print("-" * 100)

    b_total_s = b_total_n = 0
    c_total_s = c_total_n = 0
    failed_tasks: list[str] = []

    for t in all_tasks:
        bs, bn = b_agg.get(t, (0, 0))
        cs, cn = c_agg.get(t, (0, 0))
        b_total_s += bs
        b_total_n += bn
        c_total_s += cs
        c_total_n += cn

        b_rate = bs / bn if bn else float("nan")
        c_rate = cs / cn if cn else float("nan")
        ratio = c_rate / b_rate if b_rate > 0 else float("nan")

        gate = ratio >= args.threshold if not math.isnan(ratio) else False
        gate_str = f"{GREEN}PASS{RESET}" if gate else f"{RED}FAIL{RESET}"
        if not gate and not math.isnan(ratio):
            failed_tasks.append(t)

        b_lo, b_hi = _wilson_ci(bs, bn)
        c_lo, c_hi = _wilson_ci(cs, cn)
        print(f"  {t:<38s}  {bs:2d}/{bn:2d} ({b_rate:.2%})  "
              f"{cs:2d}/{cn:2d} ({c_rate:.2%})  "
              f"{ratio:.3f}     {gate_str}")
        print(f"  {' ':<38s}  CI=[{b_lo:.2%},{b_hi:.2%}]  CI=[{c_lo:.2%},{c_hi:.2%}]")

    # Overall.
    overall_b = b_total_s / b_total_n if b_total_n else 0.0
    overall_c = c_total_s / c_total_n if c_total_n else 0.0
    overall_ratio = overall_c / overall_b if overall_b > 0 else float("nan")

    print("-" * 100)
    print(f"  {BOLD}{'OVERALL':<38s}  {b_total_s:2d}/{b_total_n:2d} ({overall_b:.2%})  "
          f"{c_total_s:2d}/{c_total_n:2d} ({overall_c:.2%})  "
          f"{overall_ratio:.3f}{RESET}")

    print()
    if math.isnan(overall_ratio):
        print(f"{RED}No common tasks with non-zero JAX baseline.{RESET}")
        return 1
    if overall_ratio >= args.threshold:
        print(f"{BOLD}{GREEN}Phase 6 PASSED{RESET}  "
              f"FlashRT/JAX = {overall_ratio:.3f} ≥ {args.threshold} threshold")
        if failed_tasks:
            print(f"{DIM}Note: per-task failures: {failed_tasks}{RESET}")
        return 0
    print(f"{BOLD}{RED}Phase 6 FAILED{RESET}  "
          f"FlashRT/JAX = {overall_ratio:.3f} < {args.threshold} threshold")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 6 record + compare")
    sub = parser.add_subparsers(dest="cmd")

    p_append = sub.add_parser("append", help="Append an episode result to a log")
    p_append.add_argument("--log", required=True)
    p_append.add_argument("--server", required=True, choices=["jax", "flashrt"])
    p_append.add_argument("--task", required=True)
    p_append.add_argument("--success", required=True,
                          help="true/false/yes/no/1/0")
    p_append.add_argument("--episode-steps", type=int, default=None)
    p_append.add_argument("--notes", default="")
    p_append.set_defaults(func=cmd_append)

    p_compare = sub.add_parser("compare", help="Compare two episode logs")
    p_compare.add_argument("--baseline", required=True,
                           help="JSON log from JAX server runs")
    p_compare.add_argument("--candidate", required=True,
                           help="JSON log from FlashRT server runs")
    p_compare.add_argument("--threshold", type=float, default=0.95,
                           help="Acceptance threshold (default 0.95 = Phase 6 gate)")
    p_compare.set_defaults(func=cmd_compare)

    # Back-compat flags (so the help message in the module docstring works).
    parser.add_argument("--append", action="store_const", const="append",
                        dest="legacy_cmd")
    parser.add_argument("--compare", action="store_const", const="compare",
                        dest="legacy_cmd")

    args, extras = parser.parse_known_args()

    if args.cmd is None and getattr(args, "legacy_cmd", None) is not None:
        # Re-parse with the implicit subcommand.
        return main_with_implicit(args.legacy_cmd, extras)

    if args.cmd is None:
        parser.print_help()
        return 2

    return args.func(args)


def main_with_implicit(cmd: str, extras: list[str]) -> int:
    sys.argv = [sys.argv[0], cmd, *extras]
    return main()


if __name__ == "__main__":
    sys.exit(main())
