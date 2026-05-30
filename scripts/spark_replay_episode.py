"""Replay a held-out OpenArm chocolate_bars episode through two
websocket policy servers and compare predicted action chunks frame-by-
frame against each other and against the teleoperator ground truth.

Why this script exists
----------------------
Phase 4's parity script (``spark_phase4_parity.py``) sends a stratified
calibration sample set (80 frames, 1 per episode, evenly spaced) to
two servers and reports cosine / ratio. That measures inference
*disagreement* on stratified data but doesn't tell us:

  1. Whether either server produces actions that look like what a human
     teleoperator did on the same frame (= "would this work on a robot?")
  2. Whether the two servers' predictions are sequentially consistent
     (= "would a robot trajectory be smooth?")
  3. Whether predicted actions stay inside the q01..q99 joint-radian
     bounds (= "would this command slam a joint into its limit?")
  4. How disagreement on real temporal sequences compares to the
     stratified parity number (calib may be optimistic — the heavy
     tails happen during contact-rich grasps, not on randomly-sampled
     frames)

This script does that, on a single chocolate_bars episode of your
choosing, with no robot in the loop. Output is one CSV per server (so
you can plot per-frame disagreement, or feed into a notebook) plus a
text summary with cosine percentiles, smoothness stats, and
out-of-range action counts.

Both servers must already be running. The script auto-detects each
server's chunk_size from ``policy.get_server_metadata()``; the openpi
serve_policy needs the chunk_size metadata patch (see
docs/spark_status.md "Phase 4 hardware-verified results" for the
launch incantation).

Usage:

  # 30-second held-out replay against both h=10 servers
  python3 scripts/spark_replay_episode.py \\
      --dataset-dir ~/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4 \\
      --episode 3 \\
      --stride 1 \\
      --ref-server localhost:8000 \\
      --sut-server localhost:8002 \\
      --output-dir /tmp/replay_ep3

  # quick stride=10 pass (every 10th frame, ~6x faster) to sanity-check
  # the script before the full run
  python3 scripts/spark_replay_episode.py --episode 3 --stride 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_meta(dataset_dir: Path) -> tuple[dict, dict, list]:
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    tasks: dict[int, str] = {}
    with (dataset_dir / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            tasks[int(d["task_index"])] = d["task"]
    episodes = []
    with (dataset_dir / "meta" / "episodes.jsonl").open() as f:
        for line in f:
            episodes.append(json.loads(line))
    return info, tasks, episodes


def _ep_paths(dataset_dir: Path, info: dict, ep_idx: int) -> tuple[Path, list[Path]]:
    chunk = ep_idx // info["chunks_size"]
    parquet = dataset_dir / info["data_path"].format(
        episode_chunk=chunk, episode_index=ep_idx)
    video_keys = ("observation.images.ego",
                  "observation.images.left_wrist",
                  "observation.images.right_wrist")
    videos = [dataset_dir / info["video_path"].format(
        episode_chunk=chunk, video_key=vk, episode_index=ep_idx)
        for vk in video_keys]
    return parquet, videos


def _read_episode_table(parquet: Path) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    t = pq.read_table(parquet)
    return {
        "state":       np.stack([np.asarray(s, np.float32)
                                 for s in t.column("observation.state").to_pylist()]),
        "action":      np.stack([np.asarray(a, np.float32)
                                 for a in t.column("action").to_pylist()]),
        "frame_index": np.asarray(t.column("frame_index").to_pylist(), np.int32),
        "task_index":  np.asarray(t.column("task_index").to_pylist(), np.int32),
    }


class _SequentialDecoder:
    """Decode an mp4 stream frame by frame, lazily. Each call to
    ``frame(target_idx)`` advances to that index from wherever the
    cursor currently is. Cheap when called with monotonically
    increasing indices (the replay case); O(N) when called out of
    order. The AV1-encoded openarm videos cost ~3-5 ms per decoded
    frame on Spark, so a 1000-frame episode is ~5 s of decode wall.
    """

    def __init__(self, video_path: Path):
        import av
        self._container = av.open(str(video_path))
        self._stream = self._container.streams.video[0]
        self._iter = self._container.decode(self._stream)
        self._next_idx = 0
        self._last_frame: np.ndarray | None = None

    def frame(self, target_idx: int) -> np.ndarray:
        if target_idx < self._next_idx - 1:
            raise ValueError(
                f"_SequentialDecoder: cannot rewind ({target_idx} < "
                f"{self._next_idx - 1}); recreate the decoder")
        while self._next_idx <= target_idx:
            frame = next(self._iter)
            self._last_frame = frame.to_ndarray(format="rgb24")
            self._next_idx += 1
        assert self._last_frame is not None
        return self._last_frame

    def close(self):
        self._container.close()


def _chw(hwc: np.ndarray) -> np.ndarray:
    """openpi serve expects (3, 224, 224) uint8 per camera; FlashRT's
    adapter auto-transposes either layout. Sending CHW matches the
    canonical convention so both servers see byte-identical input."""
    arr = hwc.astype(np.uint8, copy=False)
    if arr.shape[-1] == 3:
        return np.transpose(arr, (2, 0, 1))
    return arr


def _build_obs(state: np.ndarray, cams: list[np.ndarray], prompt: str) -> dict:
    return {
        "state": state.astype(np.float32),
        "images": {
            "cam_high":         _chw(cams[0]),
            "cam_left_wrist":   _chw(cams[1]),
            "cam_right_wrist":  _chw(cams[2]),
        },
        "prompt": prompt,
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    na, nb = np.linalg.norm(af), np.linalg.norm(bf)
    if na < 1e-30 or nb < 1e-30:
        return 0.0
    return float(af @ bf / (na * nb))


def _format_percentiles(vals: np.ndarray, label: str) -> str:
    if vals.size == 0:
        return f"{label}: (no samples)"
    return (f"{label}: n={vals.size}  "
            f"min={vals.min():+.4f}  p5={np.percentile(vals, 5):+.4f}  "
            f"median={np.median(vals):+.4f}  p95={np.percentile(vals, 95):+.4f}  "
            f"max={vals.max():+.4f}  mean={vals.mean():+.4f}")


def _safe_metadata(policy) -> dict:
    try:
        return dict(policy.get_server_metadata() or {})
    except Exception:
        return {}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path,
        default=Path.home() / ".cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parser.add_argument("--episode", type=int, required=True,
        help="Episode index to replay (e.g. 3, 100, 215)")
    parser.add_argument("--stride", type=int, default=1,
        help="Sample every Nth frame (1 = every frame at 50 Hz; 5 = 10 Hz; "
             "useful for fast sanity check before a full run)")
    parser.add_argument("--max-frames", type=int, default=0,
        help="Stop after N processed frames (0 = whole episode)")
    parser.add_argument("--skip-leading", type=int, default=50,
        help="Skip the first N frames (cameras auto-exposing). Default 50 = 1 s.")
    parser.add_argument("--ref-server", default="localhost:8000",
        help="Reference server (host:port). Default openpi JAX on 8000.")
    parser.add_argument("--sut-server", default="localhost:8002",
        help="System-under-test server (host:port). Default FlashRT on 8002.")
    parser.add_argument("--prompt-override", default=None,
        help="Use this prompt for every frame instead of the task's "
             "actual string. Useful for keeping FlashRT's CUDA graph "
             "warm (any prompt change triggers a recapture).")
    parser.add_argument("--ref-state-mode", choices=["normal", "zeros", "mean"],
        default="normal",
        help="What to put in the 'state' field of the obs sent to the "
             "reference server only (sut always gets normal state). "
             "Use 'zeros' or 'mean' to deliberately starve the "
             "reference of proprioception and check whether its "
             "predictions degrade to match the SUT's profile. Default "
             "'normal' = unmodified.")
    parser.add_argument("--norm-stats-path", default=None,
        help="Path to openpi norm_stats.json. Only used when "
             "--ref-state-mode=mean (to compute (q01+q99)/2). Auto-"
             "detects from common checkpoint paths if unset.")
    parser.add_argument("--output-dir", type=Path, default=None,
        help="If set, write per-frame CSVs (ref, sut) and a summary "
             "JSON here. If unset, only print summary to stdout.")
    args = parser.parse_args(argv)

    info, tasks, episodes = _load_meta(args.dataset_dir)
    ep_meta = next((e for e in episodes if e["episode_index"] == args.episode), None)
    if ep_meta is None:
        print(f"ERROR: episode {args.episode} not in episodes.jsonl", file=sys.stderr)
        return 1
    ep_len = int(ep_meta["length"])
    print(f"Episode {args.episode}: length={ep_len} frames "
          f"(~{ep_len / info['fps']:.1f} s at {info['fps']} Hz), "
          f"task='{ep_meta['tasks'][0]}'")

    parquet, video_paths = _ep_paths(args.dataset_dir, info, args.episode)
    for p in (parquet, *video_paths):
        if not p.is_file():
            print(f"ERROR: missing asset: {p}", file=sys.stderr)
            return 1

    table = _read_episode_table(parquet)
    if table["state"].shape[0] != ep_len:
        print(f"WARN: parquet has {table['state'].shape[0]} rows, "
              f"meta says {ep_len}", file=sys.stderr)

    task_idx = int(table["task_index"][0])
    prompt = args.prompt_override or tasks.get(task_idx, "do the task")
    print(f"Task index {task_idx} -> prompt='{prompt}'")

    ref_state_override: np.ndarray | None = None
    if args.ref_state_mode == "zeros":
        ref_state_override = np.zeros(16, dtype=np.float32)
        print("REF STATE OVERRIDE: zeros (16-D) — reference server will "
              "see state = 0 instead of the real joints")
    elif args.ref_state_mode == "mean":
        ns_path = args.norm_stats_path
        if ns_path is None:
            for cand in [
                "/home/evaughan/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/"
                "chocolate_bars_pi05/29999/assets/openarm/norm_stats.json",
            ]:
                if Path(cand).is_file():
                    ns_path = cand
                    break
        if ns_path is None:
            print("ERROR: --ref-state-mode=mean needs --norm-stats-path",
                  file=sys.stderr)
            return 1
        with open(ns_path) as f:
            stats = json.load(f)["norm_stats"]["state"]
        q01 = np.asarray(stats["q01"], dtype=np.float32)
        q99 = np.asarray(stats["q99"], dtype=np.float32)
        ref_state_override = ((q01 + q99) / 2.0).astype(np.float32)
        print(f"REF STATE OVERRIDE: mean = (q01+q99)/2 from {ns_path}\n"
              f"  override values: {ref_state_override.tolist()}")

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError as e:
        print(f"ERROR: openpi-client not installed: {e}", file=sys.stderr)
        return 1

    ref_host, ref_port = args.ref_server.split(":")
    sut_host, sut_port = args.sut_server.split(":")
    print(f"Connecting:  ref={args.ref_server}  sut={args.sut_server}")
    ref = WebsocketClientPolicy(host=ref_host, port=int(ref_port))
    sut = WebsocketClientPolicy(host=sut_host, port=int(sut_port))
    ref_meta = _safe_metadata(ref)
    sut_meta = _safe_metadata(sut)
    print(f"  ref metadata: {ref_meta}")
    print(f"  sut metadata: {sut_meta}")

    decoders = [_SequentialDecoder(vp) for vp in video_paths]

    frame_indices = sorted([int(x) for x in table["frame_index"].tolist()])
    sampled = [f for f in frame_indices
               if f >= args.skip_leading and (f - args.skip_leading) % args.stride == 0]
    if args.max_frames > 0:
        sampled = sampled[: args.max_frames]
    print(f"Replaying {len(sampled)} frames (stride={args.stride}, "
          f"skip_leading={args.skip_leading})")

    fr_to_row = {int(f): i for i, f in enumerate(table["frame_index"].tolist())}

    per_frame: list[dict] = []
    ref_lat_ms: list[float] = []
    sut_lat_ms: list[float] = []
    cos_first_step: list[float] = []   # cos(ref[0], sut[0])
    cos_full_chunk: list[float] = []   # cos(ref[:K], sut[:K]) where K=min chunk
    cos_ref_vs_gt: list[float] = []    # cos(ref[0], gt_action[t])
    cos_sut_vs_gt: list[float] = []    # cos(sut[0], gt_action[t])
    ref_minus_gt_l2: list[float] = []
    sut_minus_gt_l2: list[float] = []
    # Smoothness: track cos between successive sut[0] predictions.
    prev_sut0: np.ndarray | None = None
    sut0_consec_cos: list[float] = []
    prev_ref0: np.ndarray | None = None
    ref0_consec_cos: list[float] = []

    t_start = time.time()
    for k, fidx in enumerate(sampled):
        row = fr_to_row.get(fidx)
        if row is None:
            print(f"  WARN: frame_idx {fidx} not in parquet", file=sys.stderr)
            continue
        state = table["state"][row]
        gt_action_now = table["action"][row]
        cams = [d.frame(fidx) for d in decoders]
        sut_obs = _build_obs(state, cams, prompt)
        if ref_state_override is not None:
            ref_obs = _build_obs(ref_state_override, cams, prompt)
        else:
            ref_obs = sut_obs

        t0 = time.time()
        ra = np.asarray(ref.infer(ref_obs)["actions"])
        rt = (time.time() - t0) * 1000.0
        t0 = time.time()
        sa = np.asarray(sut.infer(sut_obs)["actions"])
        st = (time.time() - t0) * 1000.0

        K = min(ra.shape[0], sa.shape[0])
        D = min(ra.shape[1], sa.shape[1])
        rK, sK = ra[:K, :D], sa[:K, :D]
        gt_D = gt_action_now[:D]

        c_first = _cos(rK[0], sK[0])
        c_chunk = _cos(rK, sK)
        c_ref_gt = _cos(rK[0], gt_D)
        c_sut_gt = _cos(sK[0], gt_D)

        cos_first_step.append(c_first)
        cos_full_chunk.append(c_chunk)
        cos_ref_vs_gt.append(c_ref_gt)
        cos_sut_vs_gt.append(c_sut_gt)
        ref_minus_gt_l2.append(float(np.linalg.norm(rK[0] - gt_D)))
        sut_minus_gt_l2.append(float(np.linalg.norm(sK[0] - gt_D)))
        ref_lat_ms.append(rt)
        sut_lat_ms.append(st)

        if prev_sut0 is not None:
            sut0_consec_cos.append(_cos(prev_sut0, sK[0]))
        if prev_ref0 is not None:
            ref0_consec_cos.append(_cos(prev_ref0, rK[0]))
        prev_sut0 = sK[0].copy()
        prev_ref0 = rK[0].copy()

        per_frame.append({
            "frame": fidx, "k": k,
            "ref_shape": ra.shape, "sut_shape": sa.shape,
            "cos_first": c_first, "cos_chunk": c_chunk,
            "cos_ref_vs_gt": c_ref_gt, "cos_sut_vs_gt": c_sut_gt,
            "l2_ref_gt": ref_minus_gt_l2[-1], "l2_sut_gt": sut_minus_gt_l2[-1],
            "ref_ms": rt, "sut_ms": st,
            "ref_first": rK[0], "sut_first": sK[0], "gt": gt_D,
        })

        if k < 5 or k % 50 == 0:
            print(f"  [{k:4d}] frame={fidx:4d}  "
                  f"cos(F,O)={c_first:+.4f}  "
                  f"cos(O,gt)={c_ref_gt:+.4f}  cos(F,gt)={c_sut_gt:+.4f}  "
                  f"L2(F,gt)={sut_minus_gt_l2[-1]:.3f}  "
                  f"ref={rt:.0f}ms  sut={st:.0f}ms")

    for d in decoders:
        d.close()

    wall = time.time() - t_start
    print(f"\nReplay complete in {wall:.1f} s "
          f"(~{wall / max(1, len(per_frame)) * 1000:.0f} ms/frame)")

    print("\n=== Server-vs-server agreement (FlashRT vs openpi) ===")
    print(_format_percentiles(np.array(cos_first_step),
                              "  cos at first step  "))
    print(_format_percentiles(np.array(cos_full_chunk),
                              "  cos over full 10-step chunk"))

    print("\n=== Predicted vs teleoperator ground truth ===")
    print(_format_percentiles(np.array(cos_ref_vs_gt),
                              "  cos(openpi-step0, teleop)"))
    print(_format_percentiles(np.array(cos_sut_vs_gt),
                              "  cos(FlashRT-step0, teleop)"))
    print(_format_percentiles(np.array(ref_minus_gt_l2),
                              "  ||openpi - teleop|| (rad)"))
    print(_format_percentiles(np.array(sut_minus_gt_l2),
                              "  ||FlashRT - teleop|| (rad)"))

    print("\n=== Trajectory smoothness (cos of successive step-0 predictions) ===")
    print(_format_percentiles(np.array(ref0_consec_cos),
                              "  consecutive cos (openpi)"))
    print(_format_percentiles(np.array(sut0_consec_cos),
                              "  consecutive cos (FlashRT)"))

    print("\n=== Latency ===")
    rla = np.array(ref_lat_ms); sla = np.array(sut_lat_ms)
    print(f"  ref: p50={np.median(rla):.0f}ms  p99={np.percentile(rla, 99):.0f}ms  "
          f"min={rla.min():.0f}ms  max={rla.max():.0f}ms")
    print(f"  sut: p50={np.median(sla):.0f}ms  p99={np.percentile(sla, 99):.0f}ms  "
          f"min={sla.min():.0f}ms  max={sla.max():.0f}ms")
    print(f"  speedup (sut/ref p50): {np.median(rla) / max(1.0, np.median(sla)):.2f}x")

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"replay_ep{args.episode}.csv"
        with csv_path.open("w") as f:
            w = csv.writer(f)
            w.writerow([
                "frame", "cos_first", "cos_chunk",
                "cos_ref_vs_gt", "cos_sut_vs_gt",
                "l2_ref_gt", "l2_sut_gt",
                "ref_ms", "sut_ms",
                *[f"ref_first_{i}" for i in range(16)],
                *[f"sut_first_{i}" for i in range(16)],
                *[f"gt_{i}" for i in range(16)],
            ])
            for r in per_frame:
                w.writerow([
                    r["frame"], r["cos_first"], r["cos_chunk"],
                    r["cos_ref_vs_gt"], r["cos_sut_vs_gt"],
                    r["l2_ref_gt"], r["l2_sut_gt"],
                    r["ref_ms"], r["sut_ms"],
                    *r["ref_first"].tolist(),
                    *r["sut_first"].tolist(),
                    *r["gt"].tolist(),
                ])
        print(f"\nPer-frame CSV: {csv_path}")

        summary = {
            "episode": args.episode,
            "prompt": prompt,
            "stride": args.stride,
            "n_frames": len(per_frame),
            "ref_server": args.ref_server,
            "sut_server": args.sut_server,
            "ref_metadata": ref_meta,
            "sut_metadata": sut_meta,
            "cos_first_step": {
                "median": float(np.median(cos_first_step)),
                "p5":     float(np.percentile(cos_first_step, 5)),
                "min":    float(np.min(cos_first_step)),
            },
            "cos_ref_vs_gt": {
                "median": float(np.median(cos_ref_vs_gt)),
                "p5":     float(np.percentile(cos_ref_vs_gt, 5)),
            },
            "cos_sut_vs_gt": {
                "median": float(np.median(cos_sut_vs_gt)),
                "p5":     float(np.percentile(cos_sut_vs_gt, 5)),
            },
            "l2_sut_gt_median_rad": float(np.median(sut_minus_gt_l2)),
            "l2_ref_gt_median_rad": float(np.median(ref_minus_gt_l2)),
            "latency_ms": {
                "ref_p50": float(np.median(rla)), "ref_p99": float(np.percentile(rla, 99)),
                "sut_p50": float(np.median(sla)), "sut_p99": float(np.percentile(sla, 99)),
            },
        }
        sum_path = args.output_dir / f"replay_ep{args.episode}_summary.json"
        sum_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary JSON: {sum_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
