"""Phase 3 part 1 — prepare a stratified calibration sample set from a
LeRobot v2.1 dataset (OpenArm bimanual, 3 cameras).

The reader follows the canonical openpi parity pattern from
openpi/scripts/diag_quant_parity.py::_load_obs and
openpi/scripts/diag_live_server_parity.py — pyarrow for the per-episode
parquet (state, frame_index, task_index) + PyAV for the per-camera
.mp4 frame decode. We deliberately stay off the LeRobot Python API
because (a) the on-disk format is stable (LeRobot v2.1) and well-defined
in meta/info.json, (b) the lerobot Python package has had two
incompatible reorganisations in the last year (lerobot.common.* ->
lerobot.*) and pulls in pandas / torchvision / etc., and (c) openpi's
own parity scripts read the same files directly.

Cameras: OpenArm has 3 cameras. The dataset stores them as
``observation.images.{ego,left_wrist,right_wrist}``; we map them to the
openpi policy convention ``{cam_high, cam_left_wrist, cam_right_wrist}``
which the OpenArmInputs transform then re-keys to
``{base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}`` for the model.

Stratification: FP8 calibration computes per-tensor activation amax;
a biased sample (only one task, only the first frame of each episode,
only one timestep) under-estimates amax and produces scales that clip
real inference. We pick samples uniformly across episodes and within
each picked episode take an evenly-spaced frame after the first second
(skip-leading=50 at 50 Hz), so the robot is past the initial wait
period that cameras spend auto-exposing. Coverage across the 4 task
indices in v4 is automatic because each episode has exactly one task.

Output: a single npz with arrays (consumed by spark_phase3_run_calib.py)
  - images_ego        (N, 224, 224, 3) uint8
  - images_left       (N, 224, 224, 3) uint8
  - images_right      (N, 224, 224, 3) uint8
  - state             (N, 16) float32        (OpenArm bimanual state)
  - prompts           (N,) object            (task language strings)
  - episodes          (N,) int32             (source episode index)
  - frame_idx         (N,) int32             (frame within episode)
  - task_idx          (N,) int32

Usage:
    python3 scripts/spark_phase3_prepare_calib.py \
        --dataset-dir ~/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4 \
        --num-samples 80 \
        --output /tmp/calib_openarm_v4_80.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _load_dataset_meta(dataset_dir: Path) -> dict[str, Any]:
    """Read LeRobot v2.1 meta/{info,tasks,episodes}.jsonl and resolve paths."""
    info = json.loads((dataset_dir / "meta" / "info.json").read_text())
    tasks = {}
    for line in (dataset_dir / "meta" / "tasks.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        tasks[int(rec["task_index"])] = rec["task"]
    episodes = []
    for line in (dataset_dir / "meta" / "episodes.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        episodes.append(json.loads(line))
    return {"info": info, "tasks": tasks, "episodes": episodes}


def _episode_paths(dataset_dir: Path, info: dict, ep_idx: int) -> tuple[Path, Path]:
    """Return (parquet_path, episode_stem) for a given episode index."""
    chunk = ep_idx // info["chunks_size"]
    ep_str = f"episode_{ep_idx:06d}"
    parquet = (
        dataset_dir
        / info["data_path"].format(episode_chunk=chunk, episode_index=ep_idx)
    )
    return parquet, ep_str, f"chunk-{chunk:03d}"


def _decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode a single (1-indexed-feeling, 0-indexed-actual) frame from mp4.

    Mirrors openpi/scripts/diag_quant_parity.py::_load_obs: open + iterate
    decoded frames until the target index. av1 (the codec used in
    openarm-teleop-16dof-v4) decodes from the start of the file every
    time, so for 800-frame episodes the worst-case decode is ~16 s wall;
    that's acceptable for a one-shot calibration prep (80 frames * a few
    seconds each = a few minutes total).
    """
    import av  # PyAV
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                return frame.to_ndarray(format="rgb24")
    raise RuntimeError(
        f"could not decode frame {frame_index} from {video_path} "
        f"(file likely shorter than expected)")


def _resize_224(img: np.ndarray) -> np.ndarray:
    """Resize HxWx3 uint8 to 224x224 (no letterbox — v4 is already 224x224)."""
    if img.shape[:2] == (224, 224):
        return img
    import cv2
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)


def _sample_plan(
    num_samples: int,
    *,
    skip_leading: int,
    skip_trailing: int,
    rng: np.random.Generator,
    episode_lengths: list[int],
    episode_tasks: list[int],
) -> list[tuple[int, int]]:
    """Build a (episode_idx, frame_idx) list of stratified samples.

    Strategy:
      1. Bucket episodes by task_index (4 in v4 = mirrored variants of
         "put the chocolate bars in the container"). Round-robin across
         buckets so the sample is task-balanced even when the dataset is
         task-imbalanced.
      2. For each picked episode, choose a frame at a random uniform
         position within [skip_leading, episode_length - skip_trailing).
         When more than one frame is taken from the same episode, the
         positions are sorted-equal-spaced + jittered so the temporal
         coverage is even (calibration cares about the full action
         trajectory, not just the grasp-and-release moments).
    """
    n_episodes = len(episode_lengths)
    if n_episodes == 0:
        return []

    by_task: dict[int, list[int]] = {}
    for ep, tidx in enumerate(episode_tasks):
        by_task.setdefault(tidx, []).append(ep)
    task_keys = sorted(by_task)

    # Round-robin picking until we have enough (episode, k) pairs to
    # cover num_samples. k starts at 1 and grows when we run out of
    # distinct episodes within a task bucket.
    ep_picks: list[tuple[int, int]] = []
    target_distinct = max(min(num_samples, n_episodes), 1)
    picked = set()
    # Shuffle each bucket once so we don't always pick the same episodes.
    for tidx in task_keys:
        rng.shuffle(by_task[tidx])
    bucket_cursors = {tidx: 0 for tidx in task_keys}
    while len(picked) < target_distinct:
        progress = False
        for tidx in task_keys:
            if len(picked) >= target_distinct:
                break
            c = bucket_cursors[tidx]
            if c >= len(by_task[tidx]):
                continue
            ep = by_task[tidx][c]
            bucket_cursors[tidx] = c + 1
            if ep in picked:
                continue
            picked.add(ep)
            progress = True
        if not progress:
            break

    distinct_eps = sorted(picked)
    # Distribute num_samples frames across the picked episodes round-robin.
    counts = {ep: 0 for ep in distinct_eps}
    if num_samples <= len(distinct_eps):
        for ep in distinct_eps[:num_samples]:
            counts[ep] = 1
    else:
        i = 0
        while sum(counts.values()) < num_samples:
            counts[distinct_eps[i % len(distinct_eps)]] += 1
            i += 1

    plan: list[tuple[int, int]] = []
    for ep in distinct_eps:
        k = counts[ep]
        if k <= 0:
            continue
        ep_len = episode_lengths[ep]
        lo = skip_leading
        hi = ep_len - skip_trailing
        if hi <= lo:
            continue
        if k == 1:
            # Single frame from this episode: pick a random position in
            # the usable range. Avoids the "every sample is frame 50"
            # degeneracy of evenly-spaced-with-k=1.
            plan.append((ep, int(rng.integers(lo, hi))))
        else:
            # Multi-frame: evenly-spaced anchors + small jitter
            # (up to 1/8 of the spacing each side) so identical episodes
            # don't always sample the same offsets.
            anchors = np.linspace(lo, hi - 1, k)
            jitter_max = max(1.0, (hi - lo) / (8.0 * k))
            jitter = rng.uniform(-jitter_max, jitter_max, size=k)
            offsets = np.clip(anchors + jitter, lo, hi - 1).astype(np.int64)
            for off in offsets:
                plan.append((ep, int(off)))
    return plan[:num_samples]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="LeRobot v2.1 dataset root (contains meta/, data/, videos/). "
             "Default for OpenArm v4: "
             "~/.cache/huggingface/lerobot/local/openarm-teleop-16dof-v4")
    parser.add_argument(
        "--num-samples", type=int, default=80,
        help="Target calibration sample count (50-100 recommended per plan)")
    parser.add_argument(
        "--skip-leading", type=int, default=50,
        help="Frames to skip at episode start (default 50 = 1 s at 50 Hz, "
             "past the auto-exposure / start-of-recording window)")
    parser.add_argument(
        "--skip-trailing", type=int, default=25,
        help="Frames to skip at episode end (default 25 = 0.5 s; many "
             "episodes record a brief 'task complete' pause that "
             "produces near-zero action samples)")
    parser.add_argument(
        "--cams", default="ego,left_wrist,right_wrist",
        help="Comma-separated LeRobot camera keys "
             "(observation.images.<KEY>.mp4) to extract. The script writes "
             "the first 3 into images_ego / images_left / images_right.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path,
                        help="Output npz path")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.dataset_dir = args.dataset_dir.expanduser()
    args.output = args.output.expanduser()

    if not args.dataset_dir.is_dir():
        print(f"ERROR: {args.dataset_dir} is not a directory", file=sys.stderr)
        return 1
    cams = args.cams.split(",")
    if len(cams) < 1:
        print("ERROR: at least one camera required", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading dataset metadata: {args.dataset_dir}")
    meta = _load_dataset_meta(args.dataset_dir)
    info = meta["info"]
    tasks = meta["tasks"]
    episodes = meta["episodes"]
    n_episodes = len(episodes)
    episode_lengths = [int(e["length"]) for e in episodes]
    print(f"  {n_episodes} episodes, {sum(episode_lengths)} total frames, "
          f"{len(tasks)} tasks, {info['fps']} Hz, cams={cams}")

    # Look up each episode's task_index from its first frame in the
    # parquet manifest (episodes.jsonl encodes the prompt string but
    # not the task_index; the parquet does). Reading episode-0-frame-0
    # is cheap; we only do it for sampling stratification.
    import pyarrow.parquet as pq
    episode_tasks: list[int] = []
    for ep in range(n_episodes):
        parquet_path, _, _ = _episode_paths(args.dataset_dir, info, ep)
        if not parquet_path.is_file():
            episode_tasks.append(-1)
            continue
        try:
            t = pq.read_table(parquet_path, columns=["task_index"])
            episode_tasks.append(int(t.column("task_index")[0].as_py()))
        except Exception:
            episode_tasks.append(-1)

    rng = np.random.default_rng(args.seed)
    plan = _sample_plan(
        args.num_samples,
        skip_leading=args.skip_leading,
        skip_trailing=args.skip_trailing,
        rng=rng,
        episode_lengths=episode_lengths,
        episode_tasks=episode_tasks,
    )
    print(f"Sample plan: {len(plan)} (episode, frame_idx) pairs across "
          f"{len(set(ep for ep, _ in plan))} episodes "
          f"({len(set(episode_tasks[ep] for ep, _ in plan))} tasks)")

    # Lazy import: pyarrow (always needed) + av (lazy in _decode_frame).
    import pyarrow.parquet as pq

    imgs_per_cam: list[list[np.ndarray]] = [[] for _ in cams]
    states: list[np.ndarray] = []
    prompts: list[str] = []
    ep_ids: list[int] = []
    fr_ids: list[int] = []
    task_ids: list[int] = []

    # Group plan by episode so we read each parquet exactly once.
    by_ep: dict[int, list[int]] = {}
    for ep, fidx in plan:
        by_ep.setdefault(ep, []).append(fidx)

    for ep, fidxs in sorted(by_ep.items()):
        parquet_path, ep_stem, chunk_name = _episode_paths(args.dataset_dir, info, ep)
        if not parquet_path.is_file():
            print(f"  WARN: missing parquet for episode {ep}: {parquet_path}",
                  file=sys.stderr)
            continue
        table = pq.read_table(parquet_path)
        # Build a frame_index -> table_row map for fast lookup.
        frame_indices = np.asarray(table.column("frame_index").to_pylist())
        for fidx in fidxs:
            rows = np.where(frame_indices == fidx)[0]
            if len(rows) == 0:
                print(f"  WARN: frame_idx {fidx} not in episode {ep} parquet",
                      file=sys.stderr)
                continue
            row = int(rows[0])
            state = np.asarray(
                table.column("observation.state")[row].as_py(), dtype=np.float32)
            tidx = int(table.column("task_index")[row].as_py())
            prompt = tasks.get(tidx, "do the task")

            ok = True
            cam_frames: list[np.ndarray] = []
            for cam in cams:
                vp = (args.dataset_dir / "videos" / chunk_name /
                      f"observation.images.{cam}" / f"{ep_stem}.mp4")
                try:
                    frame = _decode_frame(vp, fidx)
                except Exception as e:  # pragma: no cover
                    print(f"  WARN: decode {vp} @ {fidx}: {type(e).__name__}: {e}",
                          file=sys.stderr)
                    ok = False
                    break
                cam_frames.append(_resize_224(frame))
            if not ok:
                continue

            for k, frame in enumerate(cam_frames):
                imgs_per_cam[k].append(frame)
            states.append(state)
            prompts.append(prompt)
            ep_ids.append(ep)
            fr_ids.append(fidx)
            task_ids.append(tidx)

            if args.verbose:
                print(f"  ep={ep} frame={fidx} task={tidx!r} "
                      f"prompt={prompt[:40]!r}")

    n = len(states)
    if n == 0:
        print("ERROR: no calibration samples collected", file=sys.stderr)
        return 1
    print(f"\nCollected {n} samples across {len(set(ep_ids))} episodes, "
          f"{len(set(task_ids))} tasks")

    arrays: dict[str, np.ndarray] = {
        "state": np.stack(states).astype(np.float32),
        "prompts": np.asarray(prompts, dtype=object),
        "episodes": np.asarray(ep_ids, dtype=np.int32),
        "frame_idx": np.asarray(fr_ids, dtype=np.int32),
        "task_idx": np.asarray(task_ids, dtype=np.int32),
    }
    cam_keys = ("images_ego", "images_left", "images_right")
    for k, key in enumerate(cam_keys[:len(cams)]):
        arrays[key] = np.stack(imgs_per_cam[k]).astype(np.uint8)

    np.savez(args.output, **arrays)
    size_mb = args.output.stat().st_size / 1e6
    print(f"Wrote {args.output}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
