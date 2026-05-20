"""Phase 3 part 1 — prepare a stratified calibration sample set from the
OpenArm LeRobot dataset.

Why stratified: FP8 calibration computes per-tensor activation amax. A
biased sample (e.g. only one task, or only the first frame of each
episode) under-estimates amax and produces scales that clip real
inference observations. The recipe per the plan is "50-100 stratified
OpenArm observations".

Stratification:
  - Sample uniformly across episodes (no episode dominates).
  - Within each episode, take K evenly-spaced frames (not just the
    start, where the robot is at rest and pixel statistics are
    degenerate). K is chosen so that total samples ≈ --num-samples.
  - Skip the first 5 frames of each episode (typical "wait" period
    where the robot has not started moving and the cameras may still
    be auto-exposing).

Output: a single npz with arrays
  - images       (N, 224, 224, 3) uint8 — base camera
  - wrist_images (N, 224, 224, 3) uint8 — wrist camera
  - prompts      (N,) object-array of unicode strings
  - episodes     (N,) int32 — episode index each frame came from
  - frame_idx    (N,) int32 — frame-within-episode

The runner script (spark_phase3_run_calib.py) reads this npz and feeds
the frames into ``flash_rt.VLAModel.calibrate(observations, percentile=99.9)``.

Usage:
    python3 scripts/spark_phase3_prepare_calib.py \\
        --repo-id local/openarm-teleop-16dof \\
        --num-samples 80 \\
        --output /openpi_assets/calib_openarm_80.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def resize_to_224(img: np.ndarray) -> np.ndarray:
    """Resize HxWx3 uint8 to 224x224, letterboxed."""
    import cv2  # imported lazily; openpi serve_policy_ngc image has it

    h, w = img.shape[:2]
    scale = 224.0 / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    out = np.zeros((224, 224, 3), dtype=np.uint8)
    pad_h = (224 - new_h) // 2
    pad_w = (224 - new_w) // 2
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="local/openarm-teleop-16dof",
                        help="LeRobot dataset repo id")
    parser.add_argument("--num-samples", type=int, default=80,
                        help="Target number of calibration samples (50-100 recommended)")
    parser.add_argument("--output", required=True, help="Output npz path")
    parser.add_argument("--skip-leading", type=int, default=5,
                        help="Frames to skip at start of each episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-key", default="image",
                        help="Observation key for base camera (varies by dataset)")
    parser.add_argument("--wrist-key", default="wrist_image",
                        help="Observation key for wrist camera")
    parser.add_argument("--prompt-key", default="task",
                        help="Observation key for task prompt (LeRobot stores as 'task')")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Lazy import — heavy dependency, only needed for actual data load.
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("ERROR: lerobot not installed. Install with `pip install lerobot`, "
              "or run this script inside any container that ships lerobot "
              "(e.g. openpi's openpi_server_ngc training image).",
              file=sys.stderr)
        return 1

    print(f"Loading LeRobot dataset: {args.repo_id}")
    ds = LeRobotDataset(args.repo_id)

    # Episode indexing — LeRobotDataset has `episode_data_index` which gives
    # (from, to) frame ranges per episode.
    edi = ds.episode_data_index
    n_episodes = len(edi["from"])
    print(f"Total episodes: {n_episodes}, total frames: {len(ds)}")

    # How many samples per episode?
    rng = np.random.default_rng(args.seed)
    target_per_ep = max(1, args.num_samples // n_episodes)
    # If we have very few episodes, sample more from each.
    if target_per_ep * n_episodes < args.num_samples:
        # Top up by sampling additional episodes
        extra = args.num_samples - target_per_ep * n_episodes
        extra_eps = set(rng.choice(n_episodes, size=extra, replace=False).tolist())
    else:
        extra_eps = set()

    images: list[np.ndarray] = []
    wrists: list[np.ndarray] = []
    prompts: list[str] = []
    ep_ids: list[int] = []
    frame_ids: list[int] = []

    for ep in range(n_episodes):
        ep_from = int(edi["from"][ep].item())
        ep_to = int(edi["to"][ep].item())
        ep_len = ep_to - ep_from
        if ep_len <= args.skip_leading:
            continue
        start = ep_from + args.skip_leading
        usable_len = ep_to - start

        k_this = target_per_ep + (1 if ep in extra_eps else 0)
        k_this = min(k_this, usable_len)
        if k_this == 0:
            continue
        # Evenly-spaced indices within this episode.
        offsets = np.linspace(0, usable_len - 1, k_this).astype(np.int64)
        indices = (start + offsets).tolist()

        for idx in indices:
            frame: dict[str, Any] = ds[idx]
            # LeRobot stores images as torch tensors (C, H, W) float ∈ [0, 1].
            # Convert to (H, W, C) uint8.
            for key, dest in ((args.image_key, images),
                              (args.wrist_key, wrists)):
                if key not in frame:
                    print(f"WARN: key '{key}' missing in frame {idx}; available: {list(frame.keys())}",
                          file=sys.stderr)
                    return 2
                img_t = frame[key]
                if hasattr(img_t, "numpy"):
                    img = img_t.numpy()
                else:
                    img = np.asarray(img_t)
                if img.ndim == 3 and img.shape[0] in (1, 3):
                    img = img.transpose(1, 2, 0)
                if img.dtype != np.uint8:
                    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
                if img.shape[:2] != (224, 224):
                    img = resize_to_224(img)
                dest.append(img)

            prompt = frame.get(args.prompt_key, "")
            if hasattr(prompt, "item"):
                prompt = prompt.item()
            prompts.append(str(prompt))
            ep_ids.append(ep)
            frame_ids.append(idx - ep_from)

            if len(images) >= args.num_samples:
                break
        if len(images) >= args.num_samples:
            break

    if len(images) == 0:
        print("ERROR: no calibration samples collected", file=sys.stderr)
        return 1

    print(f"Collected {len(images)} samples across {len(set(ep_ids))} episodes")
    np.savez(
        out,
        images=np.stack(images),
        wrist_images=np.stack(wrists),
        prompts=np.asarray(prompts, dtype=object),
        episodes=np.asarray(ep_ids, dtype=np.int32),
        frame_idx=np.asarray(frame_ids, dtype=np.int32),
    )
    print(f"Wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
