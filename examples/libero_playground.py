#!/usr/bin/env python3
"""Interactive LIBERO playground for FlashRT + Pi0.5.

Opens a live MuJoCo viewer window with a LIBERO scene + Franka arm. You
type natural-language prompts into the terminal; the policy executes them
in real time and you watch the arm move. Switch between four chunk
execution modes on the fly to compare smoothness:

  1 — sync truncate-replan       (the canonical LIBERO eval pattern: predict
                                   10, execute 5, discard the rest, replan.
                                   Robot pauses for each inference.)
  2 — async pipelined, no blend   (AsyncChunkRunner. Background inference
                                   overlaps with execution. Hard swap at
                                   chunk seams. Robot never stalls.)
  3 — async + tail blend = 3      (Same as 2, but linearly blend the last 3
                                   actions of each chunk with the previous
                                   served action when inference deadline
                                   misses. Smoother on stalls.)
  4 — async + tail blend = 5      (Same with a 5-step blend window.)

The async modes use `flash_rt/runtime/rtc.py::AsyncChunkRunner`, which is
upstream FlashRT code. The truncate-replan mode mirrors what
`examples/thor/eval_libero.py::run_episode` does, so visual comparison is
honest.

Setup
-----
Requires the LIBERO sim stack installed in this venv::

    uv pip install "robosuite==1.4.1" mujoco bddl easydict gym \\
        robomimic hydra-core cloudpickle einops future opencv-python-headless
    # Then libero from the openpi sibling checkout:
    uv pip install -e /path/to/openpi/third_party/libero

Pi0.5 + JAX + FlashRT install per the Spark runbook.

Usage
-----
::

    PYTHONPATH=/path/to/openpi/third_party/libero \\
    python examples/libero_playground.py \\
        --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero \\
        --suite libero_object --task 0

Then type at the terminal::

    > pick up the milk and place it in the basket
    > 2                                  # switch to async pipelined
    > pick up the ketchup and place it in the basket
    > 3                                  # switch to async + blend=3
    > r                                  # reset env to initial state
    > q                                  # quit

The playground accepts ANY English prompt — Pi0.5 will tokenize it and
try its best. Out-of-distribution prompts are interesting: you can ask
"pick the red one", "stack them", etc., and watch generalization in
action. Objects not present in the scene cause the arm to reach for
plausible-but-wrong locations; that's by design.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import pathlib
import queue
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


# ──────────────────────────────────────────────────────────────────────
# EGL cleanup patches — must be applied BEFORE any robosuite import.
# Spark = unified memory (Grace + GB10); same risk as Jetson with EGL
# release races into CUDA memory. Mirrors what eval_libero.py does.
# ──────────────────────────────────────────────────────────────────────
def _patch_egl_cleanup() -> None:
    try:
        import robosuite.renderers.context.egl_context as _egl  # type: ignore
        _egl.EGLGLContext.free = lambda self: None
        _egl.EGLGLContext.__del__ = lambda self: None
    except Exception:
        pass
    try:
        import robosuite.utils.binding_utils as _bu  # type: ignore
        _bu.MjRenderContext.__del__ = lambda self: None
    except Exception:
        pass


_patch_egl_cleanup()


# ──────────────────────────────────────────────────────────────────────
# PyTorch 2.6+ defaulted `torch.load(weights_only=True)`. libero's
# init-state files are numpy-array pickles created against pre-2.6 torch
# and fail under the new default. Monkey-patch to keep `weights_only=False`
# (the legacy behaviour) for this script. The libero init-state files
# are trusted (they ship with the libero source).
# ──────────────────────────────────────────────────────────────────────
def _patch_torch_load() -> None:
    try:
        import torch  # type: ignore
        if getattr(torch.load, "_playground_patched", False):
            return
        _orig = torch.load

        def _patched(*a, **kw):
            kw.setdefault("weights_only", False)
            return _orig(*a, **kw)

        _patched._playground_patched = True  # type: ignore[attr-defined]
        torch.load = _patched
    except Exception:
        pass


_patch_torch_load()

import cv2  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer as mv  # noqa: E402

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402


logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_log = logging.getLogger("playground")


# Pi0.5 LIBERO defaults
CHUNK_SIZE = 10
ACTION_DIM = 7
LIBERO_RES = 256
POLICY_RES = 224
DUMMY_ACTION = np.array([0.0] * 6 + [-1.0])


# ──────────────────────────────────────────────────────────────────────
# Observation processing — matches examples/thor/eval_libero.py
# ──────────────────────────────────────────────────────────────────────
def _resize_with_pad(img: np.ndarray, h: int, w: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(h / ih, w / iw)
    nh, nw = int(ih * s), int(iw * s)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), dtype=img.dtype)
    ph, pw = (h - nh) // 2, (w - nw) // 2
    out[ph:ph + nh, pw:pw + nw] = resized
    return out


def _policy_inputs_from_obs(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert env obs → (agentview, wrist) at the policy's 224x224 input size."""
    # robosuite renders cameras upside-down + mirrored relative to the
    # convention Pi0.5 was trained on. eval_libero.py does the same flip.
    a = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    w = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return _resize_with_pad(a, POLICY_RES, POLICY_RES), _resize_with_pad(w, POLICY_RES, POLICY_RES)


# ──────────────────────────────────────────────────────────────────────
# Chunk execution modes
# ──────────────────────────────────────────────────────────────────────
class ChunkMode(Protocol):
    name: str
    def set_prompt(self, prompt: str) -> None: ...
    def reset(self) -> None: ...
    def next_action(self, obs: dict) -> np.ndarray: ...
    def stats(self) -> str: ...
    def close(self) -> None: ...


class TruncateReplanMode:
    """Sync truncate-and-replan, matching examples/thor/eval_libero.py."""

    def __init__(self, model: Any, replan_steps: int = 5) -> None:
        self.name = f"sync truncate-replan (k={replan_steps})"
        self._model = model
        self._k = replan_steps
        self._queue: collections.deque[np.ndarray] = collections.deque()
        self._prompt = ""
        self._n_chunks = 0
        self._n_actions = 0
        self._last_latency_ms = 0.0

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        self._queue.clear()  # invalidate old plan

    def reset(self) -> None:
        self._queue.clear()
        self._n_chunks = 0
        self._n_actions = 0

    def next_action(self, obs: dict) -> np.ndarray:
        if not self._queue:
            agent, wrist = _policy_inputs_from_obs(obs)
            t0 = time.perf_counter()
            actions = self._model.predict(images=[agent, wrist], prompt=self._prompt)
            self._last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self._n_chunks += 1
            self._queue.extend(np.asarray(actions[:self._k], dtype=np.float64))
        self._n_actions += 1
        return self._queue.popleft()

    def stats(self) -> str:
        return (f"chunks={self._n_chunks} actions={self._n_actions} "
                f"last_infer={self._last_latency_ms:.1f}ms")

    def close(self) -> None:
        pass


class AsyncBlendMode:
    """Async pipelined via flash_rt.runtime.rtc.AsyncChunkRunner."""

    def __init__(self, model: Any, blend_steps: int = 0) -> None:
        from flash_rt.runtime.rtc import AsyncChunkRunner, CallablePolicyAdapter, RTCConfig

        self.name = f"async pipelined (blend={blend_steps})"
        self._model = model
        self._prompt = ""
        self._blend_steps = blend_steps

        def _infer(observation: dict) -> np.ndarray:
            agent, wrist = _policy_inputs_from_obs(observation)
            return np.asarray(
                model.predict(images=[agent, wrist], prompt=self._prompt),
                dtype=np.float64,
            )

        adapter = CallablePolicyAdapter(fn=_infer, output_key=None)
        cfg = RTCConfig(
            target_hz=20.0,
            action_horizon=CHUNK_SIZE,
            start_next_at=CHUNK_SIZE // 2,
            miss_policy="hold_last",
            blend_steps=blend_steps,
        )
        self._runner = AsyncChunkRunner(adapter, cfg)

    def set_prompt(self, prompt: str) -> None:
        self._prompt = prompt
        # AsyncChunkRunner has no in-flight invalidation hook; let the
        # current chunk drain. The next chunk will be generated against
        # the new prompt automatically. (For a hard cut, call reset()
        # after this; see the "p" command handler.)

    def reset(self) -> None:
        # AsyncChunkRunner.reset takes an observation. We do it lazily
        # on the next next_action call by closing + reinstantiating.
        self.close()
        self.__init__(self._model, blend_steps=self._blend_steps)
        self._prompt = self._prompt or ""

    def next_action(self, obs: dict) -> np.ndarray:
        return self._runner.next_action(obs)

    def stats(self) -> str:
        s = self._runner.stats
        return (f"chunks={s.chunks_completed} actions={s.actions_served} "
                f"swaps={s.swaps} held={s.held_actions} "
                f"last_infer={s.last_latency_s * 1000:.1f}ms "
                f"misses={s.deadline_misses}")

    def close(self) -> None:
        self._runner.close()


MODE_FACTORIES = {
    "1": ("sync truncate-replan (k=5)", lambda m: TruncateReplanMode(m, replan_steps=5)),
    "2": ("async pipelined, no blend",  lambda m: AsyncBlendMode(m, blend_steps=0)),
    "3": ("async + tail blend = 3",     lambda m: AsyncBlendMode(m, blend_steps=3)),
    "4": ("async + tail blend = 5",     lambda m: AsyncBlendMode(m, blend_steps=5)),
}


HELP = """
Commands:
  <any prompt>     send a new natural-language prompt to the policy
  1 / 2 / 3 / 4    switch chunk execution mode (see header for descriptions)
  r                reset env to the current task's initial state
  t <n>            change task within the current suite (0-indexed)
  s                print current stats snapshot
  h                print this help
  q                quit
"""


# ──────────────────────────────────────────────────────────────────────
# stdin reader thread — drops lines into a queue without blocking the loop
# ──────────────────────────────────────────────────────────────────────
def _stdin_reader(out: queue.Queue) -> None:
    try:
        for line in sys.stdin:
            out.put(line.rstrip("\n"))
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main() -> int:
    parser = argparse.ArgumentParser(description="Live LIBERO playground for FlashRT + Pi0.5")
    parser.add_argument("--checkpoint", required=True,
                        help="Pi0.5 Orbax JAX checkpoint dir (e.g. ~/.cache/openpi/openpi-assets/checkpoints/pi05_libero)")
    parser.add_argument("--suite", default="libero_object",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"])
    parser.add_argument("--task", type=int, default=0, help="Task index within the suite (0..N-1)")
    parser.add_argument("--mode", default="2", choices=list(MODE_FACTORIES),
                        help="Initial chunk execution mode (default 2 = async pipelined no blend)")
    parser.add_argument("--target-hz", type=float, default=20.0,
                        help="Control loop rate (default 20 Hz, the LIBERO sim rate)")
    parser.add_argument("--autotune", type=int, default=3,
                        help="FlashRT CUDA Graph autotune trials")
    args = parser.parse_args()

    period = 1.0 / args.target_hz

    # ── load benchmark + task ─────────────────────────────────────────
    bench = benchmark.get_benchmark_dict()
    suite = bench[args.suite]()
    n_tasks = suite.n_tasks
    if not 0 <= args.task < n_tasks:
        print(f"error: --task {args.task} out of range [0, {n_tasks})", file=sys.stderr)
        return 2
    task = suite.get_task(args.task)
    task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_states = suite.get_task_init_states(args.task)

    print("=" * 70)
    print(f"  Suite:      {args.suite}  ({n_tasks} tasks)")
    print(f"  Task {args.task}:   {task.language!r}")
    print(f"  Init states: {len(init_states)}")
    print(f"  BDDL:       {task_bddl}")
    print("=" * 70)

    # ── load policy ───────────────────────────────────────────────────
    print(f"\nLoading FlashRT Pi0.5 from {args.checkpoint} ...")
    import flash_rt
    t0 = time.perf_counter()
    model = flash_rt.load_model(
        checkpoint=args.checkpoint, framework="jax",
        num_views=2, autotune=args.autotune,
    )
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    # ── build env ─────────────────────────────────────────────────────
    print(f"\nBuilding LIBERO env ...")
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl),
        camera_heights=LIBERO_RES, camera_widths=LIBERO_RES,
    )
    obs = env.reset()
    obs = env.set_init_state(init_states[0])
    # Spin idle for a few sim frames to let physics settle
    for _ in range(10):
        obs, _, _, _ = env.step(DUMMY_ACTION)

    # ── pull MuJoCo handles for the viewer ────────────────────────────
    mj_sim = env.env.sim
    mj_model = mj_sim.model._model
    mj_data = mj_sim.data._data
    print(f"  scene: {mj_model.nbody} bodies, {mj_model.ngeom} geoms")

    # ── instantiate initial mode + set initial prompt ─────────────────
    current_prompt = task.language
    mode_name, mode_factory = MODE_FACTORIES[args.mode]
    mode: ChunkMode = mode_factory(model)
    mode.set_prompt(current_prompt)
    print(f"\nInitial mode: [{args.mode}] {mode_name}")
    print(f"Initial prompt: {current_prompt!r}")

    # ── open viewer + stdin reader ────────────────────────────────────
    print("\nLaunching MuJoCo viewer ...")
    viewer = mv.launch_passive(mj_model, mj_data)
    print(HELP)
    cmd_q: queue.Queue = queue.Queue()
    threading.Thread(target=_stdin_reader, args=(cmd_q,), daemon=True).start()

    # ── main loop ─────────────────────────────────────────────────────
    step_counter = 0
    last_stats_print = time.perf_counter()
    try:
        while viewer.is_running():
            tick_start = time.perf_counter()

            # ---- command queue (non-blocking drain) ----
            try:
                while True:
                    cmd = cmd_q.get_nowait()
                    if not cmd:
                        continue
                    elif cmd == "q":
                        print("[quit]")
                        raise KeyboardInterrupt
                    elif cmd == "h":
                        print(HELP)
                    elif cmd == "s":
                        print(f"[stats] mode='{mode.name}' prompt={current_prompt!r} "
                              f"step={step_counter} {mode.stats()}")
                    elif cmd == "r":
                        env.reset()
                        env.set_init_state(init_states[0])
                        for _ in range(10):
                            obs, _, _, _ = env.step(DUMMY_ACTION)
                        mode.reset()
                        mode.set_prompt(current_prompt)
                        step_counter = 0
                        print(f"[reset] task={args.task} initial_state[0]; prompt={current_prompt!r}")
                    elif cmd.startswith("t "):
                        try:
                            new_tid = int(cmd.split()[1])
                        except (IndexError, ValueError):
                            print("[err] usage: t <task_index>")
                            continue
                        if not 0 <= new_tid < n_tasks:
                            print(f"[err] task {new_tid} out of range [0, {n_tasks})")
                            continue
                        # Rebuild env for the new task (different BDDL = different scene)
                        env.close()
                        args.task = new_tid
                        task = suite.get_task(new_tid)
                        task_bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
                        init_states = suite.get_task_init_states(new_tid)
                        env = OffScreenRenderEnv(
                            bddl_file_name=str(task_bddl),
                            camera_heights=LIBERO_RES, camera_widths=LIBERO_RES,
                        )
                        env.reset()
                        obs = env.set_init_state(init_states[0])
                        for _ in range(10):
                            obs, _, _, _ = env.step(DUMMY_ACTION)
                        mj_sim = env.env.sim
                        mj_model = mj_sim.model._model
                        mj_data = mj_sim.data._data
                        viewer.close()
                        viewer = mv.launch_passive(mj_model, mj_data)
                        current_prompt = task.language
                        mode.reset()
                        mode.set_prompt(current_prompt)
                        step_counter = 0
                        print(f"[task] {new_tid}: {task.language!r}")
                    elif cmd in MODE_FACTORIES:
                        new_name, new_factory = MODE_FACTORIES[cmd]
                        mode.close()
                        mode = new_factory(model)
                        mode.set_prompt(current_prompt)
                        print(f"[mode] [{cmd}] {new_name}")
                    else:
                        current_prompt = cmd
                        mode.set_prompt(current_prompt)
                        print(f"[prompt] {current_prompt!r}")
            except queue.Empty:
                pass

            # ---- one control tick ----
            action = mode.next_action(obs)
            action_list = action.tolist() if hasattr(action, "tolist") else list(action)
            obs, reward, done, info = env.step(action_list)
            viewer.sync()
            step_counter += 1

            if done:
                print(f"[goal!] step={step_counter} reward={reward:.2f}  "
                      f"(env will keep running; press 'r' to reset)")

            # Periodic stats line every 5s
            now = time.perf_counter()
            if now - last_stats_print > 5.0:
                print(f"[periodic] step={step_counter} {mode.stats()}")
                last_stats_print = now

            # Throttle to target_hz
            elapsed = time.perf_counter() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down ...")
        try:
            mode.close()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            viewer.close()
        except Exception:
            pass
        # mujoco/glfw cleanup sometimes segfaults on shutdown; skip the
        # interpreter's normal teardown to avoid a noisy exit.
        os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
