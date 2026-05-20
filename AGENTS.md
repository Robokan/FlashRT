# AGENTS.md — durable working rules for FlashRT agents

This file is auto-attached to every Cursor agent conversation in this
repository. It encodes context and constraints that should hold across
sessions so the next agent doesn't have to relearn them. Read it first.

## What this repo is and what we're doing here

FlashRT is a hand-written CUDA inference engine for small-batch,
latency-sensitive VLA models (Pi0, Pi0.5, Pi0-FAST, GROOT N1.6/N1.7).

The work on branch `spark-sm121-port` (Robokan/FlashRT fork) is porting
the FlashRT JAX inference path to **NVIDIA DGX Spark** (GB10, SM_121,
aarch64 Grace+Blackwell, 121 GB unified LPDDR). The goal is to swap
the existing openpi JAX server for FlashRT as the inference engine
for `pi05_openarm_ngc_lora_v4`, end-to-end, without changing the robot
client.

The plan lives at:
```
~/.cursor/plans/flashrt_jax_on_dgx_spark_f0288dd4.plan.md
```
Status of the seven phases is in `docs/spark_status.md`; the full
operational runbook is in `docs/spark_runbook.md`.

## Repository layout assumption

```
~/sparkpack/
├── FlashRT/      ← this repo (the workspace root)
└── openpi/       ← read-only sibling. Physical-Intelligence upstream.
                    Robokan/openpi fork is kept rebase-clean; do NOT
                    add FlashRT-related code here.
```

Anything FlashRT-related goes in this repo. Period. The earlier
Spark-on-FlashRT work was split between FlashRT and openpi forks; that
caused fork divergence and was consolidated into FlashRT in commit
`dc00926`. Re-introducing that split is a regression.

## Hardware specifics — DGX Spark (Grace + GB10)

- aarch64 host; nvcc 13.0; CUDA Toolkit 13.0.88 at `/usr/local/cuda`.
- GPU: NVIDIA GB10, SM_121 (`-gencode arch=compute_121a,code=sm_121a`).
- 121 GB unified LPDDR pool — shared between Grace CPU (desktop +
  Cursor + JAX caches) and the GPU. There is no separate VRAM.
- Driver 580.142 series.
- 20-core Grace CPU.
- Typical free RAM with the desktop + Cursor running: ~90 GB.
- The Spark is the *workstation*, not a remote box. The OOM killer
  hits the heaviest CPU process (the IDE) before it hits a stuck
  CUDA compile.

## Build constraints (read before running cmake)

The Spark build compiles a heavier translation-unit set than the Thor
build — 15 CUTLASS-3.x FA2 instantiations at `sm_80 + sm_120 + sm_121`,
plus the SM120 NVFP4 W4A16 GEMM and SM120 block-128 FP8 GEMM. Each of
those template TUs peaks at ~5 GB of host RAM during nvcc template
instantiation; regular hand-written kernels peak at ~1–2 GB.

**`cmake --build build -j$(nproc)` (= -j20) has crashed this box.**
The OOM killer took down the desktop and forced a hard reboot. Do not
do this on a workstation Spark.

Rule of thumb on Spark:
```
BUILD_J = max(2, floor((free_GB_at_start - 12) / 6))

30 GB free → 3      60 GB free → 8      115 GB free → 17
```

The defaults are:
- `docker/Dockerfile.spark` → `ARG BUILD_J=4`
- `docker/compose.spark.yml` → `BUILD_J: ${BUILD_J:-4}`

Check `free -h` before raising it. Headless Spark hosts can use
`BUILD_J=16` or `20`. Workstation Spark with Cursor running → keep at 4.

## Working-environment rules

1. **Use `uv` for any host-side Python work outside Docker.** Never
   `pip install --break-system-packages`. PEP 668 prevents naïve
   pip on Ubuntu's system Python; uv is already installed at
   `/usr/local/bin/uv`. The standard venv path is:
   ```bash
   cd ~/sparkpack/FlashRT
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install pybind11 ninja numpy pyyaml
   ```
2. **CUTLASS lives at `third_party/cutlass`** in the source tree
   (gitignored). The Dockerfile vendors it; native builds clone:
   ```bash
   git clone --depth 1 --branch v4.4.2 \
       https://github.com/NVIDIA/cutlass.git third_party/cutlass
   ```
3. **Don't pipe the cmake build through `tee` to a terminal.** Cursor's
   terminal mirror buffers every line; nvcc's CUTLASS-template warnings
   can blow past the buffer and lock the UI. Redirect to a file:
   ```bash
   nohup setsid cmake --build build -j4 > /tmp/flashrt_build.log 2>&1 &
   ```
4. **No `cmake --build` from the openpi workspace.** It'll work in the
   FlashRT workspace.

## Git layout

- Active branch: `spark-sm121-port` (this is where edits land).
- Remotes:
  - `origin` → `https://github.com/Robokan/FlashRT.git` (user fork; push here)
  - `upstream` → `https://github.com/LiangSu8899/FlashRT.git` (canonical; never push)
- Identity: `Robokan <eric.d.vaughan@gmail.com>`.
- The `third_party/cutlass/` clone is intentionally untracked; do not
  add it to git.
- Only create commits when the user asks for them.

## Cross-repo coupling

- `flash_rt/serving/openpi_adapter.py` imports `openpi_client.base_policy`
  (lightweight openpi-client package, installable separately from the
  full openpi server).
- `scripts/serve_policy_flashrt.py` lazy-imports
  `openpi.serving.websocket_policy_server` from the mounted openpi
  checkout.
- The flashrt_spark container in `docker/compose.spark.yml` mounts
  `~/sparkpack/openpi` at `/openpi` and adds `/openpi/src` and
  `/openpi/packages/openpi-client/src` to `PYTHONPATH`.

If you need to look at openpi code from this workspace, it's at
`/home/evaughan/sparkpack/openpi/` and is readable globally. Do not
edit it from here.

## Where to find the prior conversation

The conversation history that led to this codebase state is at:
```
/home/evaughan/.cursor/projects/home-evaughan-sparkpack-openpi/agent-transcripts/d6216242-2ae4-4778-92f0-643c93db89e5/d6216242-2ae4-4778-92f0-643c93db89e5.jsonl
```
(Existed because the workspace was openpi/ during the migration. After
switching to FlashRT as the workspace, new transcripts go to a
`home-evaughan-sparkpack-FlashRT` directory under `~/.cursor/projects`.)

Search this file for "Spark", "BUILD_J", "OOM", "uv", "compose.spark"
for the most relevant context.

## When to ask vs. do

- Reverting commits, force-pushing, or touching `git config`: ask first.
- Building C++/CUDA: confirm `free -h` first; `BUILD_J=4` is the
  default for a reason.
- Editing files in `openpi/`: don't.
- Anything that runs for > 5 minutes (docker pulls, cmake builds):
  detach into the background with `nohup setsid ... > /tmp/...log 2>&1 &`
  and poll the log; never tail-pipe long compiles through Cursor.
