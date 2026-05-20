# FlashRT JAX inference on DGX Spark — runbook

End-to-end checklist for running the existing `pi05_openarm_ngc_lora_v4`
Orbax checkpoint through FlashRT on DGX Spark, replacing the openpi JAX
server without touching the robot client.

- **Current status**: [`spark_status.md`](spark_status.md) — what's
  done vs. pending on Spark hardware right now.
- **Durable agent rules**: [`../AGENTS.md`](../AGENTS.md) — what every
  Cursor session in this repo should know up front.
- **Plan**: `~/.cursor/plans/flashrt_jax_on_dgx_spark_f0288dd4.plan.md`

---

## Layout

All Spark-specific code now lives in FlashRT. The openpi repo stays
unmodified — it's read-only on-disk infrastructure that this work
mounts (for the JAX server during parity, for `openpi-client` /
`openpi.serving`, and for the `lerobot` package during data prep).

```
~/sparkpack/
├── FlashRT/                                  # All Spark integration lives here
│   ├── CMakeLists.txt                        # SM_121 patches (Phase 0)
│   ├── docker/
│   │   ├── Dockerfile.spark                  # Grace+Blackwell build (Phase 0)
│   │   └── compose.spark.yml                 # flashrt_spark compose service
│   ├── flash_rt/
│   │   ├── frontends/jax/pi05_rtx.py         # LoRA merge in fp32 (Phase 2)
│   │   ├── frontends/torch/pi05_rtx.py       # robot_action_dim arg (Phase 4)
│   │   └── serving/openpi_adapter.py         # FlashRT VLAModel → openpi BasePolicy
│   ├── tests/test_lora_merge_jax_loader.py   # LoRA merge unit tests
│   ├── docs/spark_runbook.md                 # This file
│   └── scripts/
│       ├── spark_build_smoke.py              # Phase 0 gate
│       ├── spark_phase1_libero_smoke.py      # Phase 1 smoke (no LIBERO sim)
│       ├── spark_phase1_libero_run.sh        # Phase 1 full LIBERO eval
│       ├── spark_phase2_lora_load.py         # Phase 2 gate (LoRA merge)
│       ├── spark_phase3_prepare_calib.py     # Phase 3 part 1 (data prep)
│       ├── spark_phase3_run_calib.py         # Phase 3 part 2 (FP8 calibrate)
│       ├── spark_phase4_parity.py            # Phase 4 gate (vs JAX server)
│       ├── serve_policy_flashrt.py           # Phase 5 server
│       ├── spark_phase5_adapter_smoke.py     # Phase 5 wire test
│       ├── spark_phase6_robot_compare.py     # Phase 6 record + compare
│       └── spark_phase7_latency_breakdown.py # Phase 7 colocation decision
└── openpi/                                   # Unmodified upstream openpi
    ├── packages/openpi-client/               # mounted into the container for
    │                                         # WebsocketClientPolicy / BasePolicy
    ├── src/openpi/serving/                   # mounted in for WebsocketPolicyServer
    └── scripts/serve_policy.py               # used for the Phase 4 baseline
```

The `flashrt_spark` container mounts:
* `~/sparkpack/FlashRT`      → `/workspace/FlashRT`  (live source edits)
* `~/sparkpack/openpi`       → `/openpi`             (for `openpi.serving`, `openpi-client`, `lerobot`-equipped scripts)
* `~/.cache/openpi`          → `/openpi_assets`      (Orbax checkpoints)

---

## Phase 0 — Build FlashRT for SM_121 + aarch64

> **Memory caveat for the build step (read first).**
> The Spark build compiles a heavier TU set than the Thor build — FA2 in
> 15 CUTLASS-3.x hdim×dtype×split instantiations at sm_80 + sm_120 +
> sm_121, plus the SM120 NVFP4 W4A16 GEMM and SM120 block-128 FP8 GEMM.
> Each of those template TUs peaks at ~5 GB of host RAM during nvcc
> instantiation (regular kernels peak ~1–2 GB). On a headless Spark with
> ~115 GB free, `cmake --build -j$(nproc)` (= `-j20`) is fine. But on a
> Spark running a desktop + Cursor IDE + cached JAX runtimes
> (~30 GB already pinned), only ~90 GB is free at build start, and
> `-j16` × ~5 GB = a transient working set that has been observed to
> OOM-kill the desktop session and force a hard reboot.
>
> The unified LPDDR pool is shared between the Grace CPU and the
> Blackwell GPU, so when the OOM killer fires it targets the heaviest
> CPU consumer — usually the IDE or the desktop compositor, *not* the
> build itself. There is no helpful "build failed: out of memory"
> error; the whole UI just freezes.
>
> Quick rule of thumb on Spark:
>
> ```
> BUILD_J ≈ max(2, floor((free_GB_at_start - 12) / 6))
>
> 30 GB free → 3       60 GB free → 8       115 GB free → 17
> ```
>
> Check before you build: `free -h` and `nvidia-smi --query-gpu=memory.used --format=csv`.
>
> The Dockerfile + compose file default to `BUILD_J=4` so the desktop
> path is always safe. On a headless Spark, set `BUILD_J=16` or 20.
>
> (Thor uses `-j$(nproc)` in its Dockerfile and that works because
> Thor's CMakeLists path *skips* FA2 entirely on SM_110 — the build is
> 10–15 TUs of hand-written kernels, not the FA2 + CUTLASS sweep. The
> "Spark needs less `-j` than Thor" effect is about the *build's* memory
> footprint, not the unified-memory architecture.)

### Option A — Build via Docker (recommended, reproducible)

```bash
cd ~/sparkpack/FlashRT
# Optional override: BUILD_J=2 ... build flashrt_spark
docker compose -f docker/compose.spark.yml build flashrt_spark
```

The compose file caps the build container at 60 GB and passes
`BUILD_J=4` to the Dockerfile. First build is ~25–40 min (NGC base
image pull on cold cache + CUTLASS template codegen at low `-j`).

### Option B — Native uv venv (faster iteration on Spark, no Docker)

```bash
cd ~/sparkpack/FlashRT
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install pybind11 ninja numpy pyyaml

git clone --depth 1 --branch v4.4.2 \
    https://github.com/NVIDIA/cutlass.git third_party/cutlass

cmake -B build -S . -DGPU_ARCH=121 \
    -DPython3_EXECUTABLE=$(which python) -G Ninja

# Check free RAM and pick -j accordingly:
#   free -h
#   BUILD_J = max(2, floor((free_GB - 12) / 6))
# Safe default for a Spark with desktop + Cursor running:
cmake --build build -j4

# (Skip prlimit-based per-process caps: nvcc/cicc map large virtual
# regions for CUTLASS template specialization and a tight --as limit
# causes spurious "cannot allocate memory" failures even when resident
# usage is fine. The -j cap is the actual protection.)
```

Configure should print these lines if SM_121 is wired correctly:

```
-- NVFP4/W4A8 support: ENABLED (sm_121a)
-- Using gencode flag: -gencode=arch=compute_121a,code=sm_121a
-- FA2 in-SO attention: ENABLED (sm_121)
-- FA2 vendor arch: sm_80 + sm_120 + sm_121 AOT + compute_120 PTX fallback (Spark default)
```

The build drops `flash_rt/flash_rt_kernels*.so` and
`flash_rt/flash_rt_fa2*.so` into the source tree. Then make `flash_rt`
importable for the smoke test:

```bash
uv pip install -e ".[torch,jax]"
```

### Gate (either option)

Docker path:

```bash
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_build_smoke.py
```

Native path:

```bash
cd ~/sparkpack/FlashRT && source .venv/bin/activate
python scripts/spark_build_smoke.py
```

Exits 0 only if `get_gpu_sm_version() == 121`, `supports_nvfp4() == True`,
and a small fa2 bf16 kernel returns finite output.

All Phase 1+ commands assume `cd ~/sparkpack/FlashRT` and use the
in-FlashRT compose file. Scripts are addressed by their FlashRT path
since the container mounts FlashRT at `/workspace/FlashRT` (which is
the WORKDIR) and at `PYTHONPATH=/workspace/FlashRT`.

## Phase 1 — pi05_libero sanity

Download `pi05_libero` Orbax (~12 GB) to `/openpi_assets/`. Then:

```bash
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_phase1_libero_smoke.py \
        --checkpoint /openpi_assets/pi05_libero
```

Confirms the FlashRT JAX frontend loads + runs end-to-end with no LoRA
involvement. The full LIBERO eval (separate, requires libero/MuJoCo
installed) is launched via `scripts/spark_phase1_libero_run.sh`.

## Phase 2 — LoRA Orbax load

```bash
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_phase2_lora_load.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4
```

Verifies the new `_maybe_merge_lora` in
`flash_rt/frontends/jax/pi05_rtx.py` detects + fuses every LoRA pair
into base weights in fp32, before the existing bf16 truncation step.
Default scaling = 1.0 (matches openpi r=16/α=16 and r=32/α=32 LoRA
configs); override via `FLASHRT_LORA_SCALING` env var.

Unit tests:

```bash
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    pytest tests/test_lora_merge_jax_loader.py -v
```

## Phase 3 — FP8 calibration on OpenArm data

The data-prep step needs `lerobot` installed. The cleanest place to run
it is inside the openpi training container (which already ships
`lerobot`); the FlashRT image is intentionally minimal.

```bash
# Part 1: prepare stratified samples (in openpi training container).
cd ~/sparkpack/openpi && docker compose -f scripts/docker/compose_ngc.yml \
    run --rm openpi_server_ngc \
    python /app/../FlashRT/scripts/spark_phase3_prepare_calib.py \
        --repo-id local/openarm-teleop-16dof \
        --num-samples 80 \
        --output /openpi_assets/calib_openarm_80.npz

# Part 2: run FP8 calibration (in flashrt_spark container).
cd ~/sparkpack/FlashRT && docker compose -f docker/compose.spark.yml \
    run --rm flashrt_spark \
    python3 scripts/spark_phase3_run_calib.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --percentile 99.9
```

The prep script can also be run on bare metal anywhere `pip install lerobot`
works — no need for a container if you have it locally.

Writes the FP8 cache to `~/.flash_rt/calibration/<hash>_Se<N>.json`.

## Phase 4 — Parity vs the openpi JAX server

```bash
# Shell A: start the openpi JAX server (in the openpi compose).
cd ~/sparkpack/openpi
docker compose -f scripts/docker/compose_ngc.yml run --rm \
    -p 8001:8001 openpi_serve \
    python scripts/serve_policy.py policy:checkpoint \
        --policy.config=pi05_openarm_ngc_lora \
        --policy.dir=/openpi_assets/pi05_openarm_ngc_lora_v4 \
        --port=8001

# Shell B: run the parity harness from the FlashRT compose.
cd ~/sparkpack/FlashRT
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_phase4_parity.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --jax-server localhost:8001 \
        --robot-action-dim 16
```

Acceptance: post-unnorm cosine ≥ 0.999, ratio ∈ [0.995, 1.005] on every
sample. The `robot_action_dim=16` arg routes through the new
`FLASHRT_ROBOT_ACTION_DIM` knob and the three patched sites in
`pi05_rtx.py`.

## Phase 5 — Serve FlashRT over the openpi websocket protocol

The server script imports `openpi.serving.WebsocketPolicyServer` from
the mounted openpi checkout (`/openpi/src` in `PYTHONPATH`).

```bash
cd ~/sparkpack/FlashRT
docker compose -f docker/compose.spark.yml run --rm \
    -p 8001:8001 flashrt_spark \
    python3 scripts/serve_policy_flashrt.py \
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \
        --robot-action-dim 16 \
        --calib-data /openpi_assets/calib_openarm_80.npz \
        --default-prompt "pick up the red block" \
        --port 8001
```

Wire test (in another shell — only needs `openpi-client`, which the
flashrt_spark container has via the mounted openpi checkout):

```bash
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_phase5_adapter_smoke.py --port 8001
```

This confirms the `WebsocketClientPolicy` (which `AsyncActionChunkBroker`
wraps internally) gets back the same response shape the JAX server
produces — `actions`, `policy_timing.infer_ms`,
`_rtc_chunk_model_space`.

## Phase 6 — Real robot comparison

Run the same robot client against the JAX server, then against the
FlashRT server, recording outcomes via:

```bash
python3 scripts/spark_phase6_robot_compare.py append \
    --log jax_baseline.json --server jax --task pick_red_block \
    --success true --episode-steps 87
```

When both logs have enough episodes (≥ 20 per task is the rule of thumb):

```bash
python3 scripts/spark_phase6_robot_compare.py compare \
    --baseline jax_baseline.json --candidate flashrt_spark.json
```

Acceptance: overall ratio ≥ 0.95.

## Phase 7 — Colocation decision

With the FlashRT server running (Phase 5):

```bash
cd ~/sparkpack/FlashRT
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_phase7_latency_breakdown.py --port 8001
```

The script prints a per-component table and emits a recommendation:

- If JPEG decode > 10% of inference → switch the robot client to send
  raw uint8 arrays (`obs["image"] = np.uint8(...)`). The server already
  passes raw arrays through unchanged; only the client side changes.
- If client-side overhead (decode + msgpack + ws) > 50% of inference →
  investigate colocation. Otherwise the websocket boundary is free.

---

## Quick reference — environment vars

| var | effect |
|---|---|
| `FLASHRT_LORA_SCALING` | LoRA scaling factor passed to `_maybe_merge_lora` (default 1.0) |
| `FLASHRT_ROBOT_ACTION_DIM` | Robot DOF slice (default 7 = LIBERO; OpenArm = 16) |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | JAX preallocation fraction (0.6 in compose_ngc.yml) |
| `XLA_PYTHON_CLIENT_PREALLOCATE` | false for inference (less aggressive than training) |
| `FVK_PI05_RTX_FORCE_BF16` | Bypass FP8 for debugging (1 = bf16 only) |

## Quick reference — failure modes

| symptom | likely phase | fix |
|---|---|---|
| `import flash_rt` fails | 0 | rebuild image; check sm_121 was applied |
| `supports_nvfp4()` returns False | 0 | CMakeLists patch not applied; rebuild |
| inference produces NaNs | 1, 2, 4 | check LoRA merge ran, scales not saturated |
| post-unnorm cosine 0.92 | 2 | LoRA pre-merged in bf16 — verify Phase 2 patch |
| post-unnorm cosine ~ 0.98 | 3 | calibration set too small or unstratified |
| wrong action dim in response | 4 | pass `--robot-action-dim 16` (OpenArm) |
| robot misses every grasp | 4 → 6 | run parity again with `--robot-action-dim 16` |
| latency > 200 ms | 7 | check JPEG decode share; switch to raw arrays |
