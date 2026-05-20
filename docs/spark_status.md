# Spark port — current status

Snapshot of where the FlashRT-on-DGX-Spark work stands. Updated as
phases advance. The 7-phase plan lives at:
```
~/.cursor/plans/flashrt_jax_on_dgx_spark_f0288dd4.plan.md
```
Operational details: [`spark_runbook.md`](spark_runbook.md).
Durable agent rules: [`../AGENTS.md`](../AGENTS.md).

## TL;DR

Code for all 7 phases is in place and committed to `spark-sm121-port`
on `Robokan/FlashRT`. **Phases 0 and 1 (smoke) are now verified
end-to-end on hardware.** JAX-on-aarch64-Blackwell works out of the
box via `jax[cuda12]` PyPI wheels (no NGC base or source build
needed). Pi0.5 LIBERO loads + runs in ~57 ms/iter on the GB10 — well
inside the 200 ms ceiling and right on the plan's 50–150 ms
prediction. Phases 2–7 still un-run against real artifacts (OpenArm
LoRA checkpoint, robot, calibration data).

## Phase status

| Phase | Description | Code | Verified on Spark? |
|---|---|---|---|
| 0 | Build FlashRT for SM_121 + aarch64 | done — `docker/Dockerfile.spark`, `docker/compose.spark.yml`, `CMakeLists.txt` patches, `scripts/spark_build_smoke.py` | **configure: yes; full build: yes (native venv, -j8, ~8.4 min); smoke gate: PASS (9/9) — see "Phase 0 hardware-verified results" below** |
| 1 | `pi05_libero` Orbax load via `Pi05JaxFrontendRtx` | done — `scripts/spark_phase1_libero_smoke.py`, `scripts/spark_phase1_libero_run.sh` | **smoke gate: PASS (5/5) — 57.2 ms mean steady-state, see "Phase 1 hardware-verified results" below; full LIBERO simulator eval: still pending** |
| 2 | LoRA Orbax load + fp32 merge (`pi05_openarm_ngc_lora_v4`) | done — `_maybe_merge_lora` in `flash_rt/frontends/jax/pi05_rtx.py`, `tests/test_lora_merge_jax_loader.py`, `scripts/spark_phase2_lora_load.py` | unit tests: not run on Spark |
| 3 | FP8 calibration on stratified OpenArm samples | done — `scripts/spark_phase3_prepare_calib.py`, `scripts/spark_phase3_run_calib.py` | no |
| 4 | Parity vs the openpi JAX server | done — `scripts/spark_phase4_parity.py` and `robot_action_dim` patches in `pi05_rtx.py` (torch + jax frontends and `flash_rt/api.py`) | no |
| 5 | Serve FlashRT via openpi WebsocketPolicyServer | done — `flash_rt/serving/openpi_adapter.py` (`FlashRTPolicyAdapter`), `scripts/serve_policy_flashrt.py`, `scripts/spark_phase5_adapter_smoke.py` | no |
| 6 | End-to-end robot success comparison | done — `scripts/spark_phase6_robot_compare.py` (append + compare CLI) | no |
| 7 | Latency breakdown for colocation decision | done — `scripts/spark_phase7_latency_breakdown.py` | no |

## What's definitely working on Spark

- `nvcc -gencode arch=compute_121a,code=sm_121a` compiles and runs.
- `cmake -B build -S . -DGPU_ARCH=121` configures cleanly. It prints:
  ```
  -- NVFP4/W4A8 support: ENABLED (sm_121a)
  -- Using gencode flag: -gencode=arch=compute_121a,code=sm_121a
  -- FA2 in-SO attention: ENABLED (sm_121)
  -- Motus beta kernels: ENABLED
  -- SM120a CUTLASS block-128 FP8: ENABLED
  -- SM120a CUTLASS NVFP4 W4A16 GEMM: ENABLED
  -- FA2 vendor arch: sm_80 + sm_120 + sm_121 AOT + compute_120 PTX fallback (Spark default)
  ```
- **`cmake --build build -j8` finishes in ~8.4 min wall-time** on a
  workstation Spark (with the desktop + Cursor running). All 58
  ninja steps green; only two harmless "unused variable / function"
  warnings, zero errors. Both `.so` artifacts produced:
  - `flash_rt/flash_rt_kernels.cpython-312-aarch64-linux-gnu.so`
  - `flash_rt/flash_rt_fa2.cpython-312-aarch64-linux-gnu.so`
- Memory at `-j8`: peaked at ~40 GB free during the CUTLASS-template
  instantiation phase (8 nvcc concurrent), recovered to >100 GB once
  template TUs finished. No swap touched, no OOM-killer activity.
  This validates the AGENTS.md `BUILD_J=8` budget for the 110 GB-free
  case; the documented `BUILD_J=4` workstation default remains correct
  for tighter memory situations.

## Phase 0 hardware-verified results

`python scripts/spark_build_smoke.py` exits 0 with all 9 checks PASS:

```
PASS  import flash_rt                  version=0.1.0
PASS  locate flash_rt_kernels.so       flash_rt_kernels.cpython-312-aarch64-linux-gnu.so
PASS  locate flash_rt_fa2.so           flash_rt_fa2.cpython-312-aarch64-linux-gnu.so
PASS  host arch                        aarch64 (Grace)
PASS  get_gpu_sm_version               121 (DGX Spark GB10)
PASS  supports_fp8()                   True
PASS  supports_nvfp4()                 True
PASS  get_gpu_name                     'NVIDIA GB10'
PASS  fa2.fwd_bf16 launch              shape=(1, 1024, 8, 256) dtype=torch.bfloat16 |O|.mean()=0.041
```

The fa2.fwd_bf16 step was originally written assuming a high-level
`(q, k, v) -> Tensor` wrapper that doesn't exist — `flash_rt_fa2` is
a low-level pybind ABI taking raw device pointers + a pre-allocated
O and softmax_lse in BSHD layout (matches
`flash_rt/hardware/rtx/attn_backend.py::_call_fvk_fa2`). The smoke
script was fixed to mirror that calling convention exactly with
B=1, S=1024, H=8, D=256 — the same code path Pi0.5 will hit at
inference time. PyTorch was installed via the aarch64 CUDA wheel
channel:
```bash
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch
# → torch 2.11.0+cu128, bundles CUDA 12.8 runtime, talks to driver 580 OK
```

## Phase 1 hardware-verified results

`python scripts/spark_phase1_libero_smoke.py --checkpoint \
~/.cache/openpi/openpi-assets/checkpoints/pi05_libero` exits 0 with
all 5 checks PASS:

```
PASS  checkpoint dir
PASS  flash_rt.load_model(framework='jax')          12.4s
PASS  first infer (calibration + graph capture)     1.11s
PASS  steady-state latency                          mean=57.2ms p50=57.2ms p99=60.5ms (ceiling=200ms)
PASS  output shape                                  (10, 7)
PASS  finite outputs                                1400 values, 0 NaN/Inf
```

Reaching this PASS required three fixes on top of the previously
committed Spark port:

1. **`flash_rt/hardware/__init__.py`**: `detect_arch()` had no entry
   for `(major, minor) == (12, 1)` and refused to load on Spark.
   Added `"rtx_sm121"` arch string + `_PIPELINE_MAP` entries that
   mirror the `"rtx_sm120"` ones for pi05/pi0/groot/motus/pi0fast
   (the compiled `.so` is gencode `sm_121a` and the frontends are
   arch-agnostic, so it's a trivial dispatch alias today; keeping
   the string distinct preserves the option of Spark-specific
   dispatch later).

2. **`flash_rt/core/weights/loader.py` — openpi-path discovery**: the
   hardcoded list `["/workspace/src"]` was stale (compose.spark.yml
   mounts at `/openpi/src`, native dev has `~/sparkpack/openpi/src`).
   Extended to try `$OPENPI_SRC`, `/openpi/src`, `/workspace/src`,
   and `~/sparkpack/openpi/src` in order, so the loader picks up
   openpi's `restore_params` regardless of layout.

3. **`flash_rt/core/weights/loader.py` — orbax 0.11 metadata**: the
   direct-orbax fallback used `metadata["params"]`, which broke
   because `ckptr.metadata()` now returns a `StepMetadata` wrapper
   (not subscriptable). Tree metadata moved to
   `.item_metadata` (a `_TreeMetadataImpl` which IS subscriptable).
   Wrapped with `getattr(metadata, "item_metadata", metadata)` so
   both old and new orbax APIs work.

JAX install path that worked: `uv pip install "jax[cuda12]"
orbax-checkpoint flax ml_dtypes` against PyPI gave jax 0.10.1 +
jax-cuda12-{pjrt,plugin} which sees the GB10 immediately
(`jax.devices()==[CudaDevice(id=0)]`, `default_backend=='gpu'`). No
NGC base image or source build needed. Driver 580.142 + bundled CUDA
12.8 runtime works on SM_121. Same story for sentencepiece +
safetensors (`uv pip install sentencepiece safetensors`).

## Calibration warning (followup, not a blocker)

The first FP8 calibration on pi05_libero with 1 random synthetic
observation flagged 5 scales >20× median, the worst being
`encoder_ffn_down_w_16` at 20.571×. Calibration completed and
inference outputs are all finite, but FP8 dynamic-range headroom on
those layers is compressed. The smoke uses 1 random obs which is the
worst-possible-case for calibration; in Phase 3 we calibrate on
50–100 stratified real observations, which should pull these scales
in. If Phase 2's parity numbers come in low for similar-named layers,
revisit calibration percentile / sample count.

## What's not yet verified

- The flashrt_spark Docker image build (`docker compose -f
  docker/compose.spark.yml build flashrt_spark`).
- The full LIBERO simulator eval (`scripts/spark_phase1_libero_run.sh`)
  — needs `libero + robosuite + mujoco` installed which is a separate
  install adventure on aarch64+Blackwell. The smoke gate validates
  the FlashRT stack itself with zero new code, so this is purely a
  policy-quality regression check. Worth skipping unless we suspect
  numerical drift.
- Phases 2–7 (`pi05_openarm_ngc_lora_v4` load, calibration on real
  OpenArm data, parity vs JAX server, server wrapping, robot test,
  latency breakdown). Code is in place; not yet run on hardware.

## Lessons learned the hard way

1. **`-j$(nproc)` crashed the workstation.** Spark's unified LPDDR pool
   means a heavy parallel nvcc compile competes with the desktop + IDE
   for the same memory. The OOM killer took down Cursor, not the build.
   Hard reboot required. Safe default is now `BUILD_J=4`. See
   `docker/Dockerfile.spark`'s comment block and `AGENTS.md` for the
   formula.

2. **Don't `tee` long compiles through the shell that Cursor mirrors.**
   nvcc + CUTLASS warnings flood the terminal mirror, can lock the UI.
   Use `nohup setsid ... > /tmp/log 2>&1 &` and poll the log file with
   bounded reads.

3. **`prlimit --as=8GB` looks safe but isn't.** nvcc/cicc memory-map
   large CUTLASS template specializations; tight `--as` caps trigger
   spurious "cannot allocate memory" failures even when resident usage
   is fine. The `-j` cap is the actual protection.

4. **Don't add FlashRT code to openpi.** The openpi-side scripts and
   adapter that originally lived in `Robokan/openpi` were migrated
   into FlashRT in commit `dc00926`. Robokan/openpi `main` is now
   rebase-clean against Physical-Intelligence/openpi upstream (just
   the user's prior non-FlashRT commits + a 2-line breadcrumb in
   `compose_ngc.yml`). Keep it that way.

## Next steps (when you're ready to run a real build)

```bash
# Option A (recommended): docker
cd ~/sparkpack/FlashRT
free -h                                                  # confirm ≥ 40 GB free
docker compose -f docker/compose.spark.yml build flashrt_spark
docker compose -f docker/compose.spark.yml run --rm flashrt_spark \
    python3 scripts/spark_build_smoke.py
```

```bash
# Option B: native uv venv (already configured in this checkout — the
# .venv with pybind11/ninja and CUTLASS clone are on disk).
cd ~/sparkpack/FlashRT && source .venv/bin/activate
nohup setsid cmake --build build -j4 > /tmp/flashrt_build.log 2>&1 &
echo $! > /tmp/flashrt_build.pid

# Poll without streaming:
tail -n 5 /tmp/flashrt_build.log
ls flash_rt/*.so 2>/dev/null      # success when these appear

# Then make flash_rt importable + smoke test:
uv pip install -e ".[torch,jax]"
python scripts/spark_build_smoke.py
```

Expected wall-time for the full kernel build at `-j4` on Spark:
~25–45 min. Worst case it hits a real `sm_121` codegen incompatibility
in some CUTLASS template; that's the bug the build is designed to
expose.

## Cross-references

- `docs/spark_runbook.md` — full 7-phase runbook (commands + acceptance
  criteria for every gate)
- `AGENTS.md` — durable rules for any agent working in this repo
- `docker/Dockerfile.spark` — the SM_121 build, with the full
  `BUILD_J` rule of thumb in the comments
- `docker/compose.spark.yml` — flashrt_spark compose service
- `~/.cursor/plans/flashrt_jax_on_dgx_spark_f0288dd4.plan.md` —
  original 7-phase plan
- Prior workspace transcript:
  `/home/evaughan/.cursor/projects/home-evaughan-sparkpack-openpi/agent-transcripts/d6216242-2ae4-4778-92f0-643c93db89e5/d6216242-2ae4-4778-92f0-643c93db89e5.jsonl`
