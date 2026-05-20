#!/usr/bin/env bash
# Phase 1 full LIBERO eval on DGX Spark.
#
# Prereqs:
#   - Phase 0 smoke passed (scripts/spark_build_smoke.py exit 0)
#   - Phase 1 smoke passed (scripts/spark_phase1_libero_smoke.py exit 0)
#   - libero + robosuite + mujoco installed in the container
#     (the flashrt:spark image does NOT include these by default —
#      add `pip install libero robosuite` to a derived image)
#
# Usage:
#   scripts/spark_phase1_libero_run.sh /openpi_assets/pi05_libero
#   scripts/spark_phase1_libero_run.sh /openpi_assets/pi05_libero libero_spatial quick
#
# Acceptance gate:
#   The published FlashRT numbers for Pi0.5 + LIBERO on RTX 5090 are
#   the reference. Spark should land within 2 absolute percentage points
#   of those numbers since the JAX path is supposed to be numerically
#   parity-equivalent across consumer-Blackwell SKUs. Concrete cell to
#   check: libero_spatial quick (3 tasks x 3 trials) on RTX 5090 typically
#   reports ~89-95% success rate (varies with seed). Spark should report
#   the same range. Anything below 80% indicates either a calibration
#   regression on SM_121, a checkpoint-loading bug, or a kernel mismatch
#   — investigate before advancing to Phase 2.
#
# Output:
#   - libero_<suite>_jax_results.json in the current working directory
#   - Per-task success rates printed to stdout

set -euo pipefail

CKPT="${1:-}"
SUITE="${2:-libero_spatial}"
MODE="${3:-quick}"   # quick | full

if [[ -z "$CKPT" ]]; then
    echo "usage: $0 <checkpoint_dir> [task_suite] [quick|full]" >&2
    exit 2
fi

if [[ ! -d "$CKPT" ]]; then
    echo "error: $CKPT is not a directory" >&2
    exit 2
fi

# JAX environment for inference (matches eval_libero.py subprocess env)
export XLA_FLAGS='--xla_gpu_enable_triton_gemm=false --xla_gpu_autotune_level=0'
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL="$SCRIPT_DIR/../examples/thor/eval_libero.py"

if [[ ! -f "$EVAL" ]]; then
    echo "error: $EVAL not found" >&2
    exit 2
fi

ARGS=(
    --checkpoint "$CKPT"
    --framework jax
    --task_suite "$SUITE"
    --autotune 3
)

case "$MODE" in
    quick)
        ARGS+=(--quick)
        ;;
    full)
        # Use defaults (50 trials/task, all tasks in the suite)
        ;;
    *)
        echo "error: mode must be 'quick' or 'full' (got '$MODE')" >&2
        exit 2
        ;;
esac

echo "=== Phase 1 full LIBERO eval ==="
echo "  Checkpoint: $CKPT"
echo "  Suite:      $SUITE"
echo "  Mode:       $MODE"
echo "  Eval:       $EVAL"
echo "================================="

exec python3 "$EVAL" "${ARGS[@]}"
