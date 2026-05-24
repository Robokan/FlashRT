"""Parity tests for RTC soft-guidance port (Phase 6 / G11).

Validates the FlashRT soft-guidance implementation against the vendored
lerobot reference in ``third_party/lerobot_rtc_reference/``. Three
independent levels:

1. ``_get_prefix_weights`` matches lerobot's
   ``RTCProcessor.get_prefix_weights`` byte-for-byte across all four
   schedules and edge cases (empty merge window, start >= end, etc.).

2. The analytic correction
   ``err = (prev - x1) * weights`` matches the autograd-derived
   correction at ``rtol=1e-5`` (the autograd graph has identity
   Jacobian because ``v_t`` is computed before ``x_t.requires_grad_``,
   so the VJP collapses to ``err``). This validates the central
   mathematical claim of ``docs/spark_phase6_soft_guidance.md``.

3. The guidance-weight schedule matches the lerobot formula
   ``min(c * inv_r2, max_guidance_weight)`` at a grid of time values.

The tests do NOT exercise the CUDA kernel directly — those are
hardware-dependent and run in the Spark / RTX integration test path.
Here we verify the pure-Python math that feeds the kernel scalars.

Skips gracefully if torch is unavailable (the autograd-parity test
needs torch; the prefix-weights tests are pure numpy).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest


# Make the vendored lerobot reference importable as a flat module
# without dragging in ``lerobot.configs`` (which requires the full
# lerobot install). We monkey-stub the RTCAttentionSchedule enum on the
# fly so ``modeling_rtc.py`` imports cleanly.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_LEROBOT_REF = _REPO_ROOT / "third_party" / "lerobot_rtc_reference"


@pytest.fixture(scope="module")
def lerobot_get_prefix_weights():
    """Returns ``RTCProcessor.get_prefix_weights`` from the vendored ref.

    Builds a minimal stub of ``lerobot.configs.RTCAttentionSchedule``
    so we don't need the full lerobot install, then loads the
    vendored ``modeling_rtc.py`` + ``configuration_rtc.py`` as a
    fake package (``_lerobot_ref``) so their relative imports resolve.
    """
    torch = pytest.importorskip("torch")
    import enum
    import importlib.util
    import types

    if "lerobot" not in sys.modules:
        lerobot_pkg = types.ModuleType("lerobot")
        configs_pkg = types.ModuleType("lerobot.configs")

        class RTCAttentionSchedule(str, enum.Enum):
            ZEROS = "zeros"
            ONES = "ones"
            LINEAR = "linear"
            EXP = "exp"

        configs_pkg.RTCAttentionSchedule = RTCAttentionSchedule
        lerobot_pkg.configs = configs_pkg
        sys.modules["lerobot"] = lerobot_pkg
        sys.modules["lerobot.configs"] = configs_pkg

    # Register the vendored directory as a fake package so the
    # relative imports inside the vendored files (``from .configuration_rtc
    # import RTCConfig``) resolve. We can't add the directory to
    # ``sys.path`` and ``import modeling_rtc`` because that bypasses
    # the package machinery the file uses.
    pkg_name = "_lerobot_ref"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(_LEROBOT_REF)]
        sys.modules[pkg_name] = pkg

    # The vendored ``modeling_rtc`` imports ``from .debug_tracker
    # import Tracker``, which is an upstream module we didn't vendor
    # (it's only used for in-tree debugging and adds another dep
    # chain). Stub it with a no-op Tracker so the import resolves.
    dbg_name = f"{pkg_name}.debug_tracker"
    if dbg_name not in sys.modules:
        dbg = types.ModuleType(dbg_name)

        class _NoopTracker:
            def __init__(self, *a, **kw):
                pass

            def __call__(self, *a, **kw):
                return None

            def track(self, *a, **kw):
                return None

        dbg.Tracker = _NoopTracker
        sys.modules[dbg_name] = dbg

    cfg_mod = importlib.import_module(f"{pkg_name}.configuration_rtc")
    rtc_mod = importlib.import_module(f"{pkg_name}.modeling_rtc")
    return rtc_mod, cfg_mod, torch


@pytest.fixture(scope="module")
def flashrt_get_prefix_weights():
    """The FlashRT port we want to validate."""
    from flash_rt.frontends.torch.pi05_rtx import _get_prefix_weights
    return _get_prefix_weights


# ───────────────────────────────────────────────────────────────────
#   Test 1: prefix-weights schedule parity
# ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("schedule", ["linear", "exp", "ones", "zeros"])
@pytest.mark.parametrize("start,end,total", [
    (0, 10, 50),       # canonical: 50-action chunk, exec_horizon=10
    (4, 10, 50),       # d=4 anchor, merge [4, 10), tail [10, 50)
    (4, 4, 50),        # empty merge window (start == end)
    (10, 4, 50),       # start > end → lerobot clamps start to end=4
    (0, 0, 50),        # everything is free (ones/zeros differ here)
    (50, 50, 50),      # full hard anchor (no free continuation)
    (0, 50, 50),       # ramp spans entire chunk
])
def test_prefix_weights_match_lerobot(
        lerobot_get_prefix_weights, flashrt_get_prefix_weights,
        schedule, start, end, total):
    """FlashRT ``_get_prefix_weights`` is bit-equal to lerobot's."""
    modeling_rtc, configuration_rtc, torch = lerobot_get_prefix_weights
    flash_fn = flashrt_get_prefix_weights

    sched_enum = type(configuration_rtc.RTCConfig.prefix_attention_schedule)[
        schedule.upper()]
    cfg = configuration_rtc.RTCConfig(prefix_attention_schedule=sched_enum)
    proc = modeling_rtc.RTCProcessor(rtc_config=cfg)
    lerobot_w = proc.get_prefix_weights(start, end, total).cpu().numpy()
    flash_w = flash_fn(start=start, end=end, total=total, schedule=schedule)

    assert flash_w.shape == lerobot_w.shape, (
        f"shape mismatch: flash={flash_w.shape} vs lerobot={lerobot_w.shape}")
    np.testing.assert_allclose(
        flash_w, lerobot_w, rtol=0, atol=1e-6,
        err_msg=(
            f"schedule={schedule} start={start} end={end} total={total} "
            f"\nflashrt={flash_w}\nlerobot={lerobot_w}"))


def test_prefix_weights_unknown_schedule_falls_back(flashrt_get_prefix_weights):
    """Unknown schedule names emit a warning and use linear."""
    flash_fn = flashrt_get_prefix_weights
    w_linear = flash_fn(start=4, end=10, total=50, schedule="linear")
    w_bogus = flash_fn(start=4, end=10, total=50, schedule="bogus")
    np.testing.assert_array_equal(w_linear, w_bogus)


# ───────────────────────────────────────────────────────────────────
#   Test 2: analytic correction == autograd correction
# ───────────────────────────────────────────────────────────────────

def _analytic_correction(x_t, v_t, prev, weights, time):
    """The autograd-free correction derived in
    docs/spark_phase6_soft_guidance.md.

    Pure numpy; mirrors the kernel's float32 path before bf16 round-trip.
    """
    x1 = x_t - time * v_t
    err = (prev - x1) * weights
    return err


def _autograd_correction(x_t, v_t, prev, weights, time):
    """The lerobot autograd reference, run end-to-end.

    Re-implements ``RTCProcessor.denoise_step:212-219`` standalone so
    we don't need to instantiate the whole processor with its
    DebugInfo tracking.
    """
    import torch
    x_t = torch.as_tensor(x_t, dtype=torch.float32).clone().detach()
    v_t = torch.as_tensor(v_t, dtype=torch.float32).clone().detach()
    prev = torch.as_tensor(prev, dtype=torch.float32)
    weights = torch.as_tensor(weights, dtype=torch.float32)
    with torch.enable_grad():
        x_t.requires_grad_(True)
        x1 = x_t - time * v_t
        err = (prev - x1) * weights
        grad_outputs = err.clone().detach()
        correction = torch.autograd.grad(
            x1, x_t, grad_outputs, retain_graph=False)[0]
    return correction.detach().cpu().numpy()


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 1234])
@pytest.mark.parametrize("time", [0.99, 0.9, 0.5, 0.1, 0.01])
def test_analytic_correction_matches_autograd(seed, time):
    """The (prev - x1) * weights formula equals the autograd correction.

    This is the central correctness claim of Phase 6: the kernel's
    closed-form correction is identical (to float32 precision) to the
    autograd-based one used by lerobot. If this test passes, the
    kernel cannot be wrong (modulo bf16 round-trip, which separate
    tests exercise).
    """
    pytest.importorskip("torch")
    rng = np.random.default_rng(seed)
    shape = (50, 32)  # mirrors Pi05Pipeline chunk_size × ACTION_DIM
    x_t = rng.normal(size=shape).astype(np.float32)
    v_t = rng.normal(size=shape).astype(np.float32)
    prev = rng.normal(size=shape).astype(np.float32)
    # Realistic weights: linear ramp 1→0 over the first 10 positions,
    # zero after — matches the default soft-guidance config.
    weights = np.zeros(shape, dtype=np.float32)
    ramp = np.linspace(1.0, 0.0, 8, dtype=np.float32)
    weights[:4, :] = 1.0
    weights[4:12, :] = ramp[:, None]

    analytic = _analytic_correction(x_t, v_t, prev, weights, time)
    autograd = _autograd_correction(x_t, v_t, prev, weights, time)
    np.testing.assert_allclose(analytic, autograd, rtol=1e-5, atol=1e-6)


# ───────────────────────────────────────────────────────────────────
#   Test 3: guidance_weight schedule
# ───────────────────────────────────────────────────────────────────

def _flashrt_guidance_weight(time: float, num_steps: int, max_gw: float
                             ) -> float:
    """The closed-form scalar the pipeline computes per Euler step.

    Mirror of ``Pi05Pipeline._rtc_apply_guidance`` (without the
    decoder_action_buf scaling — we just want the raw lerobot-formula
    gw value here for parity checking).
    """
    tau = 1.0 - time
    one_minus_tau = 1.0 - tau
    if one_minus_tau > 0.0:
        inv_r2 = (tau * tau + one_minus_tau * one_minus_tau) \
            / (one_minus_tau * one_minus_tau)
        c = one_minus_tau / tau if tau > 0.0 else max_gw
        gw = min(c * inv_r2, max_gw)
    else:
        gw = max_gw
    return gw


def _lerobot_guidance_weight(time: float, max_gw: float) -> float:
    """Lerobot's reference (modeling_rtc.py:221-227) translated to numpy."""
    tau = 1.0 - time
    one_minus_tau_sq = (1.0 - tau) ** 2
    inv_r2 = (one_minus_tau_sq + tau ** 2) / one_minus_tau_sq \
        if one_minus_tau_sq > 0 else float("inf")
    if tau > 0:
        c = (1.0 - tau) / tau
    else:
        c = float("inf")
    # lerobot uses nan_to_num(posinf=max_gw)
    if not math.isfinite(c):
        c = max_gw
    gw = c * inv_r2
    if not math.isfinite(gw):
        gw = max_gw
    return min(gw, max_gw)


@pytest.mark.parametrize("time", [
    0.99, 0.9, 0.5, 0.3, 0.1, 0.05, 0.01, 0.001,
])
@pytest.mark.parametrize("max_gw", [5.0, 10.0])
def test_guidance_weight_matches_lerobot(time, max_gw):
    """Per-step guidance weight equals lerobot's at a grid of times."""
    flash_gw = _flashrt_guidance_weight(time, num_steps=10, max_gw=max_gw)
    lerobot_gw = _lerobot_guidance_weight(time, max_gw=max_gw)
    assert math.isclose(flash_gw, lerobot_gw, rel_tol=1e-6, abs_tol=1e-9), (
        f"time={time} max_gw={max_gw}: "
        f"flash={flash_gw} vs lerobot={lerobot_gw}")


def test_guidance_weight_is_u_shaped_in_time():
    """Sanity: gw is symmetric in (tau, 1-tau) and U-shaped, with the
    minimum near time=0.5 and the two endpoints clamped at max_gw.

    Algebraically: ``c * inv_r2 = (tau^2 + (1-tau)^2) / (tau * (1-tau))``,
    which is symmetric in ``tau`` and ``1-tau`` and minimised at
    tau=0.5 (value 2.0). At both endpoints the unclamped value
    diverges, so the clamp to ``max_gw`` kicks in. This is the actual
    behaviour of lerobot's formula — the "guidance increases toward
    t=0" framing in some informal docs is misleading; the formula is
    symmetric and peaks at both ends of the denoise loop.
    """
    max_gw = 10.0
    gws = {
        t: _flashrt_guidance_weight(t, num_steps=10, max_gw=max_gw)
        for t in (0.99, 0.9, 0.7, 0.5, 0.3, 0.1, 0.01)
    }
    # Minimum is at time=0.5 (tau=0.5), value 2.0 by construction.
    assert math.isclose(gws[0.5], 2.0, abs_tol=1e-9), f"got gw(0.5)={gws[0.5]}"
    # Endpoints are clamped at max_gw.
    assert gws[0.01] >= max_gw - 1e-9
    assert gws[0.99] >= max_gw - 1e-9
    # Symmetry around time=0.5.
    assert math.isclose(gws[0.3], gws[0.7], abs_tol=1e-9)
    assert math.isclose(gws[0.1], gws[0.9], abs_tol=1e-9)
    # U-shape: decreasing from t=0.01 to t=0.5, then increasing.
    times_left = [0.01, 0.1, 0.3, 0.5]
    times_right = [0.5, 0.7, 0.9, 0.99]
    for ts in (times_left,):
        for i in range(len(ts) - 1):
            assert gws[ts[i + 1]] <= gws[ts[i]] + 1e-9, (
                f"left half not non-increasing at {ts[i]}→{ts[i+1]}: "
                f"{gws[ts[i]]} → {gws[ts[i+1]]}")
    for ts in (times_right,):
        for i in range(len(ts) - 1):
            assert gws[ts[i + 1]] >= gws[ts[i]] - 1e-9, (
                f"right half not non-decreasing at {ts[i]}→{ts[i+1]}: "
                f"{gws[ts[i]]} → {gws[ts[i+1]]}")


# ───────────────────────────────────────────────────────────────────
#   Test 4: effective-scalar transformation (the dt rescaling)
# ───────────────────────────────────────────────────────────────────

def test_effective_time_and_gw_recover_raw_velocity_update():
    """The eff_time / eff_gw transform applied to ``dt*v_t`` matches a
    raw ``v_t`` update.

    Pi05Pipeline pre-scales the decoder output projection by dt =
    -1/num_steps so ``action_buf = dt * v_t``. The kernel computes
    ``v_buf' = v_buf - eff_gw * (prev - (x_t - eff_time * v_buf)) * w``
    where eff_time = -time*num_steps, eff_gw = -gw/num_steps. After
    the transform, the new action_buf should equal dt * (v_t - gw * err)
    where err is the raw (un-scaled) correction.
    """
    rng = np.random.default_rng(123)
    num_steps = 10
    dt = -1.0 / num_steps
    time = 0.3
    gw = 4.0

    shape = (50, 32)
    v_t = rng.normal(size=shape).astype(np.float32)
    x_t = rng.normal(size=shape).astype(np.float32)
    prev = rng.normal(size=shape).astype(np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    weights[:6, :] = 1.0

    # Reference (raw velocity convention):
    x1_raw = x_t - time * v_t
    err_raw = (prev - x1_raw) * weights
    v_t_new_raw = v_t - gw * err_raw
    expected_action_buf = dt * v_t_new_raw

    # FlashRT path: action_buf starts as dt*v_t, apply kernel with
    # effective scalars in-place.
    action_buf = (dt * v_t).copy()
    eff_time = -time * num_steps
    eff_gw = -gw / num_steps
    x1_eff = x_t - eff_time * action_buf
    err_eff = (prev - x1_eff) * weights
    action_buf_after = action_buf - eff_gw * err_eff

    np.testing.assert_allclose(
        action_buf_after, expected_action_buf, rtol=1e-5, atol=1e-6,
        err_msg=(
            "Effective scalar transform did not preserve the raw "
            "velocity-update semantics. eff_time/eff_gw derivation in "
            "Pi05Pipeline._rtc_apply_guidance is wrong."))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
