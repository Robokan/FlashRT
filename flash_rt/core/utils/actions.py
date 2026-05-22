"""FlashRT — Action post-processing utilities."""

import os
import numpy as np

LIBERO_ACTION_DIM = 7


def unnormalize_actions(actions, norm_stats):
    """Unnormalize actions using q01/q99 statistics (pure numpy).

    Matches openpi's ``Unnormalize._unnormalize_quantile``
    (``openpi/src/openpi/transforms.py``) which does NOT clip the model
    output to ``[-1, 1]`` before mapping back to the quantile range.

    Historical note: FlashRT used to clip raw model output to ``[-1, 1]``
    before this affine map. That caused a deterministic action-space
    bias on any joint whose model output legitimately extrapolated past
    the training quantile range — for ``pi05_openarm_ngc_lora_v4`` /
    ``chocolate_bars_pi05_h10``, the shoulder-pitch joints (L3 and R3)
    consistently produced normalized output around -1.2 (~0.2 below the
    clip), which clipping floored to ``q01`` and surfaced as a constant
    +0.278 rad bias once the openpi adapter added it back to the
    delta-channel state. JAX-h10 on the same checkpoint produces
    sensible motion on those joints because openpi never clips. See
    docs/spark_status.md G6 for the diagnostic chain.
    """
    q01 = np.array(norm_stats["actions"]["q01"], dtype=np.float32)
    q99 = np.array(norm_stats["actions"]["q99"], dtype=np.float32)
    dim = min(actions.shape[-1], len(q01))
    if os.environ.get("FLASHRT_DEBUG_UNNORM"):
        raw = np.asarray(actions)
        a0 = raw[0] if raw.ndim >= 2 else raw
        rmin = raw[..., :dim].min(axis=tuple(range(raw.ndim - 1)))
        rmax = raw[..., :dim].max(axis=tuple(range(raw.ndim - 1)))
        n_lo = (raw[..., :dim] < -1.0).sum(axis=tuple(range(raw.ndim - 1)))
        n_hi = (raw[..., :dim] > 1.0).sum(axis=tuple(range(raw.ndim - 1)))
        print(f"[UNNORM-DEBUG] shape={raw.shape} dim={dim}", flush=True)
        print(f"[UNNORM-DEBUG]   raw a[0,:dim]            = {np.round(a0[:dim], 4).tolist()}", flush=True)
        print(f"[UNNORM-DEBUG]   per-joint min over chunk = {np.round(rmin, 4).tolist()}", flush=True)
        print(f"[UNNORM-DEBUG]   per-joint max over chunk = {np.round(rmax, 4).tolist()}", flush=True)
        print(f"[UNNORM-DEBUG]   per-joint #raw<-1        = {n_lo.tolist()}", flush=True)
        print(f"[UNNORM-DEBUG]   per-joint #raw>+1        = {n_hi.tolist()}", flush=True)
    unnorm = np.asarray(actions, dtype=np.float32).copy()
    unnorm[..., :dim] = (
        (unnorm[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6)
        + q01[:dim]
    )
    return unnorm
