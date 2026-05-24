# Phase 6 — Soft-guidance RTC port (lerobot → FlashRT)

**Status:** plan, not yet implemented.  
**Prerequisite:** Phase G10 mode-5 stabilization (commit `135b85b`) shipped and tested.  
**Reference:** `third_party/lerobot_rtc_reference/README.md`

## Why

FlashRT's current mode 5 uses **hard-freeze prefix inpainting** (CUDA
mask op that clobbers the noise tensor at prefix positions and zeros
the per-step velocity update there). It works — mode 5 no longer
SafetyStops on the OpenArm chocolate_bars task — but motion is still
jerky at chunk boundaries because the model's free continuation past
the frozen region has **no continuity guarantee** against the frozen
anchor. Empirically: 1.4 rad joint jumps at the splice on outlier-
latency chunks (observed 2026-05).

LeRobot ships the actual RTC-paper algorithm: **soft guidance via
gradient correction**. The diffusion model's velocity field is nudged
each Euler step toward making the predicted endpoint match the
inflight prefix, weighted by a time-decay schedule. The model is free
to integrate the constraint into a coherent trajectory rather than
fight against fixed positions. This is what `lerobot/pi05_base` uses
on real robots and what the chocolate_bars checkpoint was trained
against (state-in-prompt Pi0.5).

## Math — the entire algorithm in 5 lines

LeRobot's `RTCProcessor.denoise_step` uses autograd, but for Pi's
flow matching the gradient is analytic. From `modeling_rtc.py:212-219`:

```python
with torch.enable_grad():
    v_t = original_denoise_step_partial(x_t)   # v_t is treated as constant
    x_t.requires_grad_(True)
    x1_t = x_t - time * v_t                    # predicted endpoint
    err = (prev_chunk_left_over - x1_t) * weights
    grad_outputs = err.clone().detach()
    correction = torch.autograd.grad(x1_t, x_t, grad_outputs)[0]
```

Since `v_t` is computed BEFORE `x_t.requires_grad_(True)` is set, the
autograd graph treats v_t as a constant. The function being
differentiated is `x1_t(x_t) = x_t - time * v_t`, whose Jacobian
w.r.t. `x_t` is the identity. The vector-Jacobian product is therefore:

```
correction = grad_outputs · I = err = (prev - x1_t) * weights
```

**No autograd. No PyTorch. Three element-wise ops, fully fusable.**

```
x1 = x_t - time * v_t
err = (prev - x1) * weights
v_new = v_t - guidance_weight * err
```

The `guidance_weight` IS time-varying per Euler step (`modeling_rtc.py:221-227`):

```
tau = 1 - time
inv_r2 = (tau^2 + (1-tau)^2) / (1-tau)^2
c = (1-tau) / tau                     # clamped at posinf → max_guidance_weight
guidance_weight = min(c * inv_r2, max_guidance_weight)
```

Where `time` decreases from 1.0 toward 0.0 across the Euler loop
(Pi convention; see `lerobot_modeling_pi05.py:833-837`,
`dt = -1.0 / num_steps`, `time = 1.0 + step * dt`). `guidance_weight`
grows as denoising progresses, peaking near `time=0`.

## Prefix weight schedule (`modeling_rtc.py:251-298`)

Per-position weight of shape `(chunk_size,)`. With
`start = inference_delay`, `end = execution_horizon`:

- Indices `[0, start)` → weight `1.0` (full guidance, "we have already
  committed to playing these")
- Indices `[start, end)` → linear ramp `1 → 0` (the merge window)
- Indices `[end, chunk_size)` → weight `0.0` (no guidance, free
  continuation)

EXP schedule replaces the linear ramp with
`lin * (e^lin - 1) / (e - 1)` (sharper falloff). LERobot's default
config is `LINEAR`; the public docs example uses `EXP`. The kinetix
canonical uses LINEAR.

## File-by-file port plan

### 1. `csrc/fvk/rtc_guidance.cu` (new file, ~80 lines)

One fused kernel per Euler step. Takes `v_t`, `x_t`, `prev`,
`weights`, `time` scalar, `guidance_weight` scalar; writes `v_new` to
the same buffer as `v_t`:

```cuda
__global__ void rtc_guidance_correction_bf16_kernel(
    __nv_bfloat16* __restrict__ v,           // in/out (decoder_action_buf)
    const __nv_bfloat16* __restrict__ x_t,   // read-only (diffusion_noise)
    const __nv_bfloat16* __restrict__ prev,  // read-only (rtc_prev_chunk_buf)
    const __nv_bfloat16* __restrict__ weights, // read-only (rtc_weights_buf)
    float time, float guidance_weight, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v_val   = __bfloat162float(v[i]);
  float x_val   = __bfloat162float(x_t[i]);
  float prev_val= __bfloat162float(prev[i]);
  float w_val   = __bfloat162float(weights[i]);
  float x1   = x_val - time * v_val;
  float err  = (prev_val - x1) * w_val;
  float vnew = v_val - guidance_weight * err;
  v[i] = __float2bfloat16(vnew);
}
```

Wrapper:
```cpp
extern "C" void fvk_rtc_guidance_correction_bf16(
    uintptr_t v, uintptr_t x_t, uintptr_t prev, uintptr_t weights,
    float time, float guidance_weight, int n, cudaStream_t stream);
```

Python binding in `flash_rt_fvk` mirrors the existing element-wise
ops (`gate_mul_residual`, `residual_add`).

### 2. `flash_rt/models/pi05/pipeline_rtx.py` — change `transformer_decoder` (~40 lines net)

In `_allocate_buffers`:

**Replace:**
```python
B["rtc_neg_mask"] = CudaBuffer.device_zeros(ds * ACTION_DIM, BF16)
B["rtc_prev_chunk_masked"] = CudaBuffer.device_zeros(ds * ACTION_DIM, BF16)
```

**With:**
```python
B["rtc_prev_chunk"] = CudaBuffer.device_zeros(ds * ACTION_DIM, BF16)
B["rtc_weights"]    = CudaBuffer.device_zeros(ds * ACTION_DIM, BF16)  # broadcast
```

In `transformer_decoder`, after the existing pre-Euler RTC init
(which currently does `noise = (1-mask)*noise + mask*prev`), **delete
those two ops entirely**. With soft guidance the noise is just noise.

Per Euler step, after computing `decoder_action_buf` (the v_t) AND
before the `residual_add(diffusion_noise, decoder_action_buf, ...)`:

**Delete:**
```python
fvk.gate_mul_residual(
    B["decoder_action_buf"], B["decoder_action_buf"],
    B["rtc_neg_mask"], n_noise, stream=stream)
```

**Insert:**
```python
# RTC soft guidance: nudge v_t toward making (x_t - time*v_t) match prev.
# Pi convention: time runs 1.0 → 0.0 across the Euler loop.
time_val = 1.0 + step * (-1.0 / self.num_steps)
gw = self._rtc_guidance_weight(time_val)  # see formula above
fvk.rtc_guidance_correction(
    B["decoder_action_buf"].ptr.value,   # v_t in/out
    B["diffusion_noise"].ptr.value,      # x_t read-only
    B["rtc_prev_chunk"].ptr.value,
    B["rtc_weights"].ptr.value,
    time_val, gw, n_noise, stream=stream)
```

Then the existing `residual_add` (`noise += action_buf`) is unchanged.

Static-graph compatibility: the kernel has the same shape every call,
and `time` / `guidance_weight` enter as kernel scalars (not pointer
arithmetic), which CUDA graph capture handles fine. No graph
recapture needed when these scalars change between inferences — the
graph instantiation uses `cudaGraphLaunchParams` with explicit
parameter pointers; we'll need to thread the scalars through the
existing graph-capture path, but it's a straight extension of how
`step` is already handled for style modulation (see
`_style_slice_ptr("decoder_style_final", step)`).

### 3. `flash_rt/frontends/torch/pi05_rtx.py` — change `_stage_rtc_inputs` (~50 lines)

Currently builds `neg_mask_np` (size `(ds, ACTION_DIM)` with -1.0 in
the first `d` rows) and `pcm_np` (prev values in first `d` rows).

Replace with:

```python
def _stage_rtc_inputs(self, observation, stream_int):
    d_raw = observation.get("_rtc_inference_delay", 0)
    prev = observation.get("_rtc_prev_chunk", None)
    execution_horizon = observation.get(
        "_rtc_execution_horizon", _DEFAULT_EXECUTION_HORIZON)
    try:
        d = int(d_raw)
    except (TypeError, ValueError):
        d = 0
    active = (
        prev is not None and d > 0 and d <= self.chunk_size
        and prev.shape[-2:] == (d, ACTION_DIM))

    if not active:
        if not self._rtc_last_call_active:
            return
        self._rtc_prev_chunk_buf.zero_()
        self._rtc_weights_buf.zero_()  # all-zero weights → guidance is identity
        # ... upload zeros to pipeline ...
        self._rtc_last_call_active = False
        return

    # Pad prev to full chunk_size (lerobot ZERO-PAD convention,
    # modeling_rtc.py:196-199). Padded positions get weight 0 so
    # they're inert.
    prev_np = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
    prev_np[:d, :] = np.ascontiguousarray(prev, dtype=np.float32)

    # Build per-position weights (modeling_rtc.py:251-298).
    weights_np = _get_prefix_weights(
        start=d, end=execution_horizon, total=self.chunk_size,
        schedule=self._rtc_schedule)
    # Broadcast over action_dim axis so the kernel does element-wise
    # multiply without a separate broadcast op.
    weights_full = np.broadcast_to(
        weights_np[:, None], (self.chunk_size, ACTION_DIM)
    ).copy()

    prev_t = torch.from_numpy(prev_np).to(bf16)
    weights_t = torch.from_numpy(weights_full).to(bf16)
    self._rtc_prev_chunk_buf.copy_(prev_t, non_blocking=True)
    self._rtc_weights_buf.copy_(weights_t, non_blocking=True)
    self._copy_tensor_to_pipeline_buf_stream(
        self._rtc_prev_chunk_buf,
        self.pipeline.rtc_prev_chunk_buf, stream_int)
    self._copy_tensor_to_pipeline_buf_stream(
        self._rtc_weights_buf,
        self.pipeline.rtc_weights_buf, stream_int)
    self._rtc_last_call_active = True
```

Where `_get_prefix_weights` is a translation of
`RTCProcessor.get_prefix_weights` (`modeling_rtc.py:251-298`),
implemented in pure numpy with the same `LINEAR` / `EXP` / `ONES` /
`ZEROS` schedule options.

### 4. `flash_rt/serving/openpi_adapter.py` — pass-through `_rtc_execution_horizon`

Trivial: currently passes `_rtc_prev_chunk` + `_rtc_inference_delay`
through to `extra_obs`. Add `_rtc_execution_horizon` and
`_rtc_schedule` to the pass-through dict.

### 5. `flash_rt/runtime/rtc.py` — minor

`_maybe_augment_with_prefix_locked` currently sizes the prev_prefix as
`cm[idx:idx+d_pred]`, which matches `lerobot.ActionQueue.get_left_over`
semantics (the unconsumed actions of the current chunk). Keep this.

Add a new config field `execution_horizon: int = 10` (lerobot default)
and stuff it into the augmented obs as `_rtc_execution_horizon`. Also
optionally include `_rtc_schedule` for explicit per-call schedule
control (default = LINEAR per lerobot's `RTCConfig`).

### 6. `flash_rt/serving/chunked_websocket_client.py` — mode 5 defaults

```python
return RTCConfig(
    **mode5_kwargs,
    blend_steps=0,                  # NO LONGER NEEDED — soft guidance
                                    # is the smoothness mechanism. The
                                    # client-side seam blend we added
                                    # in G10 was a safety net for
                                    # hard-freeze; soft guidance
                                    # subsumes it.
    enable_prefix_freeze=True,
    prefix_freeze_max_steps=None,   # No cap needed — soft guidance
                                    # degrades gracefully when
                                    # d_actual overshoots d_pred (the
                                    # later weights ramp to 0 anyway).
                                    # The cap=12 we added in G10 was
                                    # specifically a workaround for
                                    # hard-freeze's EMA pinning bug.
    execution_horizon=10,           # lerobot default
    rtc_schedule="LINEAR")          # lerobot default
```

### 7. Tests — `tests/test_rtc_guidance.py` (new)

Parity test against the vendored reference. Loads
`third_party/lerobot_rtc_reference/modeling_rtc.py` directly,
constructs a fake `original_denoise_step_partial = lambda x: torch.randn_like(x)`,
runs both implementations on a synthetic `(x_t, prev, weights)`, and
asserts the analytic correction matches autograd output within
`rtol=1e-5` (BF16 round-trip tolerance).

Specifically:
- `test_analytic_correction_matches_autograd` — verifies the simplified
  derivation `correction = (prev - x1) * weights` matches the autograd
  result for several random seeds.
- `test_guidance_weight_formula` — verifies `guidance_weight(t)` matches
  the lerobot expression at `t = {0.9, 0.5, 0.1, 0.01}`.
- `test_prefix_weights_linear` / `_exp` / `_zeros` / `_ones` — verifies
  the four schedule modes match `RTCProcessor.get_prefix_weights`
  outputs exactly.
- `test_pipeline_rtc_end_to_end` — full pipeline integration: build a
  Pi05Pipeline, feed it a `(prev, d, weights)`, capture
  `decoder_action_buf` at each Euler step, compare against a torch
  reference implementation of the same algorithm with the same inputs.

## Speed budget

Per inference: 10 Euler steps × 1 new fused element-wise kernel of
`(50 × 32) = 1600` elements. Kernel cost dominated by launch
overhead (~5 µs), so total added cost is **~50 µs per inference**
(< 0.03% of the ~200 ms BF16 latency). Negligible.

## Wire-format compatibility

The server adds `_rtc_execution_horizon` and (optionally)
`_rtc_schedule` to the response/request pass-through. Old SparkJAX
clients that don't send these will get the lerobot defaults
(execution_horizon=10, schedule=LINEAR). Old FlashRT clients calling
new FlashRT servers also work — the server already does
`obs.get(..., default)` for all `_rtc_*` fields, so missing fields
fall back to behaviour-preserving defaults.

## Open questions to resolve during implementation

1. **Schedule default.** LeRobot's `RTCConfig` defaults to LINEAR
   with a TODO comment to change it to EXP. The public docs example
   uses EXP. Pick one and document the rationale. (Suggested: EXP per
   the docs — the linear-ramp falloff lets the model "fight" the
   prefix more in the middle of the merge window, which the EXP curve
   damps faster.)

2. **`execution_horizon` exposure.** Should this be a server-side
   config (set when the model loads) or a per-call client hint?
   LeRobot makes it both — `RTCConfig.execution_horizon` is the
   default, but `predict_action_chunk(execution_horizon=N)` overrides
   per call. FlashRT can do the same with a default in the model
   config and a per-obs override field.

3. **Should mode 5 keep `miss_policy="block"`?** With soft guidance
   working, splice cliffs are smoothed by the model itself. Hard
   deadline misses (no fresh chunk ready when needed) are still a
   pipeline-rebuild issue, not a smoothness issue. `block` remains
   the right answer for SparkJAX's strict per-tick safety.

4. **Cap removal.** Once soft guidance ships, the `cap=12` we added
   in G10 can probably go away (it was a workaround for hard-freeze's
   EMA pinning). But the EMA-pinning itself is independent — the
   prev_chunk size is still EMA-sized, and that pins to `horizon/2`
   under rebuild storms. The d_pred sizing is fine for soft guidance
   because the late-position weights ramp to 0 — over-shooting
   d_pred just makes the unused tail bigger, no harm done.

5. **Switch to peak latency for `d_pred` sizing?** LeRobot uses
   `latency_tracker.max()`. FlashRT uses EMA via
   `_record_latency_locked`. Peak is more conservative and survives
   outliers better. Easy switch (~10 lines in `rtc.py`).
