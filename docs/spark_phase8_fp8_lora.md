# Phase 8 — FP8 + runtime-LoRA parity (G12)

## TL;DR

Before G12, the FP8 Pi0.5 decoder pipeline silently dropped **every**
runtime-LoRA contribution when `FLASHRT_RUNTIME_LORA=all` was set.
Result: with `--fp8` + LoRA-finetuned checkpoint (pi05_openarm_ngc_lora_v4),
the decoder ran with LoRA-stripped base weights and the policy
produced near-random actions. G12 ports the encoder's already-working
"force non-fused FP8 when LoRA on + apply LoRA via `bf16_nn_res`"
pattern to all four FP8 decoder GEMM sites (QKV, attn O, FFN gate/up,
FFN down).

## The bug

When `FLASHRT_RUNTIME_LORA` is set to `all` (the production default
since G10), the JAX converter extracts every LoRA pair out of the base
checkpoint instead of merging it. The resulting `ckpt` dict has:

* base weights **without** LoRA contribution
* `*_lora_a` / `*_lora_b` tensors stored alongside the base

At inference time the pipeline must perform the LoRA two-matmul
`out += (x @ la) @ lb` on top of every base GEMM whose weights had
LoRA extracted, or the model is missing the entire fine-tune delta.

### Encoder (working before G12)

`Pi05Pipeline.transformer_encoder` had:

```python
_enc_lora_on = (self._has_enc_ffn_gateup_lora
                or self._has_enc_ffn_gateup_lora_fused
                or self._has_enc_ffn_down_lora
                or self._has_enc_attn_lora)
fused = use_fp8 and self.fp8_calibrated and not _enc_lora_on
```

The `and not _enc_lora_on` forced `fused = False` whenever LoRA was
active, sending the layer through the `elif self.use_fp8:` branches
that wrote BF16 intermediates of `encoder_x_norm` / `encoder_hidden`.
Each of those branches then called `_apply_enc_lora(...)` to add the
LoRA delta in BF16 on top of the FP8 base GEMM output via
`bf16_nn_res` (fused residual GEMM in FP32 accumulator). Cost was the
small encoder-layer epilogue fusions, not the FP8 GEMMs themselves.

### Decoder (broken before G12)

`Pi05Pipeline.transformer_decoder` had:

```python
fused = self.use_fp8_decoder and self.fp8_calibrated   # ← no LoRA check!
```

So when runtime LoRA was on and FP8 was calibrated, the decoder ran
through `if fused:` branches that did not contain any LoRA add at
all. Worse, even the non-fused `elif self.use_fp8_decoder:` branches
that the BF16-only path uses for unfused execution had no LoRA call
(only the BF16 `else:` branches did). So *no FP8 decoder path*
applied runtime LoRA.

With `FLASHRT_RUNTIME_LORA=all` and `--fp8`:

* encoder LoRA: applied correctly via the existing wiring
* decoder LoRA: silently dropped — base decoder weights are
  LoRA-stripped, so the gemma_300m action expert effectively rolled
  back to the pre-fine-tune behaviour

On the OpenArm v4 chocolate_bars checkpoint, the symptom was a
"catastrophic" cosine collapse — the action chunks no longer matched
the trained policy, and the robot client either failed to reach for
the bars or triggered a SAFETY STOP on the first chunk-to-chunk
discontinuity.

## The fix (G12)

Four changes, all in `flash_rt/`:

1. **JAX converter** (`frontends/jax/pi05_rtx.py`) builds the fused
   `(D, 2r)` / `(2r, 2H)` block-diagonal `decoder_ffn_gateup_lora_{a,b}`
   tensors, mirroring the encoder's existing fused gateup form. Same
   `_build_padded_gateup_lora` helper; the math is identical.

2. **Pipeline `__init__`** (`models/pi05/pipeline_rtx.py`) detects
   `_has_dec_ffn_gateup_lora_fused` and widens `dec_max_neck` to `2r`
   so the shared `_dec_lora_neck` scratch can hold the fused
   intermediate.

3. **Pipeline decoder** computes `_dec_lora_on` (same shape as the
   encoder's `_enc_lora_on`), uses it to force `fused = False` when
   LoRA is active, and adds an `_apply_dec_lora(...)` call in each of
   the four non-fused FP8 decoder branches:
   - QKV (input `x_normed_buf`, output `decoder_QKV`, padded form)
   - attn O (input `dec_o_ptr`, output `x_normed_buf`)
   - FFN gate/up (input `x_normed_buf`, output `decoder_gate_merged`,
     fused gateup form)
   - FFN down (input `decoder_hidden` post-geglu, output `x_normed_buf`)

4. **Pipeline `transformer_decoder`** outer loop applies the same
   `_dec_lora_on` gate so `skip_c1` is correct (the next layer's
   AdaRMSNorm-style C1 fires independently because the previous
   layer's fused C7→C1_next path is now skipped).

The `_apply_dec_lora` helper already existed — it does `out += (x @ la) @ lb`
via two BF16 GEMMs through the per-layer `_dec_lora_neck` scratch,
with the second GEMM fusing the residual add into the FP32
accumulator. Same precision contract as the encoder.

## Perf impact

Disabling fused FP8 on the decoder costs the epilogue fusions
(`ada_rms_norm_style_fp8`, `gate_residual_ada_norm_fp8`,
`gate_geglu_merged_fp8`) but the **18 layers × 10 diffusion steps × 4
matmuls per layer** of base FP8 GEMMs still run. The four
`_apply_dec_lora` adds are ~2 BF16 matmuls each (a `(ds, D) @ (D, r)`
followed by a `(ds, r) @ (r, D)`-sized residual). On Pi0.5 OpenArm
with `ds = chunk_size = 10`, `D = 1024`, `H = 4096`, `r = 32`, the
additional FLOPs per layer per step are roughly:

* QKV LoRA: `10 × 1024 × 160 + 10 × 160 × 2560 ≈ 5.7 MFLOPs`
* attn O LoRA: `10 × 1024 × 32 + 10 × 32 × 1024 ≈ 0.7 MFLOPs`
* FFN gateup fused LoRA: `10 × 1024 × 64 + 10 × 64 × 8192 ≈ 5.9 MFLOPs`
* FFN down LoRA: `10 × 4096 × 32 + 10 × 32 × 1024 ≈ 1.6 MFLOPs`

Total per step ≈ 14 MFLOPs × 18 layers ≈ 250 MFLOPs per Euler step,
versus the ~10 GFLOPs of the FP8 base path per step — about 2.5%
extra FLOPs on what is a memory-bound phase anyway. Expected wall-clock
hit ≤ 5 ms per inference; the BF16 fallback would have cost ~50 ms.

## Validation

### Unit tests (no GPU needed)

```bash
cd ~/sparkpack/FlashRT && source .venv/bin/activate
python -m pytest tests/test_fp8_lora_decoder_wiring.py -v
```

9 cases, ~0.8 s. Covers:

* algebraic correctness of the fused `(D, 2r) / (2r, 2H)` decoder
  gateup LoRA on the real Pi0.5 dimensions (D=1024, H=4096, r=32)
* zero-cross-block invariant of the block-diagonal `lb_padded`
* source-inspection gate that `_apply_dec_lora` is invoked in each of
  the four FP8 decoder branches
* pipeline init flag detection + neck-buffer widening

These tests are a regression gate. If a future edit accidentally
removes one of the LoRA-on-FP8 sites, the test fails loudly with a
message pointing at the dropped LoRA.

### In-process parity (CUDA + checkpoint)

`scripts/spark_phase8_fp8_lora_parity.py` runs the same observations
through two FlashRT instances (BF16 reference + FP8 SUT) with
deterministic noise per-sample and compares per-step cosine, L2 ratio,
and max joint-step delta. See the script header for the three-step
invocation (`--mode bf16` → `--mode fp8` → `--compare`).

Acceptance gates (G12 should pass all three on the OpenArm v4 LoRA
checkpoint at 20+ samples):

| Metric | Threshold | Pre-G12 (broken) | Post-G12 (expected) |
|---|---|---|---|
| per-sample cosine min | ≥ 0.99 | ~0.55-0.65 | ≥ 0.99 |
| per-sample L2 ratio | in [0.95, 1.05] | 0.4-1.6 | in [0.95, 1.05] |
| max abs joint-step diff | ≤ 0.10 rad on ≥ 90 % | > 0.5 rad most | ≤ 0.10 rad on ≥ 90 % |

### On-robot regression test

After the parity gate passes:

1. Launch FP8 server with default args (`--runtime-lora=all`,
   `--fp8` implicit, prewarm prompt lens for OpenArm v4):
   ```bash
   PYTHONPATH=~/sparkpack/openpi/src:~/sparkpack/openpi/packages/openpi-client/src \\
   FLASHRT_ROBOT_ACTION_DIM=16 \\
   python scripts/serve_policy_flashrt.py \\
       --checkpoint ~/sparkpack/openpi/checkpoints/pi05_openarm_ngc_lora_v4/chocolate_bars_pi05/29999 \\
       --robot-action-dim 16 --num-views 3 \\
       --calib-data /tmp/calib_openarm_v4_80.npz \\
       --default-prompt "put the chocolate bars in the container" \\
       --prewarm-prompt-lens 70-85 \\
       --port 8002
   ```
2. Drive the robot at chunk_size=10 mode 1 (synchronous, with blending).
   The pre-G6 shoulder bias should stay absent (G10 default is still
   honored), and the chunk-to-chunk continuity should match the BF16
   server.
3. If mode 1 looks clean, switch to mode 5 (soft-guidance RTC from G11).

## Future: lerobot integration

The lerobot upstream (cloned at `~/sparkpack/lerobot`, tag `v0.5.1`)
ships its own Pi0.5 policy + RTC soft-guidance + native OpenArm
driver. The eventual integration plan is to wrap FlashRT as a
quantized remote inference engine that the lerobot client orchestrates,
keeping all the trajectory bookkeeping (RTC, chunking, async
prefetch) on the lerobot side. The G12 fix is a precondition because
without decoder runtime LoRA on FP8, FlashRT cannot serve any LoRA-
finetuned Pi0.5 checkpoint at the speeds that motivated the
integration.

The contract for the lerobot wrapper is:

* FlashRT exposes BF16 reference inference and FP8 production
  inference behind the same `model.predict(...)` API.
* Both modes apply runtime LoRA correctly (G12 closes the FP8 side).
* The wrapper sends raw observations (state + images + prompt) and
  receives `actions: (chunk, robot_action_dim)` joint-radians, plus
  optionally `_rtc_chunk_model_space` for RTC bookkeeping.
* lerobot supplies its own RTC client (`RTCProcessor`-based), so
  FlashRT's server-side soft-guidance from G11 is the **fallback**
  for non-lerobot clients (the openpi websocket policy server).
