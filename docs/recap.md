# RECAP — end-to-end usage guide

> **Target audience**: users who want to take a pi0.5 LoRA fine-tune from
> *behaviour cloning* to *advantage-conditioned* via RECAP, end-to-end,
> on FlashRT. Covers both PyTorch and JAX training paths and the
> CFG-guided inference path that consumes the trained checkpoint.
>
> **What RECAP is**: the recipe from the π\*0.6 paper
> ([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)). A small value
> function predicts time-to-completion for each frame, advantages are
> derived, and the policy is re-trained with a per-frame *advantage tag*
> (`"Advantage: positive"` / `"Advantage: negative"`) appended to the
> prompt at 70 % of training samples (the other 30 % stay unconditioned,
> per Appendix E). At inference the model runs twice per denoising
> step — once with the positive tag, once without — and the two
> velocities are combined via classifier-free guidance to bias the
> generated actions toward the high-advantage subset of the offline
> distribution.
>
> **What this doc is not**: a tutorial on training pi0.5 from scratch.
> Start with [`training/README.md`](../training/README.md) for that.
> This doc assumes you already know how to drive `Pi05Trainer.compile()`
> + a dataset to convergence and want the RECAP additions on top.

---

## 1. What ships in the repo today

RECAP is **already implemented and validated** in both training paths:

| Component | PyTorch | JAX |
|---|---|---|
| Value-function training (state-only, `StandaloneValueFunction`) | [`training/rl/train_value.py`](../training/rl/train_value.py) | [`training/jax/rl/train_value.py`](../training/jax/rl/train_value.py) |
| Value-function training (VLA, `Pi05ValueFunction` over frozen prefix) | [`training/rl/pi05_vf.py`](../training/rl/pi05_vf.py) | [`training/jax/rl/value_function.py`](../training/jax/rl/value_function.py) |
| ACP indicator annotation | [`training/rl/value_infer.py`](../training/rl/value_infer.py) | [`training/jax/rl/value_infer.py`](../training/jax/rl/value_infer.py) |
| One-shot RECAP iter (VF train + annotate) | [`training/rl/recap_iter.py`](../training/rl/recap_iter.py) | (call the JAX helpers directly) |
| ACP-conditioned policy training driver | [`training/rl/train_recap.py`](../training/rl/train_recap.py) | upstream `train_jax_lora_recap.py` + [`training/jax/scripts/run_baseline_with_fp8_patch.py`](../training/jax/scripts/run_baseline_with_fp8_patch.py) |
| LIBERO shim | [`training/rl/train_libero_recap.py`](../training/rl/train_libero_recap.py) | [`training/jax/rl/train_libero_recap.py`](../training/jax/rl/train_libero_recap.py) |
| LoRA merge → standalone safetensors / Orbax | [`training/rl/merge_lora.py`](../training/rl/merge_lora.py) | [`training/jax/merge_lora.py`](../training/jax/merge_lora.py) |
| CFG-guided inference (serial + fused B=2) | `Pi05TorchFrontendRtx` / `Pi05TorchFrontendThor` | `Pi05JaxFrontendRtx` / `Pi05JaxFrontendThor` |

Cross-language algorithm primitives — tag string format, advantage
math, soft VF loss, CFG sampler — live in
[`flash_rt/core/rl/`](../flash_rt/core/rl/) and are imported by both
stacks. Byte-equality is gated by the JAX path's parity tests (ACP hook
23/23 across 5 seeds × 4 dropouts, indicator-annotation Hamming
distance 2 / 12962 = 0.015 %).

### Validated end-to-end results

| Path | Dataset | Steps | Wall | Peak GPU | Loss first → last | NaN |
|---|---|---:|---:|---:|---|---:|
| PyTorch FP8+LoRA RECAP | LIBERO | 30 000 | 128.7 min | 14.65 GB | 0.044 → 0.012 | 0 |
| JAX FP8+LoRA RECAP | LIBERO | 30 000 | 166.75 min | 21.06 GB | 0.123 → 0.013 | 0 |

Source: [`training/README.md` § End-to-end training-step throughput](../training/README.md#end-to-end-training-step-throughput-rtx-5090-sm120-b4-lora_rank16)
and [`training/jax/README.md` § End-to-end training-step throughput](../training/jax/README.md#end-to-end-training-step-throughput-rtx-5090-sm120--b--4-lora_rank--16).

---

## 2. The four legs of a RECAP iteration

RECAP is **not a single training run** — it's a loop with four stages.
Each stage is one of the modules above, and all of them are reusable
across iterations.

```
                       ┌──────────────────────────┐
                       │  1. dataset (LeRobot v3) │
                       │   with per-frame state,  │
                       │   action, image, success │
                       └────────────┬─────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
    ┌─────────────────────────┐         ┌───────────────────────────────┐
    │  2. train_value         │         │  3. (optional) collect more   │
    │   • dense reward target │         │     rollouts with current     │
    │     from episode meta   │         │     policy → grow dataset for │
    │   • soft-CE 201-bin VF  │         │     next iter                 │
    └────────────┬────────────┘         └───────────────┬───────────────┘
                 ▼                                      │
    ┌─────────────────────────┐                         │
    │  4. value_infer         │                         │
    │   • V(o_t) per frame    │                         │
    │   • N-step advantage    │                         │
    │   • per-task ε_ℓ s.t.   │                         │
    │     ~30 % positive      │                         │
    │   • write acp_indicator │                         │
    └────────────┬────────────┘                         │
                 ▼                                      │
    ┌─────────────────────────┐                         │
    │  5. train_recap_policy  │                         │
    │   • ACP prompt hook     │                         │
    │     (70 % tagged,       │                         │
    │      30 % unconditioned)│                         │
    │   • flow-matching loss  │                         │
    │   • LoRA on FP8 base    │                         │
    └────────────┬────────────┘                         │
                 ▼                                      │
    ┌─────────────────────────┐                         │
    │  6. merge_lora_into_base│                         │
    │   • LoRA + base → fp32  │                         │
    │     merge → safetensors │                         │
    │     / Orbax             │                         │
    └────────────┬────────────┘                         │
                 ▼                                      │
    ┌─────────────────────────┐                         │
    │  7. CFG-guided infer    │─────────────────────────┘
    │   • Pi05*FrontendRtx    │
    │   • set_rl_mode(...)    │
    │   • v_uncond + β·(v_c   │
    │       − v_uncond)       │
    └─────────────────────────┘
```

The full multi-iter loop closes between step 7 and step 3 — you collect
new rollouts with the latest policy, fold them into the dataset, and
restart at step 2. The implementation today supports one iteration
out of the box; closing the loop is just a matter of writing new
parquet files between iters.

---

## 3. Prerequisites

### 3.1 Environment variables

The driver resolves dataset / checkpoint / tokenizer paths from
environment variables. These are the same vars used by the rest of the
training stack:

| Resource | Env var |
|---|---|
| pi0.5 PyTorch base ckpt (`model.safetensors` + `config.json` + `assets/`) | `FLASHVLA_PI05_CKPT_PYTORCH` |
| pi0.5 JAX (Orbax) base ckpt | `FLASHVLA_PI05_CKPT_JAX` |
| LeRobot v3 dataset root (LIBERO or your own) | `FLASHVLA_RECAP_DATASET` |
| PaliGemma SentencePiece tokenizer | `FLASHVLA_TOKENIZER_PATH` |
| Upstream openpi JAX `train_jax_lora_recap.py` | `FLASHVLA_JAX_BASELINE_SCRIPT` (JAX path only) |

A LeRobot v3 dataset is expected to follow this layout (image bytes
inline, per the v3 spec):

```
<root>/
    meta/info.json
    meta/tasks.parquet
    meta/episodes/chunk-{c:03d}/file-{f:03d}.parquet
    data/chunk-{c:03d}/file-{f:03d}.parquet
```

### 3.2 Dataset Protocol contract

For the PyTorch path, your dataset must satisfy the
[`RecapPolicyDataset`](../training/rl/dataset_protocol.py) Protocol
(policy training only) or
[`RecapMetadataDataset`](../training/rl/dataset_protocol.py) (policy +
VF training). LIBERO ships as the reference implementation; for any
other dataset, implement five methods (no inheritance — Python
`Protocol`):

```python
class MyDataset:
    def build_chunk_starts(self, action_horizon: int) -> np.ndarray: ...
    def has_acp_column(self) -> bool: ...
    def ensure_acp_indicators(self) -> np.ndarray: ...
    def get_frame(self, idx: int) -> RecapFrame: ...
    def get_action_chunk(self, idx: int, action_horizon: int) -> np.ndarray: ...
```

If you want the dataset to *also* drive the VF / annotation pipeline
(steps 2–4 above), extend with `RecapMetadataDataset`:

```python
class MyDataset:
    @property
    def num_frames(self) -> int: ...
    @property
    def state_dim(self) -> int: ...
    @property
    def episodes(self) -> list: ...           # objs with .length, .success, .task_index
    @property
    def episode_indices(self) -> np.ndarray: ...
    @property
    def frame_indices(self) -> np.ndarray: ...
    @property
    def task_indices(self) -> np.ndarray: ...
    @property
    def task_max_lengths(self) -> dict[int, int]: ...
    def ensure_state_action(self) -> tuple[np.ndarray, np.ndarray]: ...
```

### 3.3 Observation builder

The driver does not know about your camera layout — it asks the caller
for an `observation_builder` callable that turns a mini-batch's
decoded images + padded states into a pi0.5 `Observation`:

```python
def my_obs_builder(decoded_images, states_padded, *,
                   tokenized_prompt, tokenized_prompt_mask, device):
    return decoded_to_observation(
        decoded_images, states_padded,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        device=device,
    )
```

The LIBERO shim passes
[`training.rl.observation.decoded_to_observation`](../training/rl/observation.py)
which uses `LIBERO_CAMERA_MAP` (2-cam: `base_0_rgb` ← `observation.image`,
`left_wrist_0_rgb` ← `observation.wrist_image`, `right_wrist_0_rgb`
zero-filled). Pass a different `camera_map` if your dataset's cameras
are named or numbered differently.

---

## 4. PyTorch path — one iteration end to end

### 4.1 Annotate the dataset

If your dataset does **not** already ship per-frame `acp_indicator`
values, run one RECAP iteration to derive them:

```python
from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.recap_iter import run_recap_iter

dataset = LeRobotLiberoDataset("<your-libero-recap-dataset>")
result = run_recap_iter(
    dataset,
    num_steps=1_000,         # VF training steps
    seed=42,
    device="cuda",
)
indicators = result.annotation.indicators      # (num_frames,) int64
thresholds = result.annotation.thresholds      # {task_idx: ε_ℓ}
positive_ratio = result.positive_ratio_per_task
```

`run_recap_iter` runs the three sub-stages:

1. Build per-frame normalised value targets from
   `(episode_length, success, task_max_length)` via Eq. 5 of the paper
   (in [`flash_rt/core/rl/reward.py`](../flash_rt/core/rl/reward.py)).
2. Fit a `StandaloneValueFunction` (state-only, ~5 MB head) against the
   targets with the soft-cross-entropy loss in Eq. 1.
3. Predict `V(o_t)` over every frame, derive N-step advantages
   (default `n_step=50`), choose per-task ε_ℓ so ~30 % of frames are
   positive, and return the resulting `acp_indicator` array.

This is **cheap** (no images, just states; minutes on a single GPU)
and only needs to run once per dataset. The result can be cached: pass
it to the policy driver via `dataset.ensure_acp_indicators()` after
writing the indicators back to your dataset's storage.

For a stronger value function over the **VLA prefix embedding** (paper
§IV-A — uses image + language context instead of state only) use
`training.rl.pi05_vf.Pi05ValueFunction` with a frozen `PI0Pytorch`
backbone. The interface is the same `predict_value` callable that
`value_infer.annotate_with_value_function` consumes; substitute it for
the synthetic path when you want better label quality.

### 4.2 Run ACP-conditioned policy training

```python
from training.lora.inject import InjectionConfig
from training.rl.checkpoint import load_pi05_pretrained
from training.rl.lerobot_libero import LeRobotLiberoDataset
from training.rl.tokenizer import PaligemmaTokenizer
from training.rl.train_libero_recap import train_libero_recap
from training.rl.train_recap import RecapTrainConfig
from training.trainers.pi05_torch_trainer import Pi05Trainer

dataset = LeRobotLiberoDataset(os.environ["FLASHVLA_RECAP_DATASET"])
model = load_pi05_pretrained(
    os.environ["FLASHVLA_PI05_CKPT_PYTORCH"], action_horizon=10,
)
trainer = Pi05Trainer(model, device="cuda")
trainer.compile(InjectionConfig(encoder_rank=16, decoder_rank=16))
tokenizer = PaligemmaTokenizer(
    max_token_len=trainer.model.config.max_token_len, device=trainer.device,
)

cfg = RecapTrainConfig(
    num_steps=30_000,
    batch_size=4,
    lr=2.5e-5,
    acp_dropout=0.30,              # ← the only RECAP-specific knob
    use_acp=True,                  # ← turn ACP injection on
    dataloader_workers=2,          # production speed recipe
    compile_mode="reduce-overhead",
)
result = train_libero_recap(
    trainer, dataset, tokenizer,
    config=cfg,
    output_dir="<your-run-dir>",
    derive_acp_if_missing=True,    # auto-run step 4.1 if dataset lacks the column
)
print(
    f"final loss: {result.loss_history[-1]:.4f}, "
    f"peak GPU: {result.peak_memory_bytes / 1e9:.2f} GB, "
    f"wall: {result.seconds_total / 60:.1f} min, "
    f"LoRA saved to {result.final_lora_dir}",
)
```

#### What `use_acp=True` does that `use_acp=False` doesn't

A vanilla LoRA fine-tune (`use_acp=False`) trains on
`(observation, action_chunk)` pairs with a fixed prompt and the
flow-matching loss. The ACP-on path adds two steps per mini-batch:

1. Look up `acp_indicator[batch.starts]` and feed it to
   [`ACPPromptHook`](../training/rl/acp_hook.py). For each sample,
   with probability `acp_dropout` (default 0.30) the prompt is left
   unchanged; otherwise `"\nAdvantage: positive"` /
   `"\nAdvantage: negative"` is appended via
   [`build_acp_tagged_task`](../flash_rt/core/rl/acp_tags.py).
2. Re-tokenise the modified prompts and rebuild the `Observation`'s
   `tokenized_prompt` / `tokenized_prompt_mask` slots.

Everything else is identical: same AdamW (weight_decay=1e-10), same
warmup-cosine schedule (warmup=min(100, steps//30), end=peak·0.1),
same global-norm grad clip 1.0, same FP8 base GEMMs, same BF16 LoRA
backward path, same loss curve shape.

### 4.3 Merge LoRA → standalone safetensors

```python
from training.rl.merge_lora import merge_lora_into_base

merge_lora_into_base(
    base_dir=os.environ["FLASHVLA_PI05_CKPT_PYTORCH"],
    lora_dir=result.final_lora_dir,                   # e.g. <your-run-dir>/final
    output_dir="<your-merged-ckpt-dir>",
)
```

The merged directory mirrors the base layout (`config.json`,
`policy_postprocessor.json`, `policy_preprocessor.json`, `assets/`,
`model.safetensors`). The merge math runs in fp32 (`W_merged = W_base
+ scaling * B @ A`) so there is no bf16-intermediate rounding bias.

### 4.4 Serve via FP8 inference with CFG enabled

```python
from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtx

pipe = Pi05TorchFrontendRtx(
    "<your-merged-ckpt-dir>", num_views=2, autotune=3,
)

# Recommended for RL: enable fused B=2 CFG BEFORE set_prompt. The RTX
# frontend's set_batched_mode is B=2-hardcoded; the Thor frontend takes
# an explicit ``batch_size=2`` argument.
pipe.set_batched_mode(enable=True)
pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.5, advantage_positive=True)
pipe.set_prompt("pick up the red block")

# First infer call lazy-calibrates FP8 scales against the conditioned
# prompt and (because batched) recaptures the B=2 graph. Subsequent
# calls hit the cached graph.
actions = pipe.infer(obs)["actions"]    # (chunk_size, action_dim)
```

The CFG semantics are documented in detail in
[`docs/rl_inference.md`](rl_inference.md), including the per-step
combine formula `x_{k+1} = x_k + v_uncond + β·(v_cond − v_uncond)`,
the four-frontend coverage table, and measured latencies (25.9 ms
fused on RTX 5090 vs ~67 ms on Thor SM110, both at β=1.5).

---

## 5. JAX path — one iteration end to end

The JAX path uses the upstream openpi `train_jax_lora_recap.py` driver
verbatim (so loss curves match the openpi baseline 1-for-1) plus
FlashRT's FP8 monkey-patch that re-routes `lora.Einsum` and
`lora.FeedForward._dot` matmuls through the cuBLASLt FP8 XLA FFI.

### 5.1 Annotate the dataset (JAX)

```python
from training.jax.rl.value_infer import annotate_with_value_function
from training.jax.rl.train_value import train_value
from training.jax.rl.value_function import StandaloneValueFunction
```

Same Protocol as the PyTorch path — the JAX path uses the same
[`flash_rt/core/rl/`](../flash_rt/core/rl/) primitives for the
advantage math and tag strings, so the indicator outputs are byte-
equal to the PyTorch path on the same data (gated by
`tests/test_acp_hook_parity.py`, 23/23 cases across 5 seeds × 4
dropouts).

### 5.2 Train the policy with FP8 + LoRA + ACP

The patch installer is opt-in via env var and the wrapper script:

```bash
python -m training.jax.scripts.run_baseline_with_fp8_patch \
    --baseline-script $FLASHVLA_JAX_BASELINE_SCRIPT \
    --checkpoint_path $FLASHVLA_PI05_CKPT_JAX \
    --dataset_root    $FLASHVLA_RECAP_DATASET \
    --output_dir      <your-run-dir> \
    --steps 30000 --batch_size 4 --lr 2.5e-5 \
    --lora_rank 16 --acp_dropout 0.30 --log_freq 50
```

Turn the FP8 patch off via `FLASHVLA_JAX_FP8=0` — the wrapper still
launches the upstream driver, but every matmul goes through
`jnp.einsum` / `jnp.dot`. Use this for an A/B reference run if you
suspect the FP8 path. Run-to-run difference at small batch is small;
[`training/jax/README.md` § Memory and throughput notes](../training/jax/README.md#memory-and-throughput-notes)
explains why and when you'd expect to see a larger gap (B≥16 is where
the FP8 win grows).

`JAX_PLATFORMS=cuda` is required when the JAX install ships the
experimental `mlir_tensorrt` plugin (its clustering pass rejects XLA
FFI custom calls). The wrapper does not set this for you — pin
explicitly.

### 5.3 Merge LoRA — Orbax in, Orbax out

```python
from training.jax.merge_lora import merge_lora_into_base

merge_lora_into_base(
    trained_orbax_dir="<your-run-dir>/step_030000/params",
    output_orbax_dir="<your-merged-dir>/params",
    scaling=1.0,           # openpi pi05 default (alpha == rank)
    overwrite=False,
)
```

No PyTorch hop. The merged directory drops into any JAX pi0.5
consumer, including FlashRT's `Pi05JaxFrontendRtx`.

### 5.4 Serve via JAX FP8 inference with CFG

Same `set_rl_mode` API as the PyTorch frontend — inherits from the
torch frontend:

```python
from flash_rt.frontends.jax.pi05_rtx import Pi05JaxFrontendRtx

pipe = Pi05JaxFrontendRtx(
    "<your-merged-dir>", num_views=2, autotune=3,
)
pipe.set_batched_mode(enable=True)   # B=2 hardcoded on RTX, same as torch frontend
pipe.set_rl_mode(cfg_enable=True, cfg_beta=1.5, advantage_positive=True)
pipe.set_prompt("pick up the red block")
actions = pipe.infer(obs)["actions"]
```

The norm_stats file must sit at
`<merged-dir>/assets/physical-intelligence/libero/norm_stats.json`
(symlink from the base ckpt). The JAX frontend uses the same numerical
path as the PyTorch one — both inherit the CFG sampler from
[`flash_rt/core/rl/cfg_sampler.py`](../flash_rt/core/rl/cfg_sampler.py)
so the conditioned and unconditioned velocity combine is bit-equal
across stacks.

---

## 6. Choosing β at inference time

`cfg_beta` is the guidance strength. The paper recommends `[1.5, 2.5]`
for π\*0.6-style fine-tunes. Some notes from the FlashRT validation
runs:

| β | Effect | Use when |
|---|---|---|
| 1.0 | Mathematically collapses to cond-only output (`v_uncond + 1·(v_cond − v_uncond) = v_cond`). Latency is still the 2× CFG cost — wasteful in production. | Numerical sanity-check only. Prefer turning CFG off entirely (`cfg_enable=False`) for unconditioned inference. |
| 1.5 | Default; mild guidance. | Most fine-tunes. The 30k LIBERO runs validate at this point. |
| 2.0 – 2.5 | Stronger guidance. | Tasks where the policy mode-collapses without it; or when the ACP indicator labels were noisy. |
| > 2.5 | Aggressive extrapolation outside the cond / uncond span. | Rarely; can produce out-of-distribution actions. |

β does **not** affect latency — it is a scalar multiplier inside the
`cfg_combine_into_residual` kernel only. Numerically the kernel is
bit-equal to the FP32 reference at production size (max abs diff = 0,
cos = 1.0 — see [`docs/rl_inference.md` § Numerical contract](rl_inference.md#numerical-contract)).

### `advantage_positive=False`

Set this only for debugging — it swaps the conditioned slot to
`"Advantage: negative"`. The CFG combine then pushes the policy *away*
from the negative-advantage subset, which in practice should look
similar to `advantage_positive=True` for a well-balanced 30 % positive
ratio. If the two settings produce noticeably different actions, the
underlying VF is biased or the per-task ε_ℓ thresholds are off — go
back to step 4.1 and inspect `positive_ratio_per_task`.

---

## 7. Common variations

### 7.1 Vanilla LoRA fine-tune via the same driver

Set `use_acp=False`:

```python
cfg = RecapTrainConfig(
    num_steps=30_000,
    batch_size=4,
    lr=2.5e-5,
    use_acp=False,             # ← skip ACP injection entirely
    dataloader_workers=2,
    compile_mode="reduce-overhead",
)
```

No `acp_indicator` lookup, no `ACPPromptHook`, no per-step prompt
re-tokenisation. The flow-matching loss + AdamW + LR schedule + LoRA
adapters are exactly the same. This is also a useful **A/B baseline**
for evaluating a RECAP run — train the same dataset twice, once with
`use_acp=True` and once with `False`, and compare downstream task
success.

### 7.2 Custom dataset (non-LIBERO)

The driver is dataset-agnostic — see
[`training/README.md` § Train on your own dataset](../training/README.md#train-on-your-own-dataset)
for the five-method Protocol. The only RECAP-specific work on top of
that is implementing `has_acp_column` / `ensure_acp_indicators` (or
relying on `derive_acp_if_missing=True` to run §4.1 for you on the
first training step).

### 7.3 Custom camera / state schema

Pass a `camera_map` to your `observation_builder`:

```python
MY_CAMERA_MAP = {
    "base_0_rgb":         "front_cam",
    "left_wrist_0_rgb":   "left_wrist_cam",
    "right_wrist_0_rgb":  "right_wrist_cam",   # or None for 2-cam
}

def my_obs_builder(decoded_images, states_padded, *,
                   tokenized_prompt, tokenized_prompt_mask, device):
    return decoded_to_observation(
        decoded_images, states_padded,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        device=device,
    )

result = train_recap_policy(
    trainer, MyDataset(), tokenizer, my_obs_builder,
    config=cfg, output_dir="<your-run-dir>",
)
```

For state dimensions ≠ 32, pad to `PI05_STATE_DIM = 32` in your
dataset's `get_frame()` — the openpi pi0.5 architecture is fixed at
32-dim state. See
[`training/rl/observation.py`](../training/rl/observation.py)
for the LIBERO 8-dim → 32-dim padding example.

### 7.4 Multi-iter loop (collect → label → train, repeat)

The iter-1 path described in §4 is one pass through the loop. To
close it:

1. After step 4.4, run the policy in your environment to collect new
   rollouts.
2. Append the rollouts to your dataset's parquet (LeRobot v3 layout).
3. Run §4.1 again on the **combined** dataset to refresh the
   indicators. Per Appendix F of the paper, you re-fit the VF on the
   bigger dataset so the per-task ε_ℓ adapts to the new mixture.
4. Re-run §4.2 with the new indicators. You can either restart LoRA
   training from scratch or warm-start from the previous iter's LoRA
   (load via `load_lora_state(trainer, prev_dir)` before `train`).

The shipped code supports one iter cleanly. The multi-iter outer loop
is currently the caller's responsibility — no driver in the repo
loops between rollout collection and training because the rollout
side depends on the user's simulator / robot stack.

---

## 8. Validation: how to know it's working

A successful RECAP fine-tune should show:

1. **Training-side**:
   - **Loss curve**: similar shape to the non-RECAP baseline. The 30k
     LIBERO RECAP run lands at loss 0.012 (PyTorch) / 0.013 (JAX) — see
     the table in §1. If your loss is > 2× the non-RECAP baseline at
     the same step count, the ACP injection is likely too aggressive
     or the indicators are noisy.
   - **Positive ratio**: `result.positive_ratio_per_task` from
     `run_recap_iter` should be roughly 0.30 per task. Drift > ±0.10
     means the per-task ε_ℓ search didn't converge — check that your
     dataset has enough episodes per task (≥ 30 recommended).
2. **Inference-side numerical**:
   - `cfg_beta=1.0` collapse: `cos(CFG output, cond_only output) ≥ 0.999`.
     If this fails, the CFG path is broken — file an issue.
   - B=2 slot symmetry: same observation in both CFG slots should give
     `cos(slot 0, slot 1) = 1.0`. Already validated in the shipped
     tests but worth re-checking on your hardware.
3. **Inference-side task-level**:
   - Compare success rate on your eval set across three settings:
     `cfg_enable=False` (no CFG), `cfg_beta=1.0` (collapsed CFG —
     same as cond-only, 2× compute), and `cfg_beta=1.5`. RECAP is
     working if `cfg_beta=1.5` beats `cfg_enable=False` by a margin
     larger than your run-to-run variance.
4. **`advantage_positive=True` vs `False` separation**:
   - Run both on the same observations and check that the action
     distributions are visibly different on the high-advantage axes
     (per-task; ACP can be a small effect, expect ~5–15 % per-action
     deltas).

The unit tests in `training/tests/` and `training/jax/tests/` (both
gitignored, per-developer local validation) cover the algorithm-level
parity gates; once they pass, run a 1k-step smoke training before
committing to a 30k run.

---

## 9. What's not yet supported

- **FP8 backward** — the LoRA grad currently flows through a
  straight-through estimator on the FP8 base GEMM, backward stays in
  BF16. See
  [`training/README.md` § Why FP8 backward is not implemented](../training/README.md#why-fp8-backward-is-not-implemented)
  for the trade-off analysis and conditions under which it would be
  worth implementing.
- **Generic batched CFG inference at B > 2** — useful for parallel RL
  rollout collection. The fused B=2 path is shipped; B > 2 needs a new
  graph shape and per-slot prompt staging.
- **Closed-loop multi-iter driver** — §7.4. The single-iter primitives
  are all in place; the outer loop is one shell script away but is
  intentionally out of scope for the shipped code (rollout collection
  varies per user).
- **Pi0.6 paper architecture** (`gemma_860m` action expert,
  `gemma_670m` value function VLM, Gemma 3 4B backbone). RECAP-on-pi0.5
  is what's validated. Porting to the larger Pi0.6 architecture is
  ~the work of adding one new model — see
  [`training/README.md` § Adapting to a new VLA](../training/README.md#adapting-to-a-new-vla).

---

## 10. Cross-reference

| To learn more about… | Read |
|---|---|
| The training stack overall (FP8 LoRA, throughput, memory recipes) | [`training/README.md`](../training/README.md) |
| The JAX-specific path (FP8 patch, openpi baseline wrapping) | [`training/jax/README.md`](../training/jax/README.md) |
| The inference-side CFG sampler (per-step combine, fused B=2 path, latency table) | [`docs/rl_inference.md`](rl_inference.md) |
| FP8 calibration mechanics that the merged LoRA checkpoint needs | [`docs/calibration.md`](calibration.md) |
| The cross-language algorithm primitives (tag strings, advantage math, soft VF loss, CFG combine) | [`flash_rt/core/rl/`](../flash_rt/core/rl/) |
| Adding a brand-new model (not pi0.5) | [`docs/adding_new_model.md`](adding_new_model.md) |

---

## Appendix A — paper ↔ code mapping

For users coming from the π\*0.6 paper
([arXiv:2511.14759](https://arxiv.org/abs/2511.14759)), here's where
each piece lands:

| Paper | Code |
|---|---|
| §IV-A: distributional value function (201 bins, soft-CE loss) | [`flash_rt/core/rl/value_function.py`](../flash_rt/core/rl/value_function.py) — `ValueFunctionHead`, `StandaloneValueFunction`. Helpers in [`flash_rt/core/rl/reward.py`](../flash_rt/core/rl/reward.py): `compute_soft_value_loss`, `expected_value_from_logits`. |
| §V-B: ACP prompt format | [`flash_rt/core/rl/acp_tags.py`](../flash_rt/core/rl/acp_tags.py) — `build_acp_tagged_task` / `build_unconditioned_task`. Byte-equal across PyTorch and JAX. |
| Appendix E: 30 % advantage dropout | [`training/rl/acp_hook.py`](../training/rl/acp_hook.py) — `ACPPromptHook(dropout=0.30)`. JAX mirror: [`training/jax/rl/acp_hook.py`](../training/jax/rl/acp_hook.py). |
| Appendix F: N-step advantage, per-task ε_ℓ thresholds | [`flash_rt/core/rl/advantage.py`](../flash_rt/core/rl/advantage.py) — `compute_nstep_advantages`, `compute_per_task_thresholds`, `binarize_advantages`. |
| Eq. 5: normalised episode value targets | [`flash_rt/core/rl/reward.py`](../flash_rt/core/rl/reward.py) — `compute_episode_value_targets`. |
| Test-time CFG (Eq. 2: `v_guided = v_uncond + β(v_cond − v_uncond)`) | [`flash_rt/core/rl/cfg_sampler.py`](../flash_rt/core/rl/cfg_sampler.py) — combine math. CUDA kernel: `cfg_combine_into_residual` (RTX), `decoder_forward_b2(cfg_beta=...)` (Thor). |
| §IV-A: VF over frozen VLA prefix embedding | [`training/rl/pi05_vf.py`](../training/rl/pi05_vf.py) — `Pi05ValueFunction` (PyTorch). JAX mirror: [`training/jax/rl/value_function.py`](../training/jax/rl/value_function.py). |
