# RTC-lite Runtime Design

RTC-lite is an execution-layer wrapper for action chunk policies. It does not
change model weights, denoising math, CUDA kernels, or calibration. The goal is
to keep a foreground controller supplied with actions while the model generates
the next action chunk in a background worker.

## Scope

RTC-lite is intended for policies with this contract:

```text
observation -> action_chunk[horizon, action_dim]
```

Examples:

- Motus-style models: `infer(...) -> (frames, actions)`
- Pi0/Pi0.5-style models: `infer(observation) -> {"actions": actions}`

The runtime does not know about image preprocessing, prompt setup, CUDA Graph
capture, or robot transport. Those remain in the model frontend and deployment
layer.

## Components

`flash_rt.runtime.rtc.ActionChunkAdapter`

Minimal model adapter with one method:

```python
infer_actions(observation) -> np.ndarray  # [horizon, action_dim]
```

`flash_rt.runtime.rtc.CallablePolicyAdapter`

Small helper for existing frontends. It can extract actions from either a dict
return value or a tuple return value.

`flash_rt.runtime.rtc.AsyncChunkRunner`

Owns one model worker thread. The foreground loop calls `next_action()` once per
controller tick. The first call blocks to initialize the first chunk. Later
calls consume actions from the current chunk and submit the next chunk before
the current chunk is exhausted.

`flash_rt.runtime.rtc.RTCStats`

Tracks chunk starts/completions, action count, deadline misses, held actions,
and model latency.

## Execution Model

```text
foreground control loop:
  obs = latest_robot_observation()
  action = runner.next_action(obs)
  robot.step(action)

background worker:
  actions = adapter.infer_actions(latest_obs)
```

The foreground loop owns timing. RTC-lite does not sleep, run robot IO, or
change the frontend's CUDA stream policy.

## Scheduling Knobs

RTC-lite exposes the small set of scheduling knobs that the
"Real-Time Execution of Action Chunking Flow Policies" paper
(Black et al. 2025, arXiv:2506.07339) calls out, plus a couple of
deployment-engineering knobs.

`start_next_at = 0` (recommended for production)

Fire the next background inference as soon as the previous one
completes. Capped naturally by the single-worker executor: only one
inference in flight at a time, so the actual rate equals
`1 / inference_latency`. This gives the freshest possible plan at
every chunk swap. The legacy default
(`max(1, horizon // 2)`) is preserved by leaving `start_next_at = None`.

`inference_delay_steps` (= "splice at d" in the paper)

The number of control ticks the foreground loop consumes during one
inference. When a fresh chunk lands, the runner skips past
`new_chunk[:d]` (those actions correspond to time we already lived
through serving the old chunk) and serves `new_chunk[d]` first. This
is the paper's "executed prefix" handling. Without it, the first
action of every new chunk is a lookback by `d` ticks, which the PD
controller then has to wrench into the present, producing the visible
seam jerk we are trying to eliminate.

`auto_inference_delay = True`

EMA-track the measured inference latency
(`stats.ema_latency_s`) and recompute `d = ceil(ema * target_hz)` on
every swap. Adapts to drift across the run (warmup, GPU thermal
throttle, server load). Overrides `inference_delay_steps` once
populated; the explicit value is used only as the seed for the very
first swap before the EMA has samples.

`blend_steps` — seam smoothing (NEW SEMANTICS as of 2026-05)

Linear alpha ramp at the START of each freshly-promoted chunk. With
`N = blend_steps` and `k = blend_step` (0-indexed), the emitted
action is `alpha * raw + (1 - alpha) * anchor` where
`alpha = (k + 1) / (N + 1)`. So the first emitted action is mostly
the previous target (smoothest), the `N`-th is almost the raw new
target, and from `k = N` onward we serve the raw new chunk. The
anchor is a snapshot of `last_served_action` at the moment of the
swap, so multiple promotions during a single blend window do not
corrupt the ramp.

`tail_blend_steps` — deadline-miss damping (legacy)

The OLD `blend_steps` semantics, kept under a different name for the
deadline-miss path: when the chunk is about to run out and no
replacement is ready, the last `tail_blend_steps` actions are pulled
toward `last_served_action` so the PD controller does not jerk on
the held-target. Defaults to `0`.

`miss_policy = "hold_last"`

If the next chunk is not ready when the current chunk is exhausted,
repeat the last action and count a deadline miss. Non-blocking; the
standard production setting.

`miss_policy = "block"`

Block until the model returns the next chunk. Useful for offline
diagnostics or the sync-baseline mode in `ChunkedWebsocketClient`
(`mode = 1`); reintroduces pauses and is not the production default.

## Mode catalogue (ChunkedWebsocketClient)

The `flash_rt.serving.ChunkedWebsocketClient` wrapper exposes the
above knobs through a single integer mode:

| mode | scheduling | splice | seam blend | use |
|---|---|---|---|---|
| 1 | sync, k=5 truncate-replan, block on infer | n/a | n/a | baseline A/B |
| 2 | async, fire-ASAP, splice at `d` (auto) | yes | 0 | debug raw policy |
| 3 (default) | async, fire-ASAP, splice at `d` (auto) | yes | 3 | smooth, low blend cost |
| 4 | async, fire-ASAP, splice at `d` (auto) | yes | 5 | extra-smooth |
| 5 (future) | async + server-side RTC inpainting | n/a | n/a | smoothness via prefix attention guidance |

Mode 5 will replace client-side seam blending with the paper's
prefix-attention guidance ("ΠGDM" in the paper): the server receives
the unexecuted suffix of the previous chunk and inpaints the first
`d` actions of the new chunk to match what was actually executed.
That removes the seam by construction (no client-side smoothing
required). It needs server-side support which is not yet shipped.

## What This Is Not

RTC-lite does not implement a new policy or train-time chunking method. It is an
inference scheduling layer. It validates action supply at a fixed controller
rate; robot task success still has to be measured in the target environment.
