## What this directory contains

Vendored read-only reference implementation of Real-Time Chunking
(RTC) from external projects, kept here so the FlashRT port can be
parity-checked against them without depending on those repos at
runtime. None of this code is imported by FlashRT — it is purely a
specification.

### Files

| File | Origin | Why we keep it |
|---|---|---|
| `modeling_rtc.py` | `huggingface/lerobot/src/lerobot/policies/rtc/modeling_rtc.py` | The `RTCProcessor.denoise_step` method. This is the algorithm we are porting. The autograd block at lines 212-219 is the entire mathematical contract. |
| `configuration_rtc.py` | `huggingface/lerobot/src/lerobot/policies/rtc/configuration_rtc.py` | `RTCConfig` with defaults (`prefix_attention_schedule=LINEAR`, `max_guidance_weight=10.0`, `execution_horizon=10`). Mode-5 defaults in FlashRT will mirror these. |
| `action_queue.py` | `huggingface/lerobot/src/lerobot/policies/rtc/action_queue.py` | The `ActionQueue.merge` and `get_left_over` semantics — defines what `prev_chunk_left_over` actually contains (the unconsumed actions of the previous chunk, NOT a slice from somewhere in the middle). |
| `latency_tracker.py` | `huggingface/lerobot/src/lerobot/policies/rtc/latency_tracker.py` | `inference_delay = ceil(latency_tracker.max() / time_per_chunk)` — they use peak latency, not EMA. FlashRT currently uses EMA via `_record_latency_locked`; that's a known divergence. |
| `lerobot_modeling_pi05.py` | `huggingface/lerobot/src/lerobot/policies/pi05/modeling_pi05.py` | The integration point. Lines 836-869 show exactly how the Pi0.5 Euler loop wraps each `denoise_step_partial_call` with `rtc_processor.denoise_step`. This is the structural template for `Pi05Pipeline.transformer_decoder`'s port. |
| `lerobot_modeling_pi0.py` | `huggingface/lerobot/src/lerobot/policies/pi0/modeling_pi0.py` | Same as above but for Pi0 (no state-in-prompt). Useful as a cross-check for the Euler-loop structure. |
| `eval_with_real_robot.py` | `huggingface/lerobot/examples/rtc/eval_with_real_robot.py` (branch `0db5f66d`) | The reference client: how to call `policy.predict_action_chunk(obs, inference_delay=, prev_chunk_left_over=)` and feed the result into `ActionQueue.merge`. The threading model FlashRT's `ChunkedWebsocketClient` should mirror. |
| `rtc.mdx` | `huggingface/lerobot/docs/source/rtc.mdx` | Public docs explaining the RTC config knobs and the prefix-attention schedule rationale. |
| `kinetix_model.py` | `Physical-Intelligence/real-time-chunking-kinetix/src/model.py` | The canonical RTC paper implementation. `realtime_action` at line 219 is the JAX original that `modeling_rtc.py` is a port of. Useful when the lerobot code is unclear (e.g. on the time-convention sign). |
| `kinetix_eval_flow.py` | `Physical-Intelligence/real-time-chunking-kinetix/src/eval_flow.py` | The reference evaluation loop. Confirms the splice semantics: the new chunk REPLACES from index 0, and the first `inference_delay` positions are guided to match the inflight prefix. |

### How to use

Look at `modeling_rtc.py` first — the entire algorithm fits in a screen. Then
look at `lerobot_modeling_pi05.py:836-869` to see how it's plugged into the
Euler loop. Then look at `eval_with_real_robot.py:265-330` to see how
`prev_chunk_left_over` is constructed on the client side. The FlashRT port
needs to match all three.

### Updating

To refresh from upstream:

```bash
cd third_party/lerobot_rtc_reference
for entry in \
    "modeling_rtc.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/rtc/modeling_rtc.py" \
    "action_queue.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/rtc/action_queue.py" \
    "configuration_rtc.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/rtc/configuration_rtc.py" \
    "latency_tracker.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/rtc/latency_tracker.py" \
    "lerobot_modeling_pi05.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/pi05/modeling_pi05.py" \
    "lerobot_modeling_pi0.py|https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/policies/pi0/modeling_pi0.py" \
    "rtc.mdx|https://raw.githubusercontent.com/huggingface/lerobot/main/docs/source/rtc.mdx" \
    "kinetix_model.py|https://raw.githubusercontent.com/Physical-Intelligence/real-time-chunking-kinetix/main/src/model.py" \
    "kinetix_eval_flow.py|https://raw.githubusercontent.com/Physical-Intelligence/real-time-chunking-kinetix/main/src/eval_flow.py" \
    ; do
  dest="${entry%%|*}"; url="${entry##*|}"
  curl -sL -o "$dest" "$url"
done
```

`eval_with_real_robot.py` is on the non-default `0db5f66d` branch and is
copied from a manual web fetch — adjust the curl URL if you want the
main-branch version.

### Why not just `git submodule` lerobot?

`lerobot` has 200+ MB of dependencies (PyTorch+JAX+gym+...) and pulls in
`openpi`, `kinetix`, and a half dozen other transitive dependencies that
would have to be installed even to import the RTC files. Vendoring 4 small
files keeps the reference inline-readable without polluting FlashRT's
dependency surface.

### License

Both lerobot and Physical-Intelligence's real-time-chunking-kinetix are
Apache-2.0; the file headers in each vendored file preserve the upstream
copyright notice.
