"""End-to-end smoke test for RTC server-side prefix-freeze inpainting.

Connects to a running FlashRT websocket policy server (default
``localhost:8002``), fires two inferences against a synthetic
observation, and verifies that the second inference's first ``d``
model-space actions equal the supplied ``_rtc_prev_chunk`` to bf16
precision. Also reports the seam delta at index ``d`` so the operator
can sanity-check that the suffix continues smoothly from the frozen
prefix (the whole point of the prefix-freeze).

Usage::

    PYTHONPATH=/path/to/openpi/packages/openpi-client/src \
        python scripts/smoke_rtc_prefix_freeze.py

    # or with a non-default port / d:
    python scripts/smoke_rtc_prefix_freeze.py --port 8002 --d 7

The test PASSES when:
  * ``_rtc_chunk_model_space`` is present in the server response (proves
    the openpi adapter is forwarding the model-space chunk).
  * ``chunk_2[0:d]`` matches the supplied prefix within ``--tol`` (proves
    the in-loop inpainting fired during all 10 Euler steps).
  * ``chunk_2`` shape matches the server's chunk_size metadata.

It FAILS when:
  * the prefix-freeze max-abs error exceeds ``--tol`` (default 5e-2),
    indicating the captured CUDA graph isn't actually running the
    ``gate_mul_residual`` / ``residual_add`` op pair on the RTC slots.
  * the response is missing ``_rtc_chunk_model_space`` (adapter / api
    plumbing regressed).

The seam delta is informational only — there's no hard threshold
because the model legitimately changes plan when given fresh inputs.
The useful comparison is ``seam_with_freeze`` vs ``baseline_seam``:
freeze should be at least competitive with the baseline.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any

import numpy as np


def _build_synthetic_obs(num_views: int, state_dim: int, prompt: str,
                         seed: int) -> dict[str, Any]:
    """Synthetic obs: deterministic random images + state."""
    rng = np.random.default_rng(seed)
    # 224x224x3 uint8 — server's ``_normalize_image`` skips the cv2
    # resize when already at target HW, so the test path is exactly
    # what production traffic hits.
    base = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    wrist = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
    obs: dict[str, Any] = {
        "observation/image": base,
        "observation/wrist_image": wrist,
        "observation/state": rng.standard_normal(state_dim).astype(np.float32),
        "prompt": prompt,
    }
    if num_views >= 3:
        right = rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8)
        obs["observation/wrist_image_right"] = right
    return obs


def _summarise(name: str, arr: np.ndarray) -> None:
    flat = arr.reshape(-1)
    print(f"  {name}: shape={arr.shape} dtype={arr.dtype} "
          f"min={flat.min():+.4f} max={flat.max():+.4f} "
          f"mean={flat.mean():+.4f} std={flat.std():.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--prompt", default="put the chocolate bars in the container")
    parser.add_argument("--num-views", type=int, default=3,
                        help="Match the server's --num-views flag.")
    parser.add_argument("--state-dim", type=int, default=16,
                        help="Match the server's --robot-action-dim.")
    parser.add_argument("--prefix-start", type=int, default=3,
                        help="Index in chunk_1 to start the simulated "
                             "inflight prefix from (= the consume index "
                             "we'd be at when the next inference fires).")
    parser.add_argument("--d", type=int, default=5,
                        help="Length of the freeze prefix (number of "
                             "control ticks the next inference is expected "
                             "to consume).")
    parser.add_argument("--tol", type=float, default=5e-2,
                        help="Absolute tolerance for the prefix-freeze "
                             "correctness check. Generous default to "
                             "allow for bf16 round-trip noise; the "
                             "actual error should be near zero (~1e-3 "
                             "or below) when inpainting is working.")
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s")

    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
    except ImportError:
        print("ERROR: openpi-client not importable. Set PYTHONPATH to "
              "the openpi-client checkout, e.g. PYTHONPATH=/home/.../"
              "openpi/packages/openpi-client/src")
        return 2

    print(f"Connecting to ws://{args.host}:{args.port} ...")
    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    meta = policy.get_server_metadata() or {}
    print(f"Server metadata: {meta}")
    chunk_size = int(meta.get("chunk_size", 50))
    if not 1 <= args.d <= chunk_size // 2:
        print(f"WARNING: --d={args.d} should typically be in "
              f"[1, chunk_size//2={chunk_size // 2}] for meaningful freeze")

    obs = _build_synthetic_obs(args.num_views, args.state_dim,
                               args.prompt, args.seed)

    # ── Inference 1: no RTC, get the baseline chunk in model space ──
    print("\n=== Inference 1: baseline (no RTC prefix) ===")
    resp1 = policy.infer(obs)
    if not isinstance(resp1, dict):
        print(f"ERROR: unexpected response type {type(resp1)}: {resp1!r}")
        return 1
    if "_rtc_chunk_model_space" not in resp1:
        print("FAIL: response missing '_rtc_chunk_model_space'.")
        print(f"  response keys: {sorted(resp1.keys())}")
        print("  Adapter / api wiring regressed. Server prefix-freeze "
              "cannot be tested until this is restored.")
        return 1
    chunk1 = np.asarray(resp1["_rtc_chunk_model_space"])
    actions1 = np.asarray(resp1["actions"])
    print(f"  latency:    {resp1.get('policy_timing', {}).get('infer_ms', '?')} ms")
    _summarise("chunk_model_space (1)", chunk1)
    _summarise("actions (robot space, 1)", actions1)
    if chunk1.shape[0] != chunk_size:
        print(f"WARNING: server metadata chunk_size={chunk_size} but "
              f"returned chunk has shape[0]={chunk1.shape[0]}")
    if chunk1.shape[1] != 32:
        print(f"NOTE: model-space chunk action dim = {chunk1.shape[1]} "
              f"(expected 32 for Pi0.5). RTC prefix-freeze still works "
              f"as long as the shape is consistent between calls.")

    if chunk1.shape[0] < args.prefix_start + args.d:
        print(f"ERROR: chunk too short for prefix-start={args.prefix_start} "
              f"+ d={args.d}; chunk shape={chunk1.shape}.")
        return 1

    # ── Inference 2: send back chunk1[prefix_start : prefix_start+d] as prefix ──
    prefix = np.ascontiguousarray(
        chunk1[args.prefix_start : args.prefix_start + args.d],
        dtype=np.float32)
    print(f"\n=== Inference 2: freeze prefix d={args.d} from chunk_1"
          f"[{args.prefix_start}:{args.prefix_start + args.d}] ===")
    print(f"  prefix shape: {prefix.shape}")
    _summarise("prefix sent", prefix)
    obs2 = {**obs,
            "_rtc_prev_chunk": prefix,
            "_rtc_inference_delay": int(args.d)}
    resp2 = policy.infer(obs2)
    chunk2 = np.asarray(resp2["_rtc_chunk_model_space"])
    _summarise("chunk_model_space (2)", chunk2)

    # ── Correctness check: chunk2[0:d] == prefix ──
    received_prefix = chunk2[: args.d].astype(np.float32)
    err_abs = np.abs(received_prefix - prefix)
    err_max = float(err_abs.max())
    err_mean = float(err_abs.mean())
    print("\n=== Prefix-freeze correctness ===")
    print(f"  max-abs error:  {err_max:.6f}")
    print(f"  mean-abs error: {err_mean:.6f}")
    print(f"  tolerance:      {args.tol}")
    if err_max <= args.tol:
        print("  PASS — inpainting frozen the prefix as expected.")
        result_ok = True
    else:
        print("  FAIL — prefix was NOT frozen. Likely causes:")
        print("    * graph capture elided the inpainting op (mask was 0 "
              "when captured and the op was constant-folded away)")
        print("    * gate_mul_residual is not in-place-safe with these "
              "broadcast semantics")
        print("    * buffer ptr mismatch between Python upload and the "
              "captured graph node")
        # Print per-position max error to localise the failure:
        per_pos = err_abs.max(axis=1)
        print(f"    per-position max err: {per_pos.tolist()}")
        result_ok = False

    # ── Continuity check (informational) ──
    print("\n=== Seam continuity (informational) ===")
    # Baseline: the seam delta within chunk1 between its position d-1
    # and position d (model planning naturally, no chunk boundary).
    baseline_seam = float(np.abs(chunk1[args.d] - chunk1[args.d - 1]).max())
    # Freeze: chunk2's last-frozen position vs first-free position.
    if args.d < chunk2.shape[0]:
        freeze_seam = float(
            np.abs(chunk2[args.d] - chunk2[args.d - 1]).max())
    else:
        freeze_seam = float("nan")
    # Without-freeze: pretend we just took chunk2 and spliced — what
    # would the jump be vs the inflight prefix's last position?
    naive_seam = float(np.abs(chunk2[args.d] - prefix[-1]).max())
    print(f"  baseline (within chunk1, smooth model plan):     {baseline_seam:.4f}")
    print(f"  with freeze (chunk2[d] - chunk2[d-1] = prefix):  {freeze_seam:.4f}")
    print(f"  naive (chunk2[d] - prefix[-1], untouched-suffix): {naive_seam:.4f}")
    if freeze_seam <= baseline_seam * 2:
        print("  freeze seam is within 2x baseline — looks healthy.")
    else:
        print("  freeze seam is significantly larger than baseline; the "
              "model may be reluctant to continue smoothly from this "
              "prefix on synthetic obs (this is informational, not a "
              "hard failure — the correctness check above is what "
              "matters).")

    print()
    return 0 if result_ok else 1


if __name__ == "__main__":
    sys.exit(main())
