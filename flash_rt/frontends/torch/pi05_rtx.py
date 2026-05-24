"""FlashRT -- RTX Pi0.5 torch frontend.

Loads HuggingFace PyTorch safetensors checkpoints + drives the
framework-agnostic :class:`~flash_rt.models.pi05.pipeline_rtx.Pi05Pipeline`.

This is the "reference" RTX frontend. The RTX JAX frontend
(:mod:`flash_rt.frontends.jax.pi05_rtx`) mirrors this API but loads
from Orbax and uses JAX for weight quantization.

Usage::

    from flash_rt.frontends.torch.pi05_rtx import Pi05TorchFrontendRtxRtx
    pipe = Pi05TorchFrontendRtxRtx("/path/to/pi05_libero_pytorch", num_views=2)
    pipe.set_prompt("pick up the red block")
    pipe.calibrate_with_real_data([obs_dict])   # once, ~1 s
    out = pipe.infer({"image": img, "wrist_image": wrist})
    actions = out["actions"]     # (chunk_size, 7) numpy
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.utils.actions import unnormalize_actions, LIBERO_ACTION_DIM
from flash_rt.hardware.rtx.attn_backend import RtxFlashAttnBackend
from flash_rt.models.pi05.pipeline_rtx import (
    Pi05Pipeline,
    VIS_L, VIS_D, VIS_H, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H,
    DEC_L, DEC_D, DEC_H, DEC_HD,
    ACTION_DIM, NUM_STEPS_DEFAULT,
)
from flash_rt.models.pi05.pipeline_rtx_cfg import Pi05CFGPipeline
from flash_rt.models.pi05.pipeline_rtx_batched import Pi05BatchedPipeline
from flash_rt.models.pi05.pipeline_rtx_cfg_batched import Pi05CFGBatchedPipeline
from flash_rt.hardware.rtx.attn_backend_batched_pi05 import (
    PI05_BATCH_SIZE,
    RtxFlashAttnBatchedBackendPi05,
)
from flash_rt.core.utils.hardware import supports_fp8

logger = logging.getLogger(__name__)

bf16 = torch.bfloat16
fp8_e4m3 = torch.float8_e4m3fn

CHUNK_SIZE = 10
IMG_HW = 224
MAX_PROMPT_LEN_DEFAULT = 48


# ════════════════════════════════════════════════════════════════════
#   RTC soft-guidance: per-position prefix weights schedule
# ════════════════════════════════════════════════════════════════════
#
# Byte-for-byte port of
# ``third_party/lerobot_rtc_reference/modeling_rtc.py:251-298``
# (``RTCProcessor.get_prefix_weights`` + helpers). Returned as numpy
# float32 because the staging buffer is bf16 torch and the cast happens
# in the staging path. Kept here (not in ``rtc.py``) so the function
# lives next to the only frontend that uses it; if other frontends
# need it later we can promote.
_RTC_SCHEDULES = ("linear", "exp", "ones", "zeros")
_RTC_DEFAULT_SCHEDULE = "linear"      # lerobot default (configuration_rtc.py:42)
_RTC_DEFAULT_EXECUTION_HORIZON = 10   # lerobot default (configuration_rtc.py:44)


def _rtc_linweights(start: int, end: int, total: int) -> np.ndarray:
    """Lerobot ``RTCProcessor._linweights`` parity.

    Returns linspace(1, 0, linspace_steps + 2)[1:-1] over the merge
    window [start, end). Excludes the endpoints so the boundary
    constraint joins smoothly to the leading ones / trailing zeros
    (lerobot does the exact same dropping of endpoints).
    """
    skip_steps_at_end = max(total - end, 0)
    linspace_steps = total - skip_steps_at_end - start
    if end <= start or linspace_steps <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.linspace(1.0, 0.0, linspace_steps + 2, dtype=np.float32)[1:-1]


def _get_prefix_weights(start: int, end: int, total: int,
                        schedule: str) -> np.ndarray:
    """Per-position soft-guidance weights, shape ``(total,)``.

    Mirrors ``RTCProcessor.get_prefix_weights``. Always returns a fresh
    float32 numpy array.

    Args:
        start: Inference delay in chunk positions (``d``). Positions
            ``[0, start)`` are weighted 1.0 (full guidance toward the
            inflight prefix that we have already committed to playing).
        end: Execution horizon. Positions ``[start, end)`` are the
            merge window with the schedule-specific ramp; positions
            ``[end, total)`` are weighted 0.0 (free continuation).
        total: Chunk size.
        schedule: One of ``"linear"``, ``"exp"``, ``"ones"``, ``"zeros"``
            (case-insensitive). Unknown schedules fall back to
            ``"linear"`` with a warning. See
            ``RTCAttentionSchedule`` for the upstream semantics.
    """
    sched = (schedule or _RTC_DEFAULT_SCHEDULE).lower()
    if sched not in _RTC_SCHEDULES:
        logger.warning("Unknown RTC schedule %r; falling back to %r",
                       schedule, _RTC_DEFAULT_SCHEDULE)
        sched = _RTC_DEFAULT_SCHEDULE
    start = min(start, end)
    if sched == "zeros":
        w = np.zeros(total, dtype=np.float32)
        w[:start] = 1.0
        return w
    if sched == "ones":
        w = np.ones(total, dtype=np.float32)
        w[end:] = 0.0
        return w
    lin = _rtc_linweights(start, end, total)
    if sched == "exp":
        # lerobot: lin * expm1(lin) / (e - 1)
        lin = lin * np.expm1(lin) / (math.e - 1.0)
    out = np.zeros(total, dtype=np.float32)
    if start > 0:
        out[:min(start, total)] = 1.0
    if lin.size > 0:
        out[start:start + lin.size] = lin
    return out


# ════════════════════════════════════════════════════════════════════
#   HF safetensors → pipeline weight dict (BF16 torch tensors)
# ════════════════════════════════════════════════════════════════════


def _interleave_qk(w: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Interleave Q/K output dim from HF contiguous to JAX RoPE format."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return (
        w.reshape(num_heads, head_dim, in_dim)
         .reshape(num_heads, 2, head_dim // 2, in_dim)
         .permute(0, 2, 1, 3)
         .reshape(out_dim, in_dim)
    )


def convert_pi05_safetensors(safetensors_path: Union[str, pathlib.Path]) -> dict:
    """Convert a HuggingFace Pi0.5 safetensors file to BF16 torch tensor dict.

    Key transformations (verified bit-exact against the openpi PyTorch
    reference forward on LIBERO data):

      - Vision attention: separate Q/K/V → merged, transposed (in, 3*out).
      - Vision patch embedding: ``(C_out, C_in, H, W)`` → ``(H, W, C_in, C_out)``.
      - Encoder RMSNorm fold: multiply Q/K/V/gate/up weights by ``(1 + norm_w)``
        in FP32 to avoid bf16 rounding near -1.0.
      - Encoder Q/K heads: interleave for fused RoPE kernel.
      - Decoder Q/K heads: interleave (no RMS fold — AdaRMSNorm is runtime).
      - Decoder AdaRMSNorm modulation: ``input_layernorm.dense`` →
        ``pre_attn_norm_mod`` (kept separate, BF16).
      - Output projection: frontend pre-scales ``decoder_action_out_proj_w/b``
        by ``-1.0 / num_steps`` (matching the flow-matching residual accumulation).
      - 10-step sinusoidal time embeddings.
    """
    from safetensors import safe_open
    from flash_rt.executors.torch_weights import _autodetect_strip_prefix

    logger.info("Loading Pi0.5 safetensors: %s", safetensors_path)
    f = safe_open(str(safetensors_path), framework="pt")
    # Auto-strip the lerobot HF policy ``model.`` wrap so the openpi
    # bare-key lookups below resolve transparently on either layout.
    _strip = _autodetect_strip_prefix(set(f.keys()))

    def g(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key).to(bf16)

    def g_raw(key: str) -> torch.Tensor:
        return f.get_tensor((_strip + key) if _strip else key)

    ckpt: dict = {}

    # ── Vision encoder (27 SigLIP layers) ──
    vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
    pe_w = g(f"{vp}.embeddings.patch_embedding.weight")   # (1152, 3, 14, 14)
    # Target layout (14, 14, 3, 1152) flattens contiguously to (588, 1152)
    # row-major as (h, w, c, o) — matches the patch_im2col output order.
    ckpt["vision_patch_embedding_w"] = pe_w.permute(2, 3, 1, 0).contiguous()
    ckpt["vision_patch_embedding_b"] = g(f"{vp}.embeddings.patch_embedding.bias")
    ckpt["vision_position_embedding"] = g(f"{vp}.embeddings.position_embedding.weight")

    qkv_w_list, qkv_b_list = [], []
    o_w_list, o_b_list = [], []
    up_w_list, up_b_list = [], []
    down_w_list, down_b_list = [], []
    ln1_w_list, ln1_b_list = [], []
    ln2_w_list, ln2_b_list = [], []

    for i in range(VIS_L):
        lp = f"{vp}.encoder.layers.{i}"
        q_w = g(f"{lp}.self_attn.q_proj.weight")
        k_w = g(f"{lp}.self_attn.k_proj.weight")
        v_w = g(f"{lp}.self_attn.v_proj.weight")
        qkv_w_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        q_b = g(f"{lp}.self_attn.q_proj.bias")
        k_b = g(f"{lp}.self_attn.k_proj.bias")
        v_b = g(f"{lp}.self_attn.v_proj.bias")
        qkv_b_list.append(torch.cat([q_b, k_b, v_b]))

        o_w_list.append(g(f"{lp}.self_attn.out_proj.weight").t())
        o_b_list.append(g(f"{lp}.self_attn.out_proj.bias"))

        up_w_list.append(g(f"{lp}.mlp.fc1.weight").t())
        up_b_list.append(g(f"{lp}.mlp.fc1.bias"))

        down_w_list.append(g(f"{lp}.mlp.fc2.weight").t())
        down_b_list.append(g(f"{lp}.mlp.fc2.bias"))

        ln1_w_list.append(g(f"{lp}.layer_norm1.weight"))
        ln1_b_list.append(g(f"{lp}.layer_norm1.bias"))
        ln2_w_list.append(g(f"{lp}.layer_norm2.weight"))
        ln2_b_list.append(g(f"{lp}.layer_norm2.bias"))

    ckpt["vision_attn_qkv_w"] = torch.stack(qkv_w_list)
    ckpt["vision_attn_qkv_b"] = torch.stack(qkv_b_list)
    ckpt["vision_attn_o_w"] = torch.stack(o_w_list)
    ckpt["vision_attn_o_b"] = torch.stack(o_b_list)
    ckpt["vision_ffn_up_w"] = torch.stack(up_w_list)
    ckpt["vision_ffn_up_b"] = torch.stack(up_b_list)
    ckpt["vision_ffn_down_w"] = torch.stack(down_w_list)
    ckpt["vision_ffn_down_b"] = torch.stack(down_b_list)
    ckpt["vision_pre_attn_norm_w"] = torch.stack(ln1_w_list)
    ckpt["vision_pre_attn_norm_b"] = torch.stack(ln1_b_list)
    ckpt["vision_pre_ffn_norm_w"] = torch.stack(ln2_w_list)
    ckpt["vision_pre_ffn_norm_b"] = torch.stack(ln2_b_list)
    ckpt["vision_final_norm_w"] = g(f"{vp}.post_layernorm.weight")
    ckpt["vision_final_norm_b"] = g(f"{vp}.post_layernorm.bias")

    # ── Multi-modal projector ──
    mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
    ckpt["encoder_multi_modal_projector_w"] = g(f"{mp}.weight").t()
    ckpt["encoder_multi_modal_projector_b"] = g(f"{mp}.bias")

    # ── Encoder (18 Gemma-2B layers with RMSNorm fold) ──
    ep = "paligemma_with_expert.paligemma.model.language_model.layers"
    enc_qkv_list, enc_o_list = [], []
    enc_gate_list, enc_up_list, enc_down_list = [], [], []

    for i in range(ENC_L):
        # CRITICAL: fuse in FP32 — bf16 rounds values near -1.0 to exactly
        # -1.0, collapsing (1 + scale) to 0 and zeroing entire channels.
        attn_scale = g_raw(f"{ep}.{i}.input_layernorm.weight").float()
        fuse_attn = 1.0 + attn_scale  # (2048,)

        q_w = g_raw(f"{ep}.{i}.self_attn.q_proj.weight").float()
        k_w = g_raw(f"{ep}.{i}.self_attn.k_proj.weight").float()
        v_w = g_raw(f"{ep}.{i}.self_attn.v_proj.weight").float()
        q_w = _interleave_qk(q_w, 8)
        k_w = _interleave_qk(k_w, 1)
        q_w = q_w * fuse_attn.unsqueeze(0)
        k_w = k_w * fuse_attn.unsqueeze(0)
        v_w = v_w * fuse_attn.unsqueeze(0)
        qkv = torch.cat([q_w, k_w, v_w], dim=0).t().to(bf16)
        enc_qkv_list.append(qkv)

        enc_o_list.append(g(f"{ep}.{i}.self_attn.o_proj.weight").t())

        ffn_scale = g_raw(f"{ep}.{i}.post_attention_layernorm.weight").float()
        fuse_ffn = 1.0 + ffn_scale

        gate_w = g_raw(f"{ep}.{i}.mlp.gate_proj.weight").float() * fuse_ffn.unsqueeze(0)
        up_w = g_raw(f"{ep}.{i}.mlp.up_proj.weight").float() * fuse_ffn.unsqueeze(0)
        enc_gate_list.append(gate_w.t().to(bf16))
        enc_up_list.append(up_w.t().to(bf16))

        enc_down_list.append(g(f"{ep}.{i}.mlp.down_proj.weight").t())

    ckpt["encoder_attn_qkv_w"] = torch.stack(enc_qkv_list)
    ckpt["encoder_attn_o_w"] = torch.stack(enc_o_list)
    ckpt["encoder_ffn_gate_w"] = torch.stack(enc_gate_list)
    ckpt["encoder_ffn_up_w"] = torch.stack(enc_up_list)
    ckpt["encoder_ffn_down_w"] = torch.stack(enc_down_list)

    # ── Decoder (18 Gemma-300M layers) ──
    dp = "paligemma_with_expert.gemma_expert.model.layers"
    dec_qkv_list, dec_o_list = [], []
    dec_gate_list, dec_up_list, dec_down_list = [], [], []
    dec_attn_mod_w_list, dec_attn_mod_b_list = [], []
    dec_ffn_mod_w_list, dec_ffn_mod_b_list = [], []

    for i in range(DEC_L):
        dec_attn_mod_w_list.append(g(f"{dp}.{i}.input_layernorm.dense.weight").t())
        dec_attn_mod_b_list.append(g(f"{dp}.{i}.input_layernorm.dense.bias"))

        q_w = g(f"{dp}.{i}.self_attn.q_proj.weight")
        k_w = g(f"{dp}.{i}.self_attn.k_proj.weight")
        v_w = g(f"{dp}.{i}.self_attn.v_proj.weight")
        q_w = _interleave_qk(q_w.float(), 8).to(q_w.dtype)
        k_w = _interleave_qk(k_w.float(), 1).to(k_w.dtype)
        dec_qkv_list.append(torch.cat([q_w, k_w, v_w], dim=0).t())

        dec_o_list.append(g(f"{dp}.{i}.self_attn.o_proj.weight").t())

        dec_ffn_mod_w_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.weight").t())
        dec_ffn_mod_b_list.append(
            g(f"{dp}.{i}.post_attention_layernorm.dense.bias"))

        dec_gate_list.append(g(f"{dp}.{i}.mlp.gate_proj.weight").t())
        dec_up_list.append(g(f"{dp}.{i}.mlp.up_proj.weight").t())
        dec_down_list.append(g(f"{dp}.{i}.mlp.down_proj.weight").t())

    ckpt["decoder_attn_qkv_w"] = torch.stack(dec_qkv_list)
    ckpt["decoder_attn_o_w"] = torch.stack(dec_o_list)
    ckpt["decoder_ffn_gate_w"] = torch.stack(dec_gate_list)
    ckpt["decoder_ffn_up_w"] = torch.stack(dec_up_list)
    ckpt["decoder_ffn_down_w"] = torch.stack(dec_down_list)
    ckpt["decoder_pre_attn_norm_mod_w"] = torch.stack(dec_attn_mod_w_list)
    ckpt["decoder_pre_attn_norm_mod_b"] = torch.stack(dec_attn_mod_b_list)
    ckpt["decoder_pre_ffn_norm_mod_w"] = torch.stack(dec_ffn_mod_w_list)
    ckpt["decoder_pre_ffn_norm_mod_b"] = torch.stack(dec_ffn_mod_b_list)

    ckpt["decoder_final_norm_mod_w"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.weight").t()
    ckpt["decoder_final_norm_mod_b"] = g(
        "paligemma_with_expert.gemma_expert.model.norm.dense.bias")

    # ── Time MLP + sinusoidal embeddings ──
    ckpt["decoder_time_mlp_in_w"] = g("time_mlp_in.weight").t()
    ckpt["decoder_time_mlp_in_b"] = g("time_mlp_in.bias")
    ckpt["decoder_time_mlp_out_w"] = g("time_mlp_out.weight").t()
    ckpt["decoder_time_mlp_out_b"] = g("time_mlp_out.bias")

    num_steps = NUM_STEPS_DEFAULT
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    embedding_dim = DEC_D
    fraction = torch.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    time_emb_list = []
    for _ in range(num_steps):
        sinusoid_input = t.unsqueeze(-1) * (1.0 / period).unsqueeze(0) * 2 * math.pi
        time_emb_list.append(
            torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(bf16)
        )
        t = t + dt
    ckpt["decoder_time_embeds"] = torch.cat(time_emb_list, dim=0)  # (10, 1024)

    # ── Action projections (pre-scaled by frontend before pipeline build) ──
    ckpt["decoder_action_in_proj_w"] = g("action_in_proj.weight").t()
    ckpt["decoder_action_in_proj_b"] = g("action_in_proj.bias")
    ckpt["decoder_action_out_proj_w"] = g("action_out_proj.weight").t()
    ckpt["decoder_action_out_proj_b"] = g("action_out_proj.bias")

    # ── Embedding matrix (for prompt tokenisation) ──
    ckpt["embedding_weight"] = g("paligemma_with_expert.paligemma.lm_head.weight")

    logger.info("Converted %d weight groups", len(ckpt))
    return ckpt


def _embed_prompt(prompt_text: str, embedding_weight: torch.Tensor,
                  max_len: int = 48,
                  state: Optional[np.ndarray] = None,
                  pad_to_max: bool = False) -> tuple[torch.Tensor, int]:
    """Tokenise + embed via PaliGemma embedding table (CUDA, bf16).

    Args:
        prompt_text: task language string.
        embedding_weight: PaliGemma embedding matrix on CUDA, bf16.
        max_len: tokenizer max length (also pad target if ``pad_to_max``).
        state: optional normalised-to-[-1, 1] proprioceptive state vector.
            When provided, the tokenizer uses the Pi0.5 discrete-state
            format: ``f"Task: {prompt}, State: {state_str};\\nAction: "``
            with each state dim discretised into 256 bins. Equivalent to
            openpi's ``PaligemmaTokenizer.tokenize(prompt, state=state)``.
            Caller must pre-normalise state (FlashRT's frontend does this
            using ``self.norm_stats``).
        pad_to_max: if True, return embeds of shape ``(max_len, emb_dim)``
            with PaliGemma's PAD token id (0) filling unused positions
            and the returned ``prompt_len`` equal to ``max_len``. This is
            required when the caller wants to share a single
            captured-graph pipeline across prompts whose unpadded token
            counts vary (e.g. per-frame state changes in Pi0.5 — the
            unpadded count drifts by ±2 tokens as 8-bit-quantised joint
            values cross digit boundaries; without padding, each drift
            would trigger a 3 s pipeline rebuild).
    """
    # PaliGemma tokenizer resolution — see
    # `flash_rt.utils.paligemma_tokenizer` for the search order and
    # the download instructions emitted on failure.
    try:
        # Preferred: openpi's PaligemmaTokenizer (exact same vocab,
        # same prompt prefix logic FlashRT was built against).
        from openpi.models.tokenizer import PaligemmaTokenizer
        tokenizer = PaligemmaTokenizer(max_len=max_len)
        tokens_np, mask_np = tokenizer.tokenize(prompt_text, state=state)
        unpadded_len = int(mask_np.sum())
        if pad_to_max:
            # PaligemmaTokenizer already pads with PAD-id 0 to max_len;
            # take the full sequence (real + PAD) and embed it. Captured
            # graph stays valid because the buffer shape is constant.
            token_ids = torch.tensor(
                tokens_np, dtype=torch.long, device="cuda")
            prompt_len = max_len
        else:
            token_ids = torch.tensor(
                tokens_np[:unpadded_len], dtype=torch.long, device="cuda")
            prompt_len = unpadded_len
    except (ImportError, FileNotFoundError, OSError, RuntimeError):
        # Fallback: locate the SentencePiece model directly via the
        # FlashRT helper (clear error if not found — never silent
        # segfault).
        from flash_rt.utils.paligemma_tokenizer import (
            load_paligemma_sentencepiece,
        )
        sp = load_paligemma_sentencepiece()
        if state is not None:
            # Replicate openpi's Pi0.5 discrete-state-input format
            # (openpi/src/openpi/models/tokenizer.py:22).
            discretized = (np.digitize(
                np.asarray(state, dtype=np.float32),
                bins=np.linspace(-1.0, 1.0, 257)[:-1]) - 1).astype(int)
            state_str = " ".join(str(int(d)) for d in discretized)
            cleaned = prompt_text.strip().replace("_", " ").replace("\n", " ")
            full_prompt = (
                f"Task: {cleaned}, State: {state_str};\nAction: ")
            tokens = [sp.bos_id()] + sp.Encode(full_prompt)
        else:
            # 108 is PaliGemma's `\n` token, used by openpi as the
            # prompt-end separator before the action prefix.
            tokens = [sp.bos_id()] + sp.Encode(prompt_text) + [108]
        if pad_to_max and len(tokens) < max_len:
            # PaliGemma PAD token id = 0.
            tokens = tokens + [0] * (max_len - len(tokens))
        elif len(tokens) > max_len:
            # openpi truncates rather than raising; match that.
            tokens = tokens[:max_len]
        token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
        prompt_len = len(token_ids)

    if embedding_weight.device.type != "cuda":
        embedding_weight = embedding_weight.to(device="cuda")

    embeds = F.embedding(token_ids, embedding_weight)
    embeds = embeds * float(embeds.shape[-1] ** 0.5)
    return embeds, prompt_len


# ════════════════════════════════════════════════════════════════════
#   Weight FP8 quantization + precomputed decoder styles
# ════════════════════════════════════════════════════════════════════


def _quantize_fp8_e4m3(w_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor symmetric FP8 E4M3 quantization."""
    amax = w_bf16.float().abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w_bf16.float() / scale).clamp(-448.0, 448.0).to(fp8_e4m3)
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device="cuda")
    return w_fp8, scale_tensor


def _select_fp8_layout(hardware: Optional[str], fp8_layout: Optional[str]) -> str:
    """Choose the Pi0.5 FP8 weight layout.

    ``kn`` is the existing SM120 path: weights are stored as [K,N] and use
    ``fp8_nn_dev``. ``nk`` is the SM89-compatible path: weights are stored
    as [N,K] and use ``fp8_nt_dev``.
    """
    if fp8_layout is not None:
        if fp8_layout not in ("kn", "nk"):
            raise ValueError(f"fp8_layout must be 'kn' or 'nk', got {fp8_layout!r}")
        return fp8_layout
    if hardware == "rtx_sm89":
        return "nk"
    if hardware == "rtx_sm120":
        return "kn"
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major == 8 and minor == 9:
                return "nk"
    except Exception:
        pass
    return "kn"


def _precompute_decoder_styles(ckpt: dict, chunk_size: int,
                               num_steps: int = NUM_STEPS_DEFAULT) -> dict:
    """Pre-compute the time-MLP + per-layer style modulations in torch.

    Output dict has numpy arrays (dtype bf16 via torch→numpy view):
        time_emb:    (num_steps, chunk_size, DEC_D)
        style_attn:  (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_ffn:   (num_steps, DEC_L, chunk_size, 3 * DEC_D)
        style_final: (num_steps, chunk_size, 3 * DEC_D)

    All computation runs on CUDA in bf16, then is moved to CPU and viewed
    as uint16 so it can be uploaded verbatim to CudaBuffer (bf16 = 2 bytes,
    numpy doesn't natively support bf16 but the bytes round-trip).

    Time embeddings are regenerated from scratch for the given num_steps so
    that any step count works correctly (e.g. num_steps=5 gives
    t=1.0, 0.8, 0.6, 0.4, 0.2 with dt=-0.2, not a truncation of the
    10-step table stored in the checkpoint).
    """
    W = {k: v.to("cuda", bf16) if isinstance(v, torch.Tensor) else v
         for k, v in ckpt.items()}

    # Regenerate sinusoidal time embeddings for the given num_steps / dt.
    # The checkpoint stores a 10-step table; generate fresh ones so any
    # step count gets the correct (t=1, t=1-dt, …) time schedule.
    dt = -1.0 / num_steps
    t = torch.tensor(1.0, dtype=torch.float32)
    min_period, max_period = 4e-3, 4.0
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32)
    period = min_period * (max_period / min_period) ** fraction
    _time_emb_rows = []
    for _ in range(num_steps):
        # period has shape (DEC_D//2,); t is a scalar tensor → sinusoid: (DEC_D//2,)
        sinusoid = t * (1.0 / period) * 2 * math.pi
        _time_emb_rows.append(
            torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1).to(bf16))
        t = t + dt
    time_emb_schedule = torch.stack(_time_emb_rows, dim=0).to("cuda")  # (steps, DEC_D)
    t_in_w = W["decoder_time_mlp_in_w"]                       # (1024, 1024)
    t_in_b = W["decoder_time_mlp_in_b"]                       # (1024,)
    t_out_w = W["decoder_time_mlp_out_w"]
    t_out_b = W["decoder_time_mlp_out_b"]

    attn_mod_w = W["decoder_pre_attn_norm_mod_w"]             # (L, 1024, 3072)
    attn_mod_b = W["decoder_pre_attn_norm_mod_b"]             # (L, 3072)
    ffn_mod_w = W["decoder_pre_ffn_norm_mod_w"]
    ffn_mod_b = W["decoder_pre_ffn_norm_mod_b"]
    final_mod_w = W["decoder_final_norm_mod_w"]               # (1024, 3072)
    final_mod_b = W["decoder_final_norm_mod_b"]               # (3072,)

    time_emb_out = torch.empty(num_steps, chunk_size, DEC_D, dtype=bf16, device="cuda")
    style_attn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")
    style_ffn = torch.empty(num_steps, DEC_L, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")
    style_final = torch.empty(num_steps, chunk_size, 3 * DEC_D, dtype=bf16, device="cuda")

    for step in range(num_steps):
        te = time_emb_schedule[step:step + 1]                 # (1, 1024)
        tmp = te @ t_in_w + t_in_b[None, :]                   # SiLU input
        tmp = (tmp.float() * torch.sigmoid(tmp.float())).to(bf16)
        tmp2 = tmp @ t_out_w + t_out_b[None, :]
        tmp2 = (tmp2.float() * torch.sigmoid(tmp2.float())).to(bf16)
        te_expanded = tmp2.expand(chunk_size, -1).contiguous()  # (chunk, 1024)
        time_emb_out[step] = te_expanded

        for i in range(DEC_L):
            style_attn[step, i] = te_expanded @ attn_mod_w[i] + attn_mod_b[i][None, :]
            style_ffn[step, i] = te_expanded @ ffn_mod_w[i] + ffn_mod_b[i][None, :]

        style_final[step] = te_expanded @ final_mod_w + final_mod_b[None, :]

    # View as uint16 (bf16 bit pattern) so numpy can round-trip bytes.
    def _to_np_u16(t: torch.Tensor) -> np.ndarray:
        return t.contiguous().view(torch.uint16).cpu().numpy()

    return {
        "time_emb": _to_np_u16(time_emb_out),
        "style_attn": _to_np_u16(style_attn),
        "style_ffn": _to_np_u16(style_ffn),
        "style_final": _to_np_u16(style_final),
    }


# ════════════════════════════════════════════════════════════════════
#   Pi05TorchFrontendRtx frontend
# ════════════════════════════════════════════════════════════════════


class Pi05TorchFrontendRtx:
    """RTX consumer GPU Pi0.5 Torch frontend.

    Mirrors the :class:`ThorPipelineTorch` public API (``set_prompt`` +
    ``infer`` + ``calibrate_with_real_data`` + ``get_latency_stats``) so the
    same eval scripts work on both hardware families.
    """

    def __init__(self,
                 checkpoint_dir: Union[str, pathlib.Path],
                 num_views: int = 2,
                 chunk_size: int = CHUNK_SIZE,
                 max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
                 num_steps: int = NUM_STEPS_DEFAULT,
                 vision_pool_factor: int = 1,
                 vision_num_layers: Optional[int] = None,
                 cache_frames: int = 1,
                 use_fp8: bool = True,
                 hardware: Optional[str] = None,
                 fp8_layout: Optional[str] = None,
                 robot_action_dim: Optional[int] = None):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        # The model outputs ACTION_DIM=32 raw values; the frontend slices
        # the first ``robot_action_dim`` to match the robot's DOF.
        # Default 7 = LIBERO (xyz + rot + gripper). OpenArm bimanual is 16.
        # Override via constructor (preferred) or env var FLASHRT_ROBOT_ACTION_DIM.
        if robot_action_dim is None:
            robot_action_dim = int(
                os.environ.get("FLASHRT_ROBOT_ACTION_DIM", LIBERO_ACTION_DIM))
        if not 1 <= robot_action_dim <= ACTION_DIM:
            raise ValueError(
                f"robot_action_dim must be in [1, {ACTION_DIM}], got {robot_action_dim}")
        self.robot_action_dim = int(robot_action_dim)
        self.max_prompt_len = int(max_prompt_len)
        self._num_steps = int(num_steps)
        self._vision_pool_factor = int(vision_pool_factor)
        if self._num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {self._num_steps}")
        if self._vision_pool_factor not in (1, 2, 4):
            raise ValueError(
                "vision_pool_factor must be one of {1, 2, 4}; "
                f"got {self._vision_pool_factor}")
        # Temporal K/V caching: run full pipeline every `cache_frames` frames,
        # intermediate frames reuse the cached encoder K/V (decoder-only).
        # cache_frames=1 (default) = no caching, every frame is full.
        # cache_frames=2 = full, decode, full, decode, ...
        self._cache_frames = int(cache_frames)
        if self._cache_frames < 1:
            raise ValueError(f"cache_frames must be >= 1, got {self._cache_frames}")
        self._frame_count = 0
        from flash_rt.models.pi05.pipeline_rtx import VIS_L as _VIS_L
        self._vision_num_layers = _VIS_L if vision_num_layers is None else int(vision_num_layers)
        if not 1 <= self._vision_num_layers <= _VIS_L:
            raise ValueError(
                f"vision_num_layers must be in [1, {_VIS_L}], "
                f"got {self._vision_num_layers}")
        # _use_int8_vision_static is set after _force_int8_decoder below
        self.use_fp8 = bool(use_fp8)
        self.fp8_layout = _select_fp8_layout(hardware, fp8_layout)

        self.latency_records: list[float] = []
        self.calibrated = False
        self.graph_recorded = False
        # Persistent across pipeline rebuilds: cuBLASLt's per-shape tuned
        # algo is cached on the shared GemmRunner instance, not on the
        # pipeline. Re-running autotune across rebuilds (1) is wasted work
        # since the shared cache already has the best algo for each
        # (M, N, K) and (2) reliably hits a cuBLASLt illegal-memory-access
        # crash on the 3rd+ rebuild on Spark/SM121 — re-tuning the same
        # cached entry from a new pipeline's tensors triggers it (see
        # csrc/gemm/gemm_runner.cu:178 autotune warmup). The first
        # successful run sets this flag; subsequent rebuilds use the
        # cached algos via cuBLASLt's heuristic top-1 (or the
        # previously-tuned algo on shared shapes — vision and decoder).
        self._gemm_autotune_done = False
        # Snapshot of FP8 activation scales calibrated against the full
        # 80-sample dataset (with diverse states + diverse noise). Restored
        # into every rebuilt pipeline so per-frame rebuilds (state-in-prompt
        # mode) reuse the multi-frame scales instead of degenerating to
        # single-frame scales that don't cover noise variance and tank
        # cosine to ~0.6 vs the JAX reference. Keyed by FP8 GEMM name
        # ("encoder_attn_qkv_w_0", etc.), value is the float32 amax/scale.
        self._fp8_scales_snapshot: dict[str, np.ndarray] = {}
        self.current_prompt_len = 0
        self.pipeline: Optional[Pi05Pipeline] = None
        # Cache of fully-built pipelines keyed by exact prompt_len. State-in-prompt
        # mode (Pi0.5 discrete_state_input) produces a small set of prompt token
        # counts per task (OpenArm chocolate_bars: 78..82 → 5 distinct lengths;
        # LIBERO 7-DOF: ~45 → 1 length). Building one Pi05Pipeline per observed
        # length and caching them turns the second visit to a known length into a
        # pure pointer swap (no autotune, no warmup, no graph re-capture), which
        # eliminates the ~600 ms per-frame rebuild spike that otherwise breaks
        # 25 Hz robot control.
        #
        # We key on EXACT prompt_len (not on a bucket bound) so the captured
        # graph is shape-matched to the prompt — no encoder padding, no
        # attention pollution. The earlier zero-pad-to-max experiment dropped
        # cos(FlashRT, teleop) from 0.989 to 0.892; bucketing-with-padding
        # would re-introduce that. With state-in-prompt this means typically
        # 3-6 cached pipelines per task; on Spark (121 GB pool) each pipeline
        # adds ~200-300 MB of scratch + activations, which is trivial.
        #
        # FP8 scales are restored from self._fp8_scales_snapshot into every
        # newly-built bucket pipeline so they inherit the multi-frame
        # calibration instead of degenerating to single-frame scales (see
        # _restore_fp8_scales docstring).
        #
        # TODO(varlen-encoder): the bucket cache is a deliberate workaround
        # for variable prompt length, not the long-term solution. The cache
        # still pays one ~600 ms build the first time each new prompt_len is
        # seen — fine for tasks with a small, predictable token-count set
        # (OpenArm, LIBERO), but a real production deployment with
        # open-vocabulary prompts will eventually need encoder-attention
        # masking via FA2 varlen so ONE captured graph at max_prompt_len
        # covers every length with zero rebuild risk forever.
        #
        # The native FA2 varlen wrapper is already built and bit-exact
        # verified — see ``csrc/attention/fa2_wrapper.cu``
        # (``fvk_attention_fa2_fwd_bf16_varlen``) and Python binding
        # ``flash_rt.flash_rt_fa2.fwd_bf16_varlen``. The remaining work
        # to flip the pipeline to varlen is documented in
        # ``docs/spark_status.md`` § G4 (deferred work). The blocker
        # there is decoder cross-attn's K/V cache layout (chunk K/V is
        # written at offset ``enc_seq`` inside the shared encoder cache —
        # see ``pipeline_rtx.py:1753`` — which today bakes ``enc_seq``
        # into the captured graph as a Python int via pointer arithmetic;
        # the dev_offset kernel mod to fix this is sketched in G4 too).
        self._pipeline_cache: dict[int, Pi05Pipeline] = {}
        # Soft-cap warning: if the cache exceeds this many entries the
        # operator is likely seeing far more prompt-length variance than
        # expected, which would balloon memory + amortise build time
        # uncontrollably. Override with FLASHRT_PIPELINE_CACHE_WARN=N.
        self._pipeline_cache_warn_threshold = int(
            os.environ.get("FLASHRT_PIPELINE_CACHE_WARN", "8"))
        # Last prompt text passed to set_prompt() — used as a fallback by
        # _calibrate_multi_frame when calibration samples don't carry
        # their own per-sample prompt string.
        self._current_prompt: Optional[str] = None
        # RL inference configuration. ``None`` = default behaviour (single
        # forward, no advantage-conditioned prompt injection). When set
        # by :meth:`set_rl_mode`, the next :meth:`set_prompt` call builds
        # a Pi05CFGPipeline and runs classifier-free guidance.
        self._rl_config: Optional[dict] = None
        self._rl_current_prompt_text: Optional[str] = None
        self._force_int8_decoder = os.environ.get(
            "FVK_PI05_RTX_FORCE_INT8", "0") == "1"
        # FVK_PI05_RTX_INT8_ENCODER_ONLY=1: enable INT8 for encoder (large M,
        # 92% GPU utilisation) but keep decoder in BF16 (M=10 → INT8 CUTLASS
        # tile waste makes it slower than cuBLASLt BF16 for small M).
        _enc_only = os.environ.get("FVK_PI05_RTX_INT8_ENCODER_ONLY", "0") == "1"
        if _enc_only:
            self._force_int8_decoder = False   # BF16 decoder
        # On non-FP8 GPUs (e.g. Orin SM87), enable encoder INT8 alongside
        # decoder INT8 so all large GEMMs benefit from tensor-core acceleration.
        self._use_int8_encoder = self._force_int8_decoder or _enc_only
        self._int8_encoder_only = _enc_only
        # Vision GEMMs (VIS_D=1152, seq=512): static per-tensor INT8 was
        # measured to break encoder cosine (0.991 → 0.282) — disabled
        # permanently. Dynamic per-row INT8 is opt-in via
        # FVK_PI05_RTX_INT8_VISION=1 (untested at branch time; enabling
        # it requires cosine validation on the actual deployment).
        self._use_int8_vision = (
            os.environ.get("FVK_PI05_RTX_INT8_VISION", "0") == "1")
        self._use_int8_vision_static = False
        env_force_bf16 = os.environ.get("FVK_PI05_RTX_FORCE_BF16", "0") == "1"
        self._force_bf16 = (
            (env_force_bf16 or not supports_fp8()) and
            not self._force_int8_decoder
        )

        # ── Load norm_stats ──
        self._load_norm_stats(checkpoint_dir)

        # ── Load + convert safetensors ──
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path} — "
                "Pi05TorchFrontendRtx expects a HuggingFace-style PyTorch checkpoint")
        self._checkpoint_path = str(safetensors_path)
        raw_ckpt = convert_pi05_safetensors(safetensors_path)

        # Move all tensors to CUDA bf16 (retain as member attrs so their
        # memory stays alive across pipeline rebuilds).
        self._ckpt_bf16 = {}
        for k, v in raw_ckpt.items():
            if isinstance(v, torch.Tensor):
                self._ckpt_bf16[k] = v.to("cuda", bf16).contiguous()
            else:
                self._ckpt_bf16[k] = v
        self.embedding_weight = self._ckpt_bf16["embedding_weight"]

        # Pre-scale decoder action output projection by -1/num_steps.
        # Scaling is specific to the step count (ODE integration step size).
        num_steps = self._num_steps
        self._ckpt_bf16["decoder_action_out_proj_w"] = \
            self._ckpt_bf16["decoder_action_out_proj_w"] * (-1.0 / num_steps)
        self._ckpt_bf16["decoder_action_out_proj_b"] = \
            self._ckpt_bf16["decoder_action_out_proj_b"] * (-1.0 / num_steps)

        # ── Low-precision weight stores ──
        self._fp8_weights: dict = {}
        self._fp8_store: list = []  # holds tensors alive
        self._int8_weights: dict = {}
        self._int8_store: list = []
        self._int8_weight_scales: dict[str, torch.Tensor] = {}
        if self.use_fp8 and not self._force_bf16 and not self._force_int8_decoder:
            self._quantize_all_fp8()
        if self._force_int8_decoder:
            self._quantize_decoder_int8()
        if self._use_int8_encoder:
            self._quantize_encoder_int8()
        if self._use_int8_vision:
            self._quantize_vision_int8()
        if self._use_int8_vision_static:
            self._quantize_vision_int8()  # pre-quantize weights; activations use static calibrated scales

        # ── Pre-compute decoder styles (time MLP + style modulation) ──
        self._precomputed_styles = _precompute_decoder_styles(
            self._ckpt_bf16, self.chunk_size, num_steps=self._num_steps)

        # ── Attention backend (torch, owns Q/K/V/O) ──
        enc_seq_max = self.num_views * 256 + self.max_prompt_len
        self.attn_backend = RtxFlashAttnBackend(
            num_views=self.num_views,
            encoder_seq_max=enc_seq_max,
            chunk_size=self.chunk_size,
            num_encoder_layers=ENC_L)

        # ── fvk module + GemmRunner ──
        from flash_rt import flash_rt_kernels as fvk
        self.fvk = fvk
        self.gemm = fvk.GemmRunner()

        # ── Reusable pre-allocated input buffers (match Thor style) ──
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=bf16, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        # ── RTC soft-guidance staging tensors (Phase 6 / G11) ──
        # The captured pipeline always runs the per-step RTC kernel;
        # these staging tensors are uploaded into the pipeline's RTC
        # slots on every :meth:`infer` call. Default contents are zero
        # — zero ``rtc_weights`` makes the kernel a numerical no-op
        # (``v_new = v``) for non-RTC traffic. See
        # ``Pi05Pipeline._rtc_apply_guidance`` for the algorithm.
        self._rtc_prev_chunk_buf = torch.zeros(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        self._rtc_weights_buf = torch.zeros(
            self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
        # Tracks whether the last upload was non-zero, so a non-RTC
        # call right after an RTC call can re-zero in one upload
        # instead of skipping it (which would leave stale weights on
        # the GPU and cause the next chunk to apply guidance toward
        # last frame's prefix even though the client didn't ask for
        # it).
        self._rtc_last_call_active = False
        # RTC config (set by frontend caller before infer; defaults
        # match lerobot RTCConfig). These are read by ``_stage_rtc_inputs``
        # at every call so they can be updated between inferences
        # without recapturing the pipeline graph (the *values* baked
        # into the captured graph are time and guidance_weight, both
        # of which are functions of ``step`` and ``num_steps`` — not
        # of execution_horizon / schedule, which only affect the
        # uploaded weights tensor contents).
        self._rtc_execution_horizon: int = _RTC_DEFAULT_EXECUTION_HORIZON
        self._rtc_schedule: str = _RTC_DEFAULT_SCHEDULE
        from flash_rt.core.cuda_buffer import _cudart
        self._cudart = _cudart

        logger.info(
            "Pi05TorchFrontendRtx initialised (num_views=%d, chunk=%d, fp8_layout=%s)",
            self.num_views, self.chunk_size, self.fp8_layout)

    def _pipeline_precision_kwargs(self) -> dict:
        if self._force_int8_decoder or getattr(self, "_int8_encoder_only", False):
            mode = ("INT8 encoder+decoder" if self._force_int8_decoder
                    else "INT8 encoder only (decoder stays BF16 for M=10 efficiency)")
            logger.warning("FVK_PI05_RTX_FORCE_INT8/INT8_ENCODER_ONLY set: %s", mode)
            return {
                "use_fp8": False,
                "use_fp8_decoder": False,
                "use_int8_decoder": self._force_int8_decoder,
                "use_int8_encoder": self._use_int8_encoder,
                "use_int8_vision": self._use_int8_vision,
                "use_int8_vision_static": self._use_int8_vision_static,
            }
        if self._force_bf16:
            reason = (
                "FVK_PI05_RTX_FORCE_BF16=1 set"
                if os.environ.get("FVK_PI05_RTX_FORCE_BF16", "0") == "1"
                else "GPU does not advertise FP8 support"
            )
            logger.warning(
                "%s: disabling FP8 paths for the Pi0.5 RTX pipeline.",
                reason,
            )
            return {
                "use_fp8": False,
                "use_fp8_decoder": False,
                "use_int8_decoder": False,
                "use_int8_encoder": False,
                "use_int8_vision": False,
                "use_int8_vision_static": False,
            }
        return {
            "use_fp8": self.use_fp8,
            "use_fp8_decoder": self.use_fp8,
            "use_int8_decoder": False,
            "use_int8_encoder": False,
            "use_int8_vision": False,
            "use_int8_vision_static": False,
        }

    # -----------------------------------------------------------------
    # Checkpoint helpers
    # -----------------------------------------------------------------

    def _load_norm_stats(self, checkpoint_dir: pathlib.Path) -> None:
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, pi05_candidates,
        )
        try:
            self.norm_stats = load_norm_stats(
                pi05_candidates(checkpoint_dir), checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"norm_stats not found near checkpoint: {e}") from e

    def _normalize_state_for_prompt(self, state) -> np.ndarray:
        """Map raw physical state to ``[-1, 1]`` using state q01/q99.

        Mirrors openpi's ``Normalize`` transform for the ``state`` field
        (which fires upstream of ``TokenizePrompt`` in their data pipe).
        Quantile-normalised values are then clipped to ``[-1, 1]`` so the
        downstream ``np.digitize(..., linspace(-1, 1, 257)[:-1])`` lands
        inside the 256 valid bins for every dimension (out-of-range
        physical values would otherwise pile up at bin 255 and silently
        lose state resolution at the tails).
        """
        s = np.asarray(state, dtype=np.float32).reshape(-1)
        ns = self.norm_stats.get("state") if self.norm_stats else None
        if ns is None or "q01" not in ns or "q99" not in ns:
            raise RuntimeError(
                "set_prompt(state=...) requires norm_stats with a 'state' "
                "block (q01/q99). Loaded norm_stats keys: "
                f"{list(self.norm_stats.keys()) if self.norm_stats else None}")
        q01 = np.asarray(ns["q01"], dtype=np.float32).reshape(-1)
        q99 = np.asarray(ns["q99"], dtype=np.float32).reshape(-1)
        n = min(s.shape[0], q01.shape[0], q99.shape[0])
        s = s[:n]
        q01 = q01[:n]
        q99 = q99[:n]
        norm = (s - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        return np.clip(norm, -1.0, 1.0)

    def _quantize_all_fp8(self) -> None:
        """Pre-quantize all large GEMM weights to FP8 E4M3."""
        W = self._ckpt_bf16
        store = self._fp8_store
        fp8 = self._fp8_weights

        def quant(name: str, w: torch.Tensor):
            if self.fp8_layout == "nk":
                w = w.t().contiguous()
            else:
                w = w.contiguous()
            w_fp8, scale = _quantize_fp8_e4m3(w)
            store.append(w_fp8)
            store.append(scale)
            fp8[name] = (w_fp8.data_ptr(), scale.data_ptr())

        # Vision (27 layers × 4) + projector
        for i in range(VIS_L):
            quant(f"vision_attn_qkv_w_{i}", W["vision_attn_qkv_w"][i])
            quant(f"vision_attn_o_w_{i}", W["vision_attn_o_w"][i])
            quant(f"vision_ffn_up_w_{i}", W["vision_ffn_up_w"][i])
            quant(f"vision_ffn_down_w_{i}", W["vision_ffn_down_w"][i])
        quant("vision_projector_w", W["encoder_multi_modal_projector_w"])

        # Encoder (18 layers × 4) — fuse gate+up into (D, 2H)
        for i in range(ENC_L):
            quant(f"encoder_attn_qkv_w_{i}", W["encoder_attn_qkv_w"][i])
            quant(f"encoder_attn_o_w_{i}", W["encoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["encoder_ffn_gate_w"][i], W["encoder_ffn_up_w"][i]], dim=1
            ).contiguous()
            quant(f"encoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"encoder_ffn_down_w_{i}", W["encoder_ffn_down_w"][i])

        # Decoder (18 layers × 4)
        for i in range(DEC_L):
            quant(f"decoder_attn_qkv_w_{i}", W["decoder_attn_qkv_w"][i])
            quant(f"decoder_attn_o_w_{i}", W["decoder_attn_o_w"][i])
            gate_up = torch.cat(
                [W["decoder_ffn_gate_w"][i], W["decoder_ffn_up_w"][i]], dim=1
            ).contiguous()
            quant(f"decoder_ffn_gate_up_w_{i}", gate_up)
            quant(f"decoder_ffn_down_w_{i}", W["decoder_ffn_down_w"][i])

        logger.info("FP8 quantized %d GEMM weights (layout=%s)", len(fp8), self.fp8_layout)

    def _quantize_decoder_int8(self) -> None:
        """Pre-quantize the decoder hot-path GEMM weights to INT8."""
        W = self._ckpt_bf16
        store = self._int8_store
        int8_weights = self._int8_weights

        def quant(name: str, w: torch.Tensor):
            # CUTLASS fused INT8 path expects weights as [N, K] ColumnMajor,
            # so transpose once up front and keep per-output-channel scales.
            w_f32 = w.float().transpose(0, 1).contiguous()
            scale_t = torch.clamp(
                w_f32.abs().amax(dim=1) / 127.0, min=1e-12
            ).to(device=w.device, dtype=torch.float32).contiguous()
            q = torch.clamp(
                torch.round(w_f32 / scale_t[:, None]), -127, 127
            ).to(torch.int8).contiguous()
            store.append(q)
            store.append(scale_t)
            int8_weights[name] = (q.data_ptr(), scale_t.data_ptr())
            self._int8_weight_scales[name] = scale_t

        for i in range(DEC_L):
            quant(f"decoder_attn_qkv_w_{i}", W["decoder_attn_qkv_w"][i])
            quant(f"decoder_attn_o_w_{i}", W["decoder_attn_o_w"][i])
            # Separate gate and up for SiLU-gated EVT fusion (same as encoder).
            quant(f"decoder_ffn_gate_w_{i}", W["decoder_ffn_gate_w"][i])
            quant(f"decoder_ffn_up_w_{i}", W["decoder_ffn_up_w"][i])
            quant(f"decoder_ffn_down_w_{i}", W["decoder_ffn_down_w"][i])

        logger.info("INT8 quantized %d decoder GEMM weights", len(int8_weights))

    def _quantize_encoder_int8(self) -> None:
        """Pre-quantize the Gemma-2B encoder GEMM weights to INT8.

        Uses the same per-output-channel symmetric INT8 scheme as the
        decoder path. The merged gate+up weight mirrors the FP8 path to
        enable the single fused gate_geglu_merged → INT8 CUTLASS route.

        Keys written into ``self._int8_weights`` (``encoder_`` prefix):
            encoder_attn_qkv_w_{0..17}, encoder_attn_o_w_{0..17},
            encoder_ffn_gate_up_w_{0..17}  (merged),
            encoder_ffn_down_w_{0..17}
        """
        W = self._ckpt_bf16
        store = self._int8_store   # shared with decoder, keeps tensors alive
        int8_weights = self._int8_weights  # shared dict, encoder_ prefix avoids collision

        def quant(name: str, w: torch.Tensor):
            # CUTLASS rowwise INT8 expects B in [N, K] ColumnMajor layout.
            w_f32 = w.float().transpose(0, 1).contiguous()
            scale_t = torch.clamp(
                w_f32.abs().amax(dim=1) / 127.0, min=1e-12
            ).to(device=w.device, dtype=torch.float32).contiguous()
            q = torch.clamp(
                torch.round(w_f32 / scale_t[:, None]), -127, 127
            ).to(torch.int8).contiguous()
            store.append(q)
            store.append(scale_t)
            int8_weights[name] = (q.data_ptr(), scale_t.data_ptr())
            self._int8_weight_scales[name] = scale_t

        for i in range(ENC_L):
            quant(f"encoder_attn_qkv_w_{i}", W["encoder_attn_qkv_w"][i])
            quant(f"encoder_attn_o_w_{i}", W["encoder_attn_o_w"][i])
            # Keep gate and up SEPARATE for SiLU-gated EVT fusion.
            # The new cutlass_int8_silu_gated_bf16out kernel reads gate_buf
            # produced by the gate GEMM and fuses SiLU(gate)*up in the
            # epilogue, eliminating the separate gate_geglu_merged kernel.
            quant(f"encoder_ffn_gate_w_{i}", W["encoder_ffn_gate_w"][i])
            quant(f"encoder_ffn_up_w_{i}", W["encoder_ffn_up_w"][i])
            quant(f"encoder_ffn_down_w_{i}", W["encoder_ffn_down_w"][i])

        logger.info("INT8 quantized %d encoder GEMM weights", 5 * ENC_L)

    def _quantize_vision_int8(self) -> None:
        """Pre-quantize the SigLIP vision encoder GEMM weights to INT8.

        Uses the same per-output-channel symmetric INT8 scheme.  The
        vision GEMMs (seq=512, VIS_D=1152, VIS_H=4304) all fit inside
        the encoder INT8 scratch buffers that ``Pi05Pipeline`` allocates,
        so no additional device memory is needed.

        Keys written into ``self._int8_weights`` (``vision_`` prefix):
            vision_attn_qkv_w_{0..26}, vision_attn_o_w_{0..26},
            vision_ffn_up_w_{0..26}, vision_ffn_down_w_{0..26}
        """
        W = self._ckpt_bf16
        store = self._int8_store
        int8_weights = self._int8_weights

        def quant(name: str, w: torch.Tensor):
            w_f32 = w.float().transpose(0, 1).contiguous()
            scale_t = torch.clamp(
                w_f32.abs().amax(dim=1) / 127.0, min=1e-12
            ).to(device=w.device, dtype=torch.float32).contiguous()
            q = torch.clamp(
                torch.round(w_f32 / scale_t[:, None]), -127, 127
            ).to(torch.int8).contiguous()
            store.append(q)
            store.append(scale_t)
            int8_weights[name] = (q.data_ptr(), scale_t.data_ptr())
            self._int8_weight_scales[name] = scale_t

        for i in range(VIS_L):
            quant(f"vision_attn_qkv_w_{i}", W["vision_attn_qkv_w"][i])
            quant(f"vision_attn_o_w_{i}", W["vision_attn_o_w"][i])
            quant(f"vision_ffn_up_w_{i}", W["vision_ffn_up_w"][i])
            quant(f"vision_ffn_down_w_{i}", W["vision_ffn_down_w"][i])

        logger.info("INT8 quantized %d vision GEMM weights", 4 * VIS_L)

    def _build_pipeline_weights(self) -> dict:
        """Produce the pointer dict that Pi05Pipeline expects."""
        W = self._ckpt_bf16

        def p(key: str) -> int:
            return W[key].data_ptr()

        def p_list(key: str) -> list[int]:
            t = W[key]
            stride = t.stride(0) * t.element_size()
            base = t.data_ptr()
            return [base + i * stride for i in range(t.shape[0])]

        weights = {
            # Vision BF16
            "vision_patch_embedding_w": p("vision_patch_embedding_w"),
            "vision_patch_embedding_b": p("vision_patch_embedding_b"),
            "vision_position_embedding": p("vision_position_embedding"),
            "vision_pre_attn_norm_w": p_list("vision_pre_attn_norm_w"),
            "vision_pre_attn_norm_b": p_list("vision_pre_attn_norm_b"),
            "vision_pre_ffn_norm_w": p_list("vision_pre_ffn_norm_w"),
            "vision_pre_ffn_norm_b": p_list("vision_pre_ffn_norm_b"),
            "vision_attn_qkv_w": p_list("vision_attn_qkv_w"),  # BF16 fallback
            "vision_attn_qkv_b": p_list("vision_attn_qkv_b"),
            "vision_attn_o_w": p_list("vision_attn_o_w"),
            "vision_attn_o_b": p_list("vision_attn_o_b"),
            "vision_ffn_up_w": p_list("vision_ffn_up_w"),
            "vision_ffn_up_b": p_list("vision_ffn_up_b"),
            "vision_ffn_down_w": p_list("vision_ffn_down_w"),
            "vision_ffn_down_b": p_list("vision_ffn_down_b"),
            "vision_final_norm_w": p("vision_final_norm_w"),
            "vision_final_norm_b": p("vision_final_norm_b"),

            # Encoder
            "encoder_multi_modal_projector_w": p("encoder_multi_modal_projector_w"),
            "encoder_multi_modal_projector_b": p("encoder_multi_modal_projector_b"),
            "encoder_attn_qkv_w": p_list("encoder_attn_qkv_w"),
            "encoder_attn_o_w": p_list("encoder_attn_o_w"),
            "encoder_ffn_gate_w": p_list("encoder_ffn_gate_w"),
            "encoder_ffn_up_w": p_list("encoder_ffn_up_w"),
            "encoder_ffn_down_w": p_list("encoder_ffn_down_w"),

            # Decoder
            "decoder_action_in_proj_w": p("decoder_action_in_proj_w"),
            "decoder_action_in_proj_b": p("decoder_action_in_proj_b"),
            "decoder_action_out_proj_w": p("decoder_action_out_proj_w"),
            "decoder_action_out_proj_b": p("decoder_action_out_proj_b"),
            "decoder_attn_qkv_w": p_list("decoder_attn_qkv_w"),
            "decoder_attn_o_w": p_list("decoder_attn_o_w"),
            "decoder_ffn_gate_w": p_list("decoder_ffn_gate_w"),
            "decoder_ffn_up_w": p_list("decoder_ffn_up_w"),
            "decoder_ffn_down_w": p_list("decoder_ffn_down_w"),

            # FP8 quantized weights
            "fp8": self._fp8_weights,
            "int8": self._int8_weights,
            "fp8_layout": self.fp8_layout,

            # Precomputed decoder styles (numpy bf16 as uint16 view)
            "precomputed": self._precomputed_styles,
        }

        # Runtime LoRA passthrough (Pi05Pipeline auto-detects by key
        # presence). The JAX frontend stashes these in ``_ckpt_bf16``
        # when ``FLASHRT_RUNTIME_LORA`` is set (see
        # ``frontends/jax/pi05_rtx.py::_extract_lora_pairs``). They are
        # stored as full torch tensors of shape ``(L, ...)`` so the
        # pipeline can both read ``.shape[-1]`` (rank/neck detection)
        # AND get a per-layer pointer via ``W[key][i].data_ptr()``.
        # Skipped silently if not present (normal merged-LoRA path).
        for _k in (
            "encoder_ffn_gate_lora_a", "encoder_ffn_gate_lora_b",
            "encoder_ffn_up_lora_a",   "encoder_ffn_up_lora_b",
            "encoder_ffn_gateup_lora_a", "encoder_ffn_gateup_lora_b",
            "encoder_ffn_down_lora_a", "encoder_ffn_down_lora_b",
            "encoder_attn_qkv_lora_a", "encoder_attn_qkv_lora_b",
            "encoder_attn_o_lora_a",   "encoder_attn_o_lora_b",
            "decoder_ffn_gate_lora_a", "decoder_ffn_gate_lora_b",
            "decoder_ffn_up_lora_a",   "decoder_ffn_up_lora_b",
            "decoder_ffn_gateup_lora_a", "decoder_ffn_gateup_lora_b",
            "decoder_ffn_down_lora_a", "decoder_ffn_down_lora_b",
            "decoder_attn_qkv_lora_a", "decoder_attn_qkv_lora_b",
            "decoder_attn_o_lora_a",   "decoder_attn_o_lora_b",
        ):
            if _k in W:
                weights[_k] = W[_k]
        if "runtime_lora_scaling" in W:
            weights["runtime_lora_scaling"] = float(W["runtime_lora_scaling"])

        return weights

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def set_rl_mode(
        self,
        *,
        cfg_enable: bool = True,
        cfg_beta: float = 1.5,
        advantage_positive: bool = True,
    ) -> None:
        """Enable / configure advantage-conditioned RL inference (opt-in).

        Once enabled, subsequent :meth:`set_prompt` calls will build a
        :class:`Pi05CFGPipeline` instead of the standard
        :class:`Pi05Pipeline`. The conditioned prompt has the
        ``"Advantage: positive"`` (or ``"negative"``) tag appended; the
        unconditioned prompt is the original task text. Each denoising
        step runs the action expert twice and combines the two velocity
        predictions with strength ``cfg_beta``.

        Calling this with ``cfg_enable=False`` clears any RL configuration
        so the next :meth:`set_prompt` reverts to the standard pipeline
        (this rebuilds the pipeline so the change takes effect).

        Args:
            cfg_enable: If ``True``, activate CFG inference. If
                ``False``, clear any previous RL configuration.
            cfg_beta: CFG guidance strength. Must be ``>= 1.0``. Common
                deployment range is ``[1.5, 2.5]``. Ignored when
                ``cfg_enable`` is ``False``.
            advantage_positive: Whether the conditioned prompt uses the
                positive advantage tag (the standard "select for high
                advantage" use case). Set ``False`` only for debugging.
        """
        if not cfg_enable:
            self._rl_config = None
            # If a CFG pipeline was previously built, drop it so the
            # next set_prompt rebuilds the standard pipeline.
            if isinstance(self.pipeline, Pi05CFGPipeline):
                self.pipeline = None
                self.current_prompt_len = 0
                self.graph_recorded = False
                self.calibrated = False
            return
        if cfg_beta < 1.0:
            raise ValueError(
                f"cfg_beta must be >= 1.0 (1.0 disables CFG); got {cfg_beta}")
        new_config = {
            "cfg_beta": float(cfg_beta),
            "advantage_positive": bool(advantage_positive),
        }
        if self._rl_config != new_config:
            self._rl_config = new_config
            # Force pipeline rebuild on next set_prompt so the new mode
            # / beta takes effect.
            self.pipeline = None
            self.current_prompt_len = 0
            self.graph_recorded = False
            self.calibrated = False
        logger.info(
            "RL mode enabled: cfg_beta=%.2f, advantage_positive=%s",
            new_config["cfg_beta"], new_config["advantage_positive"])

    def _build_pipeline_for_prompt_len(self, prompt_len: int) -> Pi05Pipeline:
        """Build (don't cache) a fresh Pi05Pipeline shaped for prompt_len.

        Pure factory. Sets vision-INT8 reset flags and restores the
        multi-frame FP8 scales snapshot (no-op if calibration hasn't run
        yet) so the new pipeline inherits the same activation scales as
        the initial 80-sample pass. Caller is responsible for caching.
        """
        logger.info("Building Pi05Pipeline for prompt_len=%d...", prompt_len)
        pipeline_weights = self._build_pipeline_weights()
        pipe = Pi05Pipeline(
            gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
            weights=pipeline_weights,
            num_views=self.num_views,
            max_prompt_len=prompt_len,
            chunk_size=self.chunk_size,
            num_steps=self._num_steps,
            vision_pool_factor=self._vision_pool_factor,
            vision_num_layers=self._vision_num_layers,
            **self._pipeline_precision_kwargs())
        # Static INT8 vision scales are per-pipeline-instance. Reset so
        # the predict-time single-frame fallback collects fresh scales.
        if pipe.use_int8_vision_static:
            pipe.vis_int8_static_calibrated = False
            pipe.vis_int8_static_scales = {}
        # Restore the multi-frame FP8 scales snapshot into the new pipeline
        # (no-op if no snapshot exists yet — first-build flow). _restore_fp8_scales
        # operates on self.pipeline, so temporarily point at the new one.
        prior_pipeline = self.pipeline
        self.pipeline = pipe
        try:
            self._restore_fp8_scales()
        finally:
            self.pipeline = prior_pipeline
        return pipe

    def _get_or_build_pipeline_for_prompt_len(
            self, prompt_len: int) -> tuple[Pi05Pipeline, bool]:
        """Return cached pipeline for prompt_len (build + cache on miss).

        Returns
        -------
        (pipeline, was_built)
            ``was_built`` is True iff a fresh Pi05Pipeline was constructed
            on this call (cache miss). Callers use this to decide whether
            to reset graph_recorded / calibrated flags and to log the
            higher-cost-rebuild event vs the cheap pointer swap.
        """
        cached = self._pipeline_cache.get(prompt_len)
        if cached is not None:
            return cached, False

        pipe = self._build_pipeline_for_prompt_len(prompt_len)
        self._pipeline_cache[prompt_len] = pipe
        n_cached = len(self._pipeline_cache)
        if n_cached >= self._pipeline_cache_warn_threshold:
            logger.warning(
                "Pi05 pipeline cache has %d entries (keys=%s). Each entry "
                "owns ~200-300 MB of scratch + activations. If you expect "
                "this many distinct prompt-lengths, raise the warning "
                "threshold via FLASHRT_PIPELINE_CACHE_WARN=N. Otherwise "
                "check why prompt tokenisation is so unstable across "
                "frames — production Pi0.5 state-in-prompt typically "
                "produces 3-6 lengths per task.",
                n_cached, sorted(self._pipeline_cache.keys()))
        return pipe, True

    def prewarm_prompt_buckets(self, prompt_lens: list[int]) -> None:
        """Pre-build cached pipelines for an explicit list of prompt lengths.

        Pays the per-bucket ~600 ms build cost up-front at startup instead
        of letting it land as per-frame spikes during inference. Each
        bucket pipeline inherits the current FP8 scales snapshot (call
        AFTER ``calibrate_with_real_data`` so the multi-frame scales
        already exist). Idempotent: lengths already cached are skipped.

        Graph capture for each bucket still happens lazily on first
        :meth:`set_prompt` + :meth:`infer` for that bucket, because graph
        recording requires an actual observation for warmup. The
        pre-built pipeline is fully calibrated and autotuned, so the
        lazy graph capture is just one short single-frame warmup pass
        (~120 ms) instead of the full 600 ms rebuild.

        Typical usage::

            api.calibrate_with_real_data(obs_list)
            # OpenArm chocolate_bars seen prompt-lens in calibration:
            api.frontend.prewarm_prompt_buckets([78, 80, 82])
        """
        for plen in prompt_lens:
            if plen in self._pipeline_cache:
                continue
            self._pipeline_cache[plen] = self._build_pipeline_for_prompt_len(plen)
        logger.info(
            "Pi05 pipeline cache: prewarm complete (cached prompt_lens=%s)",
            sorted(self._pipeline_cache.keys()))

    def set_prompt(self, prompt_text: str, state=None) -> None:
        """Tokenise prompt + (re)build the pipeline for the exact prompt length.

        When ``state`` is provided, the prompt is tokenised in the Pi0.5
        ``discrete_state_input`` format
        (``f"Task: {prompt}, State: {state_str};\\nAction: "`` — see
        ``openpi/src/openpi/models/pi0_config.py:29-39``). The state
        vector is first normalised to ``[-1, 1]`` via ``self.norm_stats``
        (matching openpi's ``Normalize`` transform upstream of
        ``TokenizePrompt``), then discretised into 256 bins and embedded
        as text tokens prepended to the prompt.

        Default behaviour rebuilds the pipeline to the actual unpadded
        token count on every state change, because FlashRT does not yet
        apply an attention mask to padded positions. Padded-token
        attention corrupts predictions (chocolate_bars replay
        cos(FlashRT, teleop) regressed from 0.989 to 0.892 when we
        zero-padded to ``self.max_prompt_len``). The rebuild costs ~1 s
        for the ~30% of frames whose discretised state crosses a
        token-count boundary; on the remaining frames the captured graph
        is reused and inference completes in ~220 ms (BF16).

        ``FLASHRT_PAD_STATE=1`` opts into the padded mode (stable
        latency, degraded predictions) for performance experiments. The
        eventual production fix is to add encoder attention masking so
        padded positions are -inf masked out of softmax.

        When ``state`` is ``None`` we keep the legacy fast path: the
        pipeline is rebuilt to exactly the prompt's token length and no
        padding is processed (matches the pre-Pi0.5-state behaviour for
        LIBERO and base pi05).

        When RL mode is enabled (see :meth:`set_rl_mode`), this also
        builds the unconditioned prompt embeddings and uploads both into
        the CFG-aware pipeline.
        """
        if self._rl_config is not None:
            if state is not None:
                logger.warning("set_prompt: RL mode does not yet support "
                               "Pi0.5 discrete-state-input; ignoring state.")
            self._set_prompt_rl(prompt_text)
            return

        # State-in-prompt path: by default rebuild to the exact unpadded
        # token count, because FlashRT does not yet apply an attention
        # mask to padded positions (openpi does — see
        # ``openpi/src/openpi/models/pi0.py:155``). When we tried
        # zero-padding to ``self.max_prompt_len`` so the captured graph
        # could be reused across per-frame state changes (faster, no
        # rebuilds), the padded PAD-id-0 tokens corrupted attention and
        # the chocolate_bars replay regressed from
        # cos(FlashRT, teleop)=0.989 -> 0.892 (||err||=0.47 -> 1.75 rad).
        # The per-frame rebuild cost is ~1 s for the ~30% of frames where
        # the discretised state crosses a token-count boundary (78<->82
        # for OpenArm). Acceptable for parity testing; not yet acceptable
        # for real-time robot control at 25 Hz. Set
        # ``FLASHRT_PAD_STATE=1`` to opt back into padded mode (stable
        # latency, degraded predictions) for performance experiments.
        # Eventual fix: add attention masking to the encoder so padded
        # positions are -inf masked out of softmax, matching openpi.
        is_state_in_prompt = state is not None
        pad_state = os.environ.get("FLASHRT_PAD_STATE") == "1"
        if is_state_in_prompt:
            # Normalise physical state to [-1, 1] before passing to the
            # tokenizer (its digitize bins are in [-1, 1]). openpi does
            # this via the Normalize transform upstream of TokenizePrompt;
            # FlashRT's adapter passes the raw physical state through, so
            # we normalise here using self.norm_stats[state] q01/q99.
            state_norm = self._normalize_state_for_prompt(state)
            embeds, prompt_len = _embed_prompt(
                prompt_text, self.embedding_weight,
                max_len=self.max_prompt_len,
                state=state_norm, pad_to_max=pad_state)
            # Warn if max_prompt_len is too small for the expected state-
            # in-prompt token count. For OpenArm 16-DOF chocolate_bars the
            # discretised prompt is 78-82 tokens; LIBERO 7-DOF is ~45.
            # max_prompt_len < 96 is almost certainly silent truncation.
            if not getattr(self, "_warned_short_max_prompt", False) \
                    and self.max_prompt_len < 96:
                logger.warning(
                    "set_prompt: state-in-prompt mode but "
                    "max_prompt_len=%d is too small for typical "
                    "Pi0.5 state prompts (OpenArm 16-DOF: ~80, "
                    "LIBERO 7-DOF: ~45). The discretised state will "
                    "be silently truncated and the model will see "
                    "only a partial state. Re-init the frontend with "
                    "max_prompt_len>=128.", self.max_prompt_len)
                self._warned_short_max_prompt = True
        else:
            embeds, prompt_len = _embed_prompt(
                prompt_text, self.embedding_weight,
                max_len=MAX_PROMPT_LEN_DEFAULT)

        # Pipeline cache lookup. Three cases:
        #   1) First-ever call: cache empty → build, cache, set as active
        #   2) Same prompt_len as last call: same Pi05Pipeline instance
        #      already active → pointer-stable, just re-upload embeds
        #   3) Different prompt_len already in cache: pointer-swap to
        #      the cached pipeline (no rebuild, no autotune, no warmup)
        # The cache replaces the old "rebuild on every length change"
        # path; per-frame 25 Hz operation in state-in-prompt mode
        # bounces between 3-6 distinct lengths typically and now pays
        # the rebuild cost ONCE per length over the whole task instead
        # of repeatedly.
        prior_pipeline_id = id(self.pipeline) if self.pipeline is not None else None
        new_pipeline, was_built = self._get_or_build_pipeline_for_prompt_len(
            prompt_len)
        pipeline_changed = id(new_pipeline) != prior_pipeline_id
        if pipeline_changed:
            self.pipeline = new_pipeline
            self.current_prompt_len = prompt_len
            # The "is calibrated + graph captured" state lives on the
            # pipeline itself. Read it back to set the frontend's flags
            # correctly:
            #   - Cache MISS (was_built=True): the new pipeline has FP8
            #     scales (restored by _build) but no captured graph yet.
            #     Reset frontend flags so api.predict's fallback fires
            #     calibrate_with_real_data once, which short-circuits
            #     calibrate_fp8 (pipeline.fp8_calibrated=True) and
            #     proceeds straight to record_infer_graph + warmup.
            #   - Cache HIT (was_built=False) with graph NOT YET captured:
            #     same as miss — graph capture is still needed. This
            #     covers the corner case where pipeline_A was built and
            #     cached but never had predict() called before the
            #     caller swapped to pipeline_B and back.
            #   - Cache HIT with graph already captured: leave flags
            #     untouched so api.predict skips its calibrate re-fire.
            #     This is the steady-state case after each bucket has
            #     been used once — the swap is a pure pointer move and
            #     the next infer() is straight graph replay (~165 ms
            #     vs ~600 ms if calibrate re-fired).
            pipeline_has_graph = (
                getattr(new_pipeline, "_graph", None) is not None)
            if was_built or not pipeline_has_graph:
                self.graph_recorded = False
                self.calibrated = False
            else:
                logger.debug(
                    "Pipeline cache HIT for prompt_len=%d (pre-built + "
                    "pre-calibrated + graph captured \u2014 next infer() is "
                    "pure graph replay)", prompt_len)

        # Upload language embeds into pipeline's encoder_x slot
        embeds_np = embeds.contiguous().view(torch.uint16).cpu().numpy()
        self.pipeline.set_language_embeds(embeds_np)
        self._frame_count = 0
        self._current_prompt = prompt_text
        # Per-frame Pi0.5 state-in-prompt calls would flood the log at
        # 25-30 Hz. Three-tier log:
        #   - was_built (cache miss + new build): INFO with build note
        #   - pipeline_changed but cache hit (pointer-swap): DEBUG
        #   - same-pipeline (re-upload embeds only): DEBUG
        if was_built:
            logger.info(
                "Set prompt: '%s' (%d tokens%s, NEW pipeline cached, "
                "cache size=%d)", prompt_text, prompt_len,
                ", state-in-prompt" if is_state_in_prompt else "",
                len(self._pipeline_cache))
        elif pipeline_changed:
            logger.debug(
                "Set prompt: '%s' (%d tokens, pipeline cache hit)",
                prompt_text, prompt_len)
        else:
            logger.debug("Set prompt: '%s' (%d tokens, in-place)",
                         prompt_text, prompt_len)

    def _set_prompt_rl(self, prompt_text: str) -> None:
        """RL-mode set_prompt: build conditioned + unconditioned embeddings.

        When batched mode is also active (Phase 3b), the pipeline type
        is :class:`Pi05CFGBatchedPipeline` which runs cond + uncond as
        the two slots of a B=2 fused forward. Otherwise the serial
        :class:`Pi05CFGPipeline` runs them sequentially (Phase 1+2).
        """
        from flash_rt.core.rl import build_acp_tagged_task

        cfg = self._rl_config
        if cfg is None:
            raise RuntimeError("_set_prompt_rl called without RL config")

        cond_text = build_acp_tagged_task(
            prompt_text, is_positive=cfg["advantage_positive"])
        uncond_text = prompt_text

        cond_embeds, cond_len = _embed_prompt(
            cond_text, self.embedding_weight, max_len=MAX_PROMPT_LEN_DEFAULT)
        uncond_embeds, uncond_len = _embed_prompt(
            uncond_text, self.embedding_weight, max_len=MAX_PROMPT_LEN_DEFAULT)
        target_len = max(cond_len, uncond_len)

        use_batched_cfg = getattr(self, "_batched_active", False)

        if use_batched_cfg:
            expected_cls = Pi05CFGBatchedPipeline
            cls_name = "Pi05CFGBatchedPipeline"
        else:
            expected_cls = Pi05CFGPipeline
            cls_name = "Pi05CFGPipeline"

        rebuild = (
            self.pipeline is None
            or not isinstance(self.pipeline, expected_cls)
            or target_len != self.current_prompt_len
            or self.pipeline.cfg_beta != cfg["cfg_beta"])

        if rebuild:
            logger.info(
                "Building %s for prompt_len=%d (cfg_beta=%.2f)...",
                cls_name, target_len, cfg["cfg_beta"])
            self.current_prompt_len = target_len
            self.graph_recorded = False
            self.calibrated = False

            pipeline_weights = self._build_pipeline_weights()
            if use_batched_cfg:
                # Need the batched attention backend (already set up by
                # set_batched_mode).
                if not isinstance(self.attn_backend,
                                  RtxFlashAttnBatchedBackendPi05):
                    raise RuntimeError(
                        "batched CFG requires set_batched_mode(enable=True) "
                        "to have been called first to install the batched "
                        "attention backend")
                self.pipeline = Pi05CFGBatchedPipeline(
                    gemm=self.gemm, fvk=self.fvk,
                    attn_backend=self.attn_backend,
                    weights=pipeline_weights,
                    num_views=self.num_views,
                    max_prompt_len=target_len,
                    chunk_size=self.chunk_size,
                    **self._pipeline_precision_kwargs(),
                    cfg_beta=cfg["cfg_beta"])
            else:
                self.pipeline = Pi05CFGPipeline(
                    gemm=self.gemm, fvk=self.fvk,
                    attn_backend=self.attn_backend,
                    weights=pipeline_weights,
                    num_views=self.num_views,
                    max_prompt_len=target_len,
                    chunk_size=self.chunk_size,
                    **self._pipeline_precision_kwargs(),
                    cfg_beta=cfg["cfg_beta"])

        cond_np = cond_embeds.contiguous().view(torch.uint16).cpu().numpy()
        uncond_np = uncond_embeds.contiguous().view(torch.uint16).cpu().numpy()

        if use_batched_cfg:
            # Pad both to target_len here (the batched set_language_embeds_batch
            # inherited by Pi05CFGBatchedPipeline expects equal prompt lengths).
            def _pad(arr, to_len):
                if arr.shape[0] == to_len:
                    return np.ascontiguousarray(arr)
                pad = np.zeros((to_len - arr.shape[0], arr.shape[1]),
                               dtype=arr.dtype)
                return np.ascontiguousarray(np.concatenate([arr, pad], axis=0))
            cond_np = _pad(cond_np, target_len)
            uncond_np = _pad(uncond_np, target_len)
            # Also seed parent's B=1 lang slot for the FP8 calibration pass
            # (same pattern set_prompt_batch uses).
            self.pipeline.set_language_embeds(cond_np)

        self.pipeline.set_language_embeds_pair(cond_np, uncond_np)
        self._rl_current_prompt_text = prompt_text
        self._frame_count = 0
        logger.info(
            "Set RL prompt: '%s' (cond_len=%d, uncond_len=%d, padded=%d, batched=%s)",
            prompt_text, cond_len, uncond_len, target_len, use_batched_cfg)

    def calibrate(
        self,
        observations,
        *,
        percentile: float = 99.9,
        max_samples: Optional[int] = None,
        verbose: bool = False,
    ) -> None:
        """Unified calibration entry point (see Pi0TorchFrontendRtx.calibrate).

        N=1 → single-frame path, bit-equal to legacy.
        N>=2 → per-sample amax, reduced via ``np.percentile(..., axis=0)``.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before calibrate")
        if self.calibrated:
            logger.warning(
                "calibrate() called a second time; returning without re-running.")
            return

        if isinstance(observations, dict):
            obs_list = [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = list(observations)
        if max_samples is not None:
            obs_list = obs_list[:max_samples]
        n = len(obs_list)
        if n == 0:
            raise ValueError("observations must contain at least 1 sample")
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile}")

        if getattr(self.pipeline, "use_int8_decoder", False):
            if n > 1:
                logger.info(
                    "INT8 decoder path uses runtime-dynamic activation scales; "
                    "using the first sample to warm buffers and capture the graph.")
            self._calibrate_single_frame(obs_list[0])
            return

        if n == 1:
            self._calibrate_single_frame(obs_list[0])
        else:
            self._calibrate_multi_frame(
                obs_list, percentile=percentile, verbose=verbose)

    def calibrate_with_real_data(self, sample_observations) -> None:
        """Legacy alias for :meth:`calibrate`."""
        self.calibrate(sample_observations)

    def _calibrate_single_frame(self, sample) -> None:
        logger.info("Preparing Pi0.5 runtime with a single real sample...")

        # Create a dedicated torch stream for both the calibration pass and
        # graph capture so flash_attn_func + our fvk kernels land on the
        # same stream.
        self._graph_torch_stream = torch.cuda.Stream()

        with torch.cuda.stream(self._graph_torch_stream):
            images = self._stack_images(sample)
            noise = torch.randn(
                self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")

            stream_int = self._graph_torch_stream.cuda_stream
            self._copy_tensor_to_pipeline_buf_stream(
                images, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                noise, self.pipeline.input_noise_buf, stream_int)

            # Batched pipelines carry their own calibrate_fp8 that drives
            # a parent-B=1 forward internally — calling run_pipeline here
            # would fire the batched path with only the parent's B=1
            # slots populated. Skip the preemptive run for batched
            # subclasses and let calibrate_fp8 do the work.
            if not isinstance(self.pipeline, Pi05BatchedPipeline):
                self.pipeline.run_pipeline(stream=stream_int)

            self._cudart.cudaStreamSynchronize(
                ctypes.c_void_p(stream_int))

            # FP8 calibration (no-op for INT8 pipelines).
            self.pipeline.calibrate_fp8()
            # Static INT8 vision: the run_pipeline() call above already ran one
            # vision forward with quantize_int8_device, writing per-site scales
            # into vis_int8_static_scales. Flip the flag to switch to the fast
            # static path (quantize_int8_static) for all subsequent calls.
            if self.pipeline.use_int8_vision_static:
                self.pipeline.vis_int8_static_calibrated = True
                logger.info("Static INT8 vision calibrated: %d sites",
                            len(self.pipeline.vis_int8_static_scales))
            # Static encoder INT8 (opt-in via FVK_PI05_RTX_INT8_ENCODER_STATIC=1).
            # After run_pipeline() above wrote per-row scales via the
            # dynamic kernel, freeze them and flip the hot path to
            # quantize_int8_rowwise_static (single-pass, no per-row amax
            # reduction).
            #
            # WARNING — measured on Orin SM87, single-frame calibration:
            #   * Latency saving: ~1.4 ms p50 (125.9 → 124.5 ms). Smaller
            #     than the roofline-predicted 4-8 ms because most of the
            #     encoder time is in the CUTLASS GEMM, not the quantize.
            #   * Cosine vs dynamic baseline: drops from 0.991 to
            #     ~0.93-0.98 across a 6-frame test sequence. Failed the
            #     "lossless" bar — frozen per-row scales calibrated on
            #     one sample don't generalize: vision-token rows whose
            #     magnitude exceeds the calibration max get clipped.
            # Default OFF. Opt-in only when the application explicitly
            # accepts this trade-off (or after a future multi-sample
            # calibration with proper safety inflation makes the cosine
            # drop acceptable).
            if (self.pipeline.use_int8_encoder
                    and os.environ.get(
                        "FVK_PI05_RTX_INT8_ENCODER_STATIC", "0") == "1"):
                self.pipeline.int8_encoder_static_calibrated = True
                logger.warning(
                    "Static INT8 encoder enabled — frozen per-row scales "
                    "from one calibration sample. Expect cosine drop "
                    "(~0.96 vs dynamic 0.991 on test sequence). Set "
                    "FVK_PI05_RTX_INT8_ENCODER_STATIC=0 to disable.")
            if not self._gemm_autotune_done:
                self.pipeline.autotune_gemms()
                self._gemm_autotune_done = True
            else:
                logger.info("Skipping autotune_gemms (already tuned in this "
                            "process; cuBLASLt entries are cached on the "
                            "shared GemmRunner across pipeline rebuilds).")
            self.pipeline.record_infer_graph(
                external_stream_int=stream_int, skip_autotune=True)

        self.calibrated = True
        self.graph_recorded = True
        self._precision_spec = self._snapshot_precision_spec(
            method="single_frame", n=1, percentile=None)
        self._warn_if_scale_ceiling_exceeded()
        logger.info("Calibration + graph capture complete")

    def _calibrate_multi_frame(
        self, obs_list, *, percentile: float, verbose: bool,
    ) -> None:
        from flash_rt.core.calibration import (
            accumulate_amax,
            format_summary,
            summarize_amax_dispersion,
        )

        n = len(obs_list)
        logger.info(
            "Preparing Pi0.5 runtime across %d real samples (percentile=%.2f)...",
            n, percentile)
        self._graph_torch_stream = torch.cuda.Stream()
        self.pipeline.fp8_calibrated = False

        # Pi0.5 discrete_state_input: the encoder activation distribution
        # depends on the language tokens, which include the discretised
        # per-frame state. To calibrate scales that cover the actual
        # inference distribution (not just one fixed state), re-fire
        # set_prompt per sample when the obs carries state. Only safe
        # when the captured graph shape is reused — i.e. when
        # FLASHRT_PAD_STATE=1 (constant max_prompt_len) — because a
        # rebuild here would throw away the in-progress calibration. We
        # detect the mode by checking whether the first sample's
        # set_prompt would change prompt_len, and skip per-sample
        # set_prompt if so (a logger warning calls this out so the
        # operator knows the calibration is single-state).
        has_state = any(obs.get("state") is not None for obs in obs_list)
        per_sample_state = False
        if has_state:
            pad_state = os.environ.get("FLASHRT_PAD_STATE") == "1"
            if pad_state:
                per_sample_state = True
                logger.info(
                    "Pi0.5 state-in-prompt calibration: re-firing "
                    "set_prompt with per-sample state across %d samples "
                    "(FLASHRT_PAD_STATE=1 keeps the captured graph "
                    "shape reusable).", n)
            else:
                logger.warning(
                    "Pi0.5 state-in-prompt mode detected but "
                    "FLASHRT_PAD_STATE!=1 so per-frame set_prompt would "
                    "rebuild the pipeline and wipe in-progress FP8 "
                    "scales. Calibrating against a single fixed state "
                    "(whatever the pre-calibration set_prompt was given). "
                    "Production inference with per-frame state will "
                    "trigger rebuilds and the calibrated scales will "
                    "be lost on the first rebuild. Add encoder attention "
                    "masking and set FLASHRT_PAD_STATE=1 for a "
                    "production-viable FP8 path.")

        per_sample: list[np.ndarray] = []
        names: Optional[list[str]] = None

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            for i, obs in enumerate(obs_list):
                # Pi0.5 state-in-prompt: re-tokenise + re-embed for this
                # sample's state so the encoder sees the actual
                # per-frame language distribution. In padded mode the
                # pipeline shape is unchanged so this is just an embeds
                # re-upload (no rebuild, scales preserved).
                if per_sample_state:
                    sample_state = obs.get("state")
                    sample_prompt = obs.get("prompt") or self._current_prompt
                    if sample_prompt is None:
                        raise RuntimeError(
                            "Pi0.5 per-sample state calibration needs a "
                            "prompt — either pass obs['prompt'] in each "
                            "calibration sample, or call set_prompt() "
                            "once before calibrate() so a default exists.")
                    self.set_prompt(sample_prompt, state=sample_state)

                images = self._stack_images(obs)
                noise = torch.randn(
                    self.chunk_size, ACTION_DIM, dtype=bf16, device="cuda")
                self._copy_tensor_to_pipeline_buf_stream(
                    images, self.pipeline.input_images_buf, stream_int)
                self._copy_tensor_to_pipeline_buf_stream(
                    noise, self.pipeline.input_noise_buf, stream_int)
                self._zero_pipeline_scales()
                self.pipeline.run_pipeline(stream=stream_int)
                self._cudart.cudaStreamSynchronize(
                    ctypes.c_void_p(stream_int))

                if names is None:
                    names = list(self.pipeline.fp8_act_scales.keys())
                sample_vec = np.array(
                    [float(self.pipeline.fp8_act_scales[k].download_new(
                        (1,), np.float32)[0]) for k in names],
                    dtype=np.float32)
                per_sample.append(sample_vec)

                if verbose and (i + 1) % max(1, n // 10) == 0:
                    logger.info("  calibration sample %d/%d", i + 1, n)

            final_amax = accumulate_amax(per_sample, percentile=percentile)
            if verbose:
                logger.info(format_summary(
                    summarize_amax_dispersion(per_sample, final_amax)))

            for idx, name in enumerate(names or []):
                self.pipeline.fp8_act_scales[name].upload(
                    np.array([final_amax[idx]], dtype=np.float32))

            self.pipeline.fp8_calibrated = True
            if not self._gemm_autotune_done:
                self.pipeline.autotune_gemms()
                self._gemm_autotune_done = True
            else:
                logger.info("Skipping autotune_gemms (already tuned in this "
                            "process).")
            self.pipeline.record_infer_graph(
                external_stream_int=stream_int, skip_autotune=True)

        self.calibrated = True
        self.graph_recorded = True
        self._snapshot_fp8_scales()
        self._precision_spec = self._snapshot_precision_spec(
            method="percentile", n=n, percentile=percentile)
        self._warn_if_scale_ceiling_exceeded(label=f"pi05_rtx_N{n}")
        logger.info(
            "Pi0.5 multi-frame calibration + graph capture complete "
            "(N=%d, percentile=%.2f)", n, percentile)

    def _zero_pipeline_scales(self) -> None:
        for buf in self.pipeline.fp8_act_scales.values():
            buf.zero_()
        for buf in getattr(self.pipeline, "int8_act_scales", {}).values():
            buf.zero_()

    def _snapshot_fp8_scales(self) -> None:
        """Capture the calibrated FP8 activation scales into a host-side dict.

        Called after a successful multi-frame calibration. The snapshot is
        replayed into every subsequently-rebuilt pipeline by
        :meth:`_restore_fp8_scales`, so per-frame rebuilds in state-in-prompt
        mode don't fall back to single-frame re-calibration (which gives
        scales that don't cover diffusion-noise variance and tanks cos to
        ~0.6 vs the JAX reference).
        """
        snap: dict[str, np.ndarray] = {}
        for name, buf in self.pipeline.fp8_act_scales.items():
            snap[name] = buf.download_new((1,), np.float32).copy()
        self._fp8_scales_snapshot = snap
        if snap:
            logger.info(
                "Snapshotted %d FP8 activation scales for rebuild reuse",
                len(snap))

    def _restore_fp8_scales(self) -> bool:
        """Push the snapshotted FP8 scales into the current pipeline.

        Returns True if scales were restored (and the pipeline marked
        ``fp8_calibrated = True``), False if there's nothing to restore.
        """
        if not self._fp8_scales_snapshot or self.pipeline is None:
            return False
        if not getattr(self.pipeline, "use_fp8", False):
            return False
        for name, val in self._fp8_scales_snapshot.items():
            buf = self.pipeline._fp8_scale_buf(name)
            buf.upload(val.astype(np.float32))
        self.pipeline.fp8_calibrated = True
        logger.info(
            "Restored %d FP8 activation scales from multi-frame snapshot "
            "(skipping per-rebuild single-frame re-calibration)",
            len(self._fp8_scales_snapshot))
        return True

    def _warn_if_scale_ceiling_exceeded(self, label: str = "pi05_rtx") -> None:
        """Diagnostic warning if any FP8 scale exceeds the sanity ceiling."""
        from flash_rt.core.calibration import check_scale_ceiling
        scales = {
            name: float(buf.download_new((1,), np.float32)[0])
            for name, buf in self.pipeline.fp8_act_scales.items()
        }
        check_scale_ceiling(scales, label=label)

    def _snapshot_precision_spec(self, *, method: str, n: int,
                                  percentile: Optional[float]):
        from flash_rt.core.precision_spec import (
            ModelPrecisionSpec,
            PrecisionSpec,
        )

        if getattr(self.pipeline, "use_int8_decoder", False):
            spec = ModelPrecisionSpec(source="manual")
            for name, scale_t in self._int8_weight_scales.items():
                scale_val = scale_t.detach().cpu().numpy().astype(np.float32, copy=False)
                entry = PrecisionSpec(
                    dtype="int8",
                    granularity="per_tensor",
                    scheme="symmetric",
                    scale_source="manual",
                    scale=scale_val,
                )
                entry.validate()
                spec.weight_specs[name] = entry

            for name, buf in self.pipeline.int8_act_scales.items():
                count = buf.nbytes // np.dtype(np.float32).itemsize
                scale_val = buf.download_new((count,), np.float32)
                entry = PrecisionSpec(
                    dtype="int8",
                    granularity="per_tensor",
                    scheme="symmetric",
                    scale_source="runtime_dynamic",
                    scale=scale_val,
                    calibration_method=method,
                    calibration_samples=n,
                    calibration_percentile=percentile,
                )
                entry.validate()
                spec.decoder_layer_specs[name] = entry
            return spec

        spec = ModelPrecisionSpec(source="calibration")
        for name, buf in self.pipeline.fp8_act_scales.items():
            scale_val = float(buf.download_new((1,), np.float32)[0])
            entry = PrecisionSpec(
                dtype="fp8_e4m3",
                granularity="per_tensor",
                scheme="symmetric",
                scale_source="calibration",
                scale=np.array([scale_val], dtype=np.float32),
                calibration_method=method,
                calibration_samples=n,
                calibration_percentile=percentile,
            )
            entry.validate()
            if name.startswith("vision_"):
                spec.activation_specs[name] = entry
            elif name.startswith("encoder_"):
                spec.encoder_layer_specs[name] = entry
            elif name.startswith("decoder_") or name.startswith("action_"):
                spec.decoder_layer_specs[name] = entry
            else:
                spec.activation_specs[name] = entry
        return spec

    @property
    def precision_spec(self):
        """:class:`ModelPrecisionSpec` captured at calibration time."""
        return getattr(self, "_precision_spec", None)

    def infer(self, observation: dict, debug: bool = False) -> dict:
        """Run inference on a single observation.

        All GPU work happens on ``self._graph_torch_stream`` — the same
        stream the graph was captured on — so replay + pre/post D2D copies
        are serialized correctly.

        When the active pipeline is :class:`Pi05CFGBatchedPipeline`
        (RL mode + batched mode both on), this routes through a B=2
        forward that fuses CFG's conditioned and unconditioned branches
        into a single captured graph. The single ``observation`` is
        replicated across both batch slots (cond and uncond use the
        same image / state); the two prompts differ and were already
        uploaded by :meth:`_set_prompt_rl`.
        """
        if self.pipeline is None:
            raise RuntimeError("set_prompt must be called before infer")

        if isinstance(self.pipeline, Pi05CFGBatchedPipeline):
            return self._infer_cfg_batched(observation, debug=debug)

        t0 = time.perf_counter()

        # Temporal K/V caching: every cache_frames-th frame runs the full
        # pipeline (vision + encoder + decoder); intermediate frames skip
        # vision and encoder and replay only the decoder with fresh noise,
        # reusing the encoder K/V cache from the last full forward.
        self._frame_count += 1
        use_full = (self._cache_frames <= 1 or
                    self._frame_count % self._cache_frames == 1)

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            self._noise_buf.normal_()
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf, self.pipeline.input_noise_buf, stream_int)
            self._stage_rtc_inputs(observation, stream_int)

            if use_full:
                self._fill_img_buf(observation)
                self._copy_tensor_to_pipeline_buf_stream(
                    self._img_buf, self.pipeline.input_images_buf, stream_int)
                out_ptr = self.pipeline.forward()
            else:
                # Decode-only: skip vision+encoder, reuse cached K/V
                out_ptr = self.pipeline.forward_decode_only()

            # D2D download → staging torch tensor
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = self._noise_out.float().cpu().numpy()  # (chunk, 32)
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :self.robot_action_dim]

        if debug:
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("Latency: %.1f ms", latency_ms)

        # ``_rtc_chunk_model_space`` returns the **normalized** action
        # chunk (32 dims, ``[-1, 1]``) so RTC clients can feed it back
        # as ``_rtc_prev_chunk`` on the next call. The unnormalized
        # robot-space slice in ``actions`` is what the controller uses.
        return {
            "actions": robot_actions,
            "_rtc_chunk_model_space": raw_actions,
        }

    def _infer_cfg_batched(self, observation: dict,
                           debug: bool = False) -> dict:
        """Batched CFG inference: single obs replicated across cond + uncond slots."""
        t0 = time.perf_counter()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream

            # Replicate the single observation into both batch slots.
            stacked = self._stack_images(observation)
            for b in range(PI05_BATCH_SIZE):
                self._img_buf_b2[b].copy_(stacked)
            # Each denoising step starts from independent noise in each
            # slot; cond slot is the one CFG reads / updates. Sampling
            # once and copying into both slots ensures the uncond slot
            # starts at the same noise the cond does, which matches
            # the paper-faithful CFG contract.
            self._noise_buf.normal_()
            for b in range(PI05_BATCH_SIZE):
                self._noise_buf_b2[b].copy_(self._noise_buf)

            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf_b2, self.pipeline.input_images_buf_b2, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf_b2, self.pipeline.input_noise_buf_b2, stream_int)

            # Graph replay returns the cond slot's noise pointer.
            out_ptr = self.pipeline.forward()

            # D2D download of just the cond slot (chunk * ACTION_DIM bf16)
            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        raw_actions = self._noise_out.float().cpu().numpy()
        unnorm = unnormalize_actions(raw_actions, self.norm_stats)
        robot_actions = unnorm[:, :self.robot_action_dim]

        if debug:
            logger.info(
                "CFG batched raw actions[0,:5]: %s", raw_actions[0, :5])
            logger.info("CFG batched latency: %.1f ms", latency_ms)

        return {"actions": robot_actions}

    # -----------------------------------------------------------------
    # Batched (B=2) inference path — additive, default API unchanged
    # -----------------------------------------------------------------

    def set_batched_mode(self, *, enable: bool = True) -> None:
        """Enable / disable the B=2 batched inference path (opt-in).

        Once enabled, the next :meth:`set_prompt_batch` call builds a
        :class:`Pi05BatchedPipeline` (with a
        :class:`RtxFlashAttnBatchedBackendPi05` attention backend) and
        :meth:`infer_batch` becomes available. The single-sample
        :meth:`infer` API path remains untouched.

        Disabling rebuilds the standard single-sample pipeline on the
        next :meth:`set_prompt`.
        """
        if not enable:
            if isinstance(self.pipeline, Pi05BatchedPipeline):
                self.pipeline = None
                self.current_prompt_len = 0
                self.graph_recorded = False
                self.calibrated = False
                self._batched_active = False
            return
        # Switch to a batched-capable attention backend if not already.
        if not isinstance(self.attn_backend, RtxFlashAttnBatchedBackendPi05):
            enc_seq_max = self.num_views * 256 + self.max_prompt_len
            self.attn_backend = RtxFlashAttnBatchedBackendPi05(
                num_views=self.num_views,
                encoder_seq_max=enc_seq_max,
                chunk_size=self.chunk_size,
                num_encoder_layers=ENC_L)
        self._batched_active = True
        # Force pipeline rebuild so set_prompt_batch picks the batched class.
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            self.pipeline = None
            self.current_prompt_len = 0
            self.graph_recorded = False
            self.calibrated = False
        # Pre-allocate batched input/output staging tensors.
        self._img_buf_b2 = torch.empty(
            PI05_BATCH_SIZE, self.num_views, IMG_HW, IMG_HW, 3,
            dtype=bf16, device="cuda")
        self._noise_buf_b2 = torch.empty(
            PI05_BATCH_SIZE, self.chunk_size, ACTION_DIM,
            dtype=bf16, device="cuda")
        self._noise_out_b2 = torch.empty(
            PI05_BATCH_SIZE, self.chunk_size, ACTION_DIM,
            dtype=bf16, device="cuda")
        logger.info(
            "Pi05TorchFrontendRtx: batched mode enabled (B=%d)",
            PI05_BATCH_SIZE)

    def set_prompt_batch(self, prompts: list) -> None:
        """Set per-sample prompts for the batched pipeline.

        Args:
            prompts: list of length B (currently 2). Each entry is a
                task description string. Prompts are individually
                tokenised, then padded to a common length so the
                encoder sees a fixed-shape buffer.
        """
        if not getattr(self, "_batched_active", False):
            raise RuntimeError(
                "set_batched_mode(enable=True) must be called before "
                "set_prompt_batch")
        if len(prompts) != PI05_BATCH_SIZE:
            raise ValueError(
                f"set_prompt_batch expects {PI05_BATCH_SIZE} prompts, "
                f"got {len(prompts)}")
        embeds_list = []
        prompt_lens = []
        for p in prompts:
            e, plen = _embed_prompt(p, self.embedding_weight,
                                    max_len=MAX_PROMPT_LEN_DEFAULT)
            embeds_list.append(e)
            prompt_lens.append(plen)
        target_len = max(prompt_lens)

        # Pad each embed to target_len (BF16 zeros are valid pad tokens).
        padded_np_list = []
        for e, plen in zip(embeds_list, prompt_lens):
            arr = e.contiguous().view(torch.uint16).cpu().numpy()
            if plen < target_len:
                pad = np.zeros(
                    (target_len - plen, arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            padded_np_list.append(np.ascontiguousarray(arr))

        rebuild = (
            self.pipeline is None
            or not isinstance(self.pipeline, Pi05BatchedPipeline)
            or target_len != self.current_prompt_len)

        if rebuild:
            logger.info(
                "Building Pi05BatchedPipeline (B=%d) for prompt_len=%d...",
                PI05_BATCH_SIZE, target_len)
            self.current_prompt_len = target_len
            self.graph_recorded = False
            self.calibrated = False
            pipeline_weights = self._build_pipeline_weights()
            self.pipeline = Pi05BatchedPipeline(
                gemm=self.gemm, fvk=self.fvk, attn_backend=self.attn_backend,
                weights=pipeline_weights,
                num_views=self.num_views,
                max_prompt_len=target_len,
                chunk_size=self.chunk_size,
                **self._pipeline_precision_kwargs())
        # B=1 pipeline path is what calibrate_fp8 uses for FP8 scale collection.
        self.pipeline.set_language_embeds(padded_np_list[0])
        self.pipeline.set_language_embeds_batch(padded_np_list)
        self._frame_count = 0
        logger.info(
            "Set batch prompt (B=%d, padded_len=%d): %s",
            PI05_BATCH_SIZE, target_len,
            [p[:30] + ("…" if len(p) > 30 else "") for p in prompts])

    def calibrate_batch(self, sample_observations) -> None:
        """Calibrate FP8 scales for the batched pipeline.

        Uses the parent B=1 calibration pass (per-tensor scales are
        sample-invariant) on the first observation; the batched B=2
        forward then reuses those scales.
        """
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            raise RuntimeError(
                "calibrate_batch requires set_prompt_batch to have built a "
                "Pi05BatchedPipeline first")
        if isinstance(sample_observations, dict):
            sample_observations = [sample_observations]
        sample = sample_observations[0]

        # Mirror calibrate(): write inputs into the parent B=1 buffers,
        # call parent's calibrate_fp8 + autotune + record graph.
        self._graph_torch_stream = torch.cuda.Stream()
        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            images = self._stack_images(sample)
            noise = torch.randn(self.chunk_size, ACTION_DIM,
                                dtype=bf16, device="cuda")
            self._copy_tensor_to_pipeline_buf_stream(
                images, self.pipeline.input_images_buf, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                noise, self.pipeline.input_noise_buf, stream_int)
            self.pipeline.calibrate_fp8()
            if not self._gemm_autotune_done:
                self.pipeline.autotune_gemms()
                self._gemm_autotune_done = True
            else:
                logger.info("Skipping autotune_gemms (already tuned in this "
                            "process; batched-CFG path).")
            self.pipeline.record_infer_graph(
                external_stream_int=stream_int, skip_autotune=True)
        self.calibrated = True
        self.graph_recorded = True

    def infer_batch(self, observations: list) -> list:
        """Run B=2 inference on two independent observations.

        Args:
            observations: list of length B (currently 2) of obs dicts
                matching :meth:`infer`'s contract (``image``,
                ``wrist_image`` if ``num_views >= 2``, ``state``).

        Returns:
            List of length B; each entry is ``{"actions": (action_horizon, action_dim)}``.
        """
        if not isinstance(self.pipeline, Pi05BatchedPipeline):
            raise RuntimeError("set_batched_mode + set_prompt_batch required")
        if len(observations) != PI05_BATCH_SIZE:
            raise ValueError(
                f"infer_batch expects {PI05_BATCH_SIZE} observations, "
                f"got {len(observations)}")
        t0 = time.perf_counter()

        # Stage per-sample inputs into the B=2 staging tensors, then D2D.
        for b, obs in enumerate(observations):
            self._img_buf_b2[b].copy_(self._stack_images(obs))
        self._noise_buf_b2.normal_()

        with torch.cuda.stream(self._graph_torch_stream):
            stream_int = self._graph_torch_stream.cuda_stream
            self._copy_tensor_to_pipeline_buf_stream(
                self._img_buf_b2, self.pipeline.input_images_buf_b2, stream_int)
            self._copy_tensor_to_pipeline_buf_stream(
                self._noise_buf_b2, self.pipeline.input_noise_buf_b2, stream_int)

            out_ptr = self.pipeline.forward()

            self._cudart.cudaMemcpyAsync(
                ctypes.c_void_p(self._noise_out_b2.data_ptr()),
                ctypes.c_void_p(out_ptr),
                self._noise_out_b2.numel() * 2, 3, stream_int)

        self._cudart.cudaStreamSynchronize(
            ctypes.c_void_p(self._graph_torch_stream.cuda_stream))

        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)

        results = []
        for b in range(PI05_BATCH_SIZE):
            raw = self._noise_out_b2[b].float().cpu().numpy()
            unnorm = unnormalize_actions(raw, self.norm_stats)
            results.append({"actions": unnorm[:, :self.robot_action_dim]})
        return results

    def get_latency_stats(self) -> dict:
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "std_ms": float(np.std(lat)),
            "min_ms": float(np.min(lat)),
            "max_ms": float(np.max(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "hz": float(1000 / np.mean(lat)),
        }

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------

    def _stack_images(self, observation: dict) -> torch.Tensor:
        """Stack and normalize observation images into a new bf16 tensor."""
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        tensors = []
        for im in img_list[:self.num_views]:
            tensors.append(
                torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0).to("cuda", bf16))
        return torch.stack(tensors)

    def _fill_img_buf(self, observation: dict) -> None:
        """Fill ``self._img_buf`` in place without allocating new tensors."""
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
            if self.num_views >= 3 and "wrist_image_right" in observation:
                img_list.append(observation["wrist_image_right"])
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(bf16))

    def _copy_tensor_to_pipeline_buf(self, src: torch.Tensor, dst_buf) -> None:
        """D2D cudaMemcpyAsync from a torch tensor into a CudaBuffer slot.

        Uses the current torch stream so downstream ops see the copy.
        """
        stream_int = torch.cuda.current_stream().cuda_stream
        self._copy_tensor_to_pipeline_buf_stream(src, dst_buf, stream_int)

    def _copy_tensor_to_pipeline_buf_stream(
            self, src: torch.Tensor, dst_buf, stream_int: int) -> None:
        """D2D cudaMemcpyAsync on a specific stream."""
        nbytes = src.numel() * src.element_size()
        assert nbytes == dst_buf.nbytes, \
            f"size mismatch: src {nbytes} vs dst {dst_buf.nbytes}"
        self._cudart.cudaMemcpyAsync(
            dst_buf.ptr, ctypes.c_void_p(src.data_ptr()), nbytes, 3, stream_int)

    def _stage_rtc_inputs(self, observation: dict, stream_int: int) -> None:
        """Fill the pipeline's RTC soft-guidance buffers from ``observation``.

        Phase 6 (G11) replacement for hard-freeze inpainting. Builds
        the zero-padded previous chunk + per-position weights tensor
        and uploads both to the pipeline. The captured per-step kernel
        ``rtc_guidance_correction_bf16`` then nudges the velocity field
        toward continuity at each Euler step. See
        ``Pi05Pipeline._rtc_apply_guidance`` for the algorithm,
        ``docs/spark_phase6_soft_guidance.md`` for the math, and
        ``third_party/lerobot_rtc_reference/modeling_rtc.py`` for the
        upstream reference.

        Recognised observation keys (all RTC-related keys default
        sensibly when missing; only ``_rtc_prev_chunk`` + a positive
        ``_rtc_inference_delay`` are required to activate guidance):

        ``_rtc_prev_chunk``
            ``np.ndarray`` of shape ``(d, 32)`` (model-space, normalized
            to ``[-1, 1]``) holding the previous chunk's actions
            starting at the splice position. Effectively the inflight
            prefix that the new chunk must continue smoothly. Padded
            with zeros to ``(chunk_size, 32)`` before upload; padded
            positions have weight 0 so they're inert in the kernel.

        ``_rtc_inference_delay``
            Integer ``d`` ∈ ``[1, chunk_size]``. The number of leading
            new-chunk positions strongly anchored to the prefix
            (weighted 1.0 in the kernel). Must equal
            ``_rtc_prev_chunk.shape[0]``.

        ``_rtc_execution_horizon`` (optional)
            Integer ``end`` ∈ ``[d, chunk_size]``. The merge window
            extends from ``d`` to ``end``; positions ``[end, chunk_size)``
            are free (weight 0). Defaults to
            ``self._rtc_execution_horizon`` (lerobot default 10).
            Capped at the prefix length per
            ``RTCProcessor.denoise_step:189-190``.

        ``_rtc_schedule`` (optional)
            One of ``"linear"``, ``"exp"``, ``"ones"``, ``"zeros"``.
            Defaults to ``self._rtc_schedule`` (lerobot default
            ``"linear"``). Controls the ramp shape in the merge window.

        For non-RTC traffic (no prev chunk / d=0) this re-zeros the
        weight buffer iff the previous call was active (cheap ~3 KB
        upload), so stale weights don't bleed across inferences.
        """
        d_raw = observation.get("_rtc_inference_delay", 0)
        prev = observation.get("_rtc_prev_chunk", None)
        try:
            d = int(d_raw)
        except (TypeError, ValueError):
            d = 0
        prev_shape = getattr(prev, "shape", None)
        prev_len = int(prev_shape[0]) if (
            prev_shape is not None and len(prev_shape) >= 1) else 0
        active = (
            prev is not None
            and d > 0
            and d <= self.chunk_size
            and prev_shape is not None
            and len(prev_shape) == 2
            and prev_shape[1] == ACTION_DIM
            and prev_len >= d)

        if not active:
            if not self._rtc_last_call_active:
                return
            # Zero only the weights buffer — prev_chunk is irrelevant
            # when weights are all zero. Save one DMA upload.
            self._rtc_weights_buf.zero_()
            self._copy_tensor_to_pipeline_buf_stream(
                self._rtc_weights_buf,
                self.pipeline.rtc_weights_buf, stream_int)
            self._rtc_last_call_active = False
            return

        # Resolve per-call config (fall back to frontend defaults).
        exec_horizon_raw = observation.get(
            "_rtc_execution_horizon", self._rtc_execution_horizon)
        try:
            exec_horizon = int(exec_horizon_raw)
        except (TypeError, ValueError):
            exec_horizon = self._rtc_execution_horizon
        # ``RTCProcessor.denoise_step:189-190``: can't merge past the
        # end of what the client sent — cap end at the available prev
        # length. If the client only sent ``d`` positions (legacy/
        # hard-freeze format) the merge window collapses to empty and
        # we effectively do hard anchor + free continuation. If the
        # client sent the full unconsumed tail (FlashRT G11+) we get
        # a proper merge window up to ``exec_horizon``.
        exec_horizon = min(exec_horizon, prev_len)
        exec_horizon = min(exec_horizon, self.chunk_size)
        # ``RTCProcessor.get_prefix_weights:252``: start = min(start, end).
        # Keep that semantics by clamping d at exec_horizon when the
        # merge window has collapsed — otherwise ``start=d > end`` would
        # extend the weight=1.0 region past where prev actually has
        # values to anchor against.
        d_eff = min(d, exec_horizon) if exec_horizon > 0 else d
        schedule = observation.get("_rtc_schedule", self._rtc_schedule)

        # Pad / truncate prev to (chunk_size, ACTION_DIM); trailing rows
        # are zero and (paired with weight 0) inert in the kernel.
        prev_np = np.zeros(
            (self.chunk_size, ACTION_DIM), dtype=np.float32)
        prev_clip = min(prev_len, self.chunk_size)
        prev_np[:prev_clip, :] = np.ascontiguousarray(
            prev[:prev_clip], dtype=np.float32)

        # Per-position weights, broadcast across the action_dim axis so
        # the in-graph kernel can do element-wise multiply without a
        # broadcast op (the C++ side wants a flat (ds*ACTION_DIM,) bf16
        # buffer).
        weights_1d = _get_prefix_weights(
            start=d_eff, end=exec_horizon, total=self.chunk_size,
            schedule=schedule)
        weights_full = np.broadcast_to(
            weights_1d[:, None],
            (self.chunk_size, ACTION_DIM)).copy()

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
