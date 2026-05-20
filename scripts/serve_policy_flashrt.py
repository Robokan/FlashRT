"""Drop-in replacement for `scripts/serve_policy.py` that uses FlashRT for
inference instead of the openpi JAX/PyTorch reference path.

Designed to be wire-compatible with existing robot clients
(``AsyncActionChunkBroker`` over websockets) so the only change is the
server-side `--policy.config` knob.

Usage (inside the flashrt:spark container)::

    python3 scripts/serve_policy_flashrt.py \\
        --checkpoint /openpi_assets/pi05_openarm_ngc_lora_v4 \\
        --robot-action-dim 16 \\
        --calib-data /openpi_assets/calib_openarm_80.npz \\
        --default-prompt "pick up the red block" \\
        --port 8001

If --calib-data is omitted the server will lazy-calibrate on the first
robot frame (adds ~3 s latency to the first inference). Prefer to
pre-calibrate, which is also what Phase 3 verified.
"""

from __future__ import annotations

import argparse
import logging
import socket
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve FlashRT pi05 over the openpi websocket protocol")
    parser.add_argument("--checkpoint", required=True,
                        help="Pi0.5 Orbax JAX checkpoint dir")
    parser.add_argument("--framework", default="jax", choices=["jax", "torch"],
                        help="FlashRT frontend (default jax for Orbax checkpoints)")
    parser.add_argument("--robot-action-dim", type=int, default=None,
                        help="Robot action dimensions to slice from the 32-dim model "
                             "output. Default = LIBERO_ACTION_DIM (7). OpenArm bimanual = 16.")
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--autotune", type=int, default=3,
                        help="CUDA Graph autotune intensity (0=off, 3=default, 5=thorough)")
    parser.add_argument("--default-prompt", default=None,
                        help="Prompt fallback when the client sends no 'prompt' field")
    parser.add_argument("--calib-data", default=None,
                        help="Optional path to an npz of stratified observations "
                             "(see scripts/spark_phase3_prepare_calib.py). "
                             "Triggers eager calibration before the server starts.")
    parser.add_argument("--port", type=int, default=8001,
                        help="Websocket port (default 8001 to avoid clashing with "
                             "the JAX server's 8000 / openpi default)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--metadata-config", default=None,
                        help="Optional path to a JSON file with extra metadata "
                             "fields to expose at the websocket handshake.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    # 1. Load FlashRT model.
    try:
        import flash_rt
    except ImportError as e:
        logger.error("Could not import flash_rt: %s. "
                     "Run inside the flashrt:spark container.", e)
        return 1

    logger.info("Loading FlashRT model from %s (framework=%s, robot_action_dim=%s)",
                args.checkpoint, args.framework, args.robot_action_dim)
    model = flash_rt.load_model(
        checkpoint=args.checkpoint,
        framework=args.framework,
        num_views=args.num_views,
        autotune=args.autotune,
        robot_action_dim=args.robot_action_dim,
    )

    # 2. Eager calibrate if asked.
    if args.calib_data:
        calib_path = Path(args.calib_data)
        if not calib_path.is_file():
            logger.error("--calib-data %s not found", calib_path)
            return 1
        data = np.load(calib_path, allow_pickle=True)
        prompts = data["prompts"]
        first_prompt = (str(prompts[0]) if len(prompts) > 0 and prompts[0]
                        else args.default_prompt or "pick up the red block")
        obs_list = [{"image": im, "wrist_image": wr}
                    for im, wr in zip(data["images"], data["wrist_images"])]
        model._pipe.set_prompt(first_prompt)
        model._current_prompt = first_prompt
        logger.info("Calibrating with %d samples (percentile=99.9)", len(obs_list))
        model.calibrate(obs_list, percentile=99.9)
        logger.info("Calibration complete; first robot frame will replay the CUDA graph immediately.")

    # 3. Build metadata. Default mirrors the openpi policy metadata
    # shape (the AsyncActionChunkBroker uses `chunk_size` from this).
    metadata: dict = {
        "model": "flash_rt.pi05",
        "framework": args.framework,
        "chunk_size": model._pipe.chunk_size,
        "robot_action_dim": model._pipe.robot_action_dim,
    }
    if args.metadata_config:
        import json
        with open(args.metadata_config) as f:
            metadata.update(json.load(f))

    # 4. Wrap as a BasePolicy.
    from flash_rt.serving.openpi_adapter import FlashRTPolicyAdapter
    adapter = FlashRTPolicyAdapter(
        model,
        default_prompt=args.default_prompt,
        chunk_size=model._pipe.chunk_size,
        metadata=metadata,
    )

    # 5. Serve.
    # The WebsocketPolicyServer lives in the main openpi package (not in
    # the lightweight openpi-client). To install:
    #     pip install -e ../openpi[serving]   # editable from a sibling checkout
    # The FlashRTPolicyAdapter above only needs openpi-client.
    try:
        from openpi.serving import websocket_policy_server
    except ImportError as e:
        logger.error(
            "Could not import openpi.serving.websocket_policy_server: %s. "
            "This server script needs the openpi package on PYTHONPATH. "
            "On Spark, mount the openpi repo and set PYTHONPATH=/openpi/src, "
            "or pip install -e <openpi-root>.", e,
        )
        return 1
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "?"
    logger.info("Creating server (host: %s, ip: %s, port: %d)", hostname, local_ip, args.port)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=adapter,
        host=args.host,
        port=args.port,
        metadata=metadata,
    )
    logger.info("Ready. AsyncActionChunkBroker can connect to ws://%s:%d",
                local_ip, args.port)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
