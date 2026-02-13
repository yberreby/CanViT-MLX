"""Load a CanViT model from pre-converted MLX safetensors checkpoint.

Use convert.py to convert PyTorch HF Hub weights to MLX format first.
"""

__all__ = ["load_canvit"]

import json
import logging
from pathlib import Path

import mlx.core as mx

from .canvit import CanViT
from .config import CanViTConfig

log = logging.getLogger(__name__)


def load_canvit(weights_path: str, config_path: str | None = None) -> CanViT:
    wp = Path(weights_path)
    cp = Path(config_path) if config_path else wp.with_suffix(".json")

    mc = json.loads(cp.read_text())["model_config"]
    cfg = CanViTConfig(**mc)
    log.info("Config: %s", mc)

    model = CanViT(cfg)

    from safetensors import safe_open
    weights: list[tuple[str, mx.array]] = []
    with safe_open(str(wp), framework="numpy") as f:
        for key in f.keys():
            weights.append((key, mx.array(f.get_tensor(key))))
    model.load_weights(weights)
    log.info("Loaded %d tensors from %s", len(weights), wp)
    return model
