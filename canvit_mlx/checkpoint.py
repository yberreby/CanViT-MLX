"""Load a CanViT model from safetensors checkpoint (local or HuggingFace Hub)."""

__all__ = ["load_canvit"]

import json
import logging
from pathlib import Path

import mlx.core as mx

from .canvit import CanViT
from .config import CanViTConfig

log = logging.getLogger(__name__)

DEFAULT_HF_REPO = "canvit/canvitb16-vpe-pretrain-g128px-s512px-in21k-dv3b16"


def _load_from_local(weights_path: Path, config_path: Path) -> CanViT:
    mc = json.loads(config_path.read_text())["model_config"]
    cfg = CanViTConfig(**mc)
    log.info("Config: %s", mc)
    model = CanViT(cfg)
    from safetensors import safe_open
    weights: list[tuple[str, mx.array]] = []
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            weights.append((key, mx.array(f.get_tensor(key))))
    model.load_weights(weights)
    log.info("Loaded %d tensors from %s", len(weights), weights_path)
    return model


def load_canvit(source: str = DEFAULT_HF_REPO, config_path: str | None = None) -> CanViT:
    """Load CanViT from a local path or HuggingFace Hub repo ID.

    If source is a local file path, loads directly.
    Otherwise, treats it as a HF Hub repo ID and downloads.
    """
    local = Path(source)
    if local.exists():
        cp = Path(config_path) if config_path else local.with_suffix(".json")
        return _load_from_local(local, cp)

    # Treat as HF Hub repo ID
    from huggingface_hub import hf_hub_download
    log.info("Downloading from HuggingFace Hub: %s", source)
    weights_path = Path(hf_hub_download(source, "model.safetensors"))
    cp = Path(hf_hub_download(source, "config.json"))
    return _load_from_local(weights_path, cp)
