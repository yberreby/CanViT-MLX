"""Load a CanViT model from pre-converted MLX safetensors.

Supports local paths and HuggingFace Hub repo IDs.
HF Hub loading requires the [hub] extra (huggingface_hub).
Use convert.py + push_to_hub.py to create and upload MLX weights.
"""

__all__ = ["load_canvit"]

import json
import logging
from pathlib import Path

import mlx.core as mx

from .canvit import CanViT
from .config import CanViTConfig

log = logging.getLogger(__name__)

DEFAULT_HF_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx"


def _load_local(weights_path: Path, config_path: Path) -> CanViT:
    mc = json.loads(config_path.read_text())["model_config"]
    cfg = CanViTConfig(**mc)
    model = CanViT(cfg)

    from safetensors import safe_open
    weights: list[tuple[str, mx.array]] = []
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            weights.append((key, mx.array(f.get_tensor(key))))
    model.load_weights(weights)
    log.info("Loaded %d tensors from %s", len(weights), weights_path)
    return model


def _download_from_hub(repo_id: str) -> tuple[Path, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            f"huggingface_hub is required to load from HF Hub ({repo_id!r}). Install the [hub] extra to enable this."
        ) from None
    log.info("Downloading from HF Hub: %s", repo_id)
    weights = Path(hf_hub_download(repo_id, "model.safetensors"))
    config = Path(hf_hub_download(repo_id, "config.json"))
    return weights, config


def load_canvit(source: str = DEFAULT_HF_REPO) -> CanViT:
    """Load CanViT from a local .safetensors path or HF Hub repo ID."""
    local = Path(source)
    if local.exists():
        return _load_local(local, local.with_suffix(".json"))
    weights, config = _download_from_hub(source)
    return _load_local(weights, config)
