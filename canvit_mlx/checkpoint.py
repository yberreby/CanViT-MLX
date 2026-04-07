"""Load a CanViT model from pre-converted MLX safetensors.

Provides two entry points:
  - load_from_local: from local .safetensors + config.json
  - load_from_hf_hub: download from HuggingFace Hub, then load
"""

__all__ = ["load_from_local", "load_from_hf_hub"]

import json
import logging
from pathlib import Path

import mlx.core as mx

from .canvit import CanViT, CanViTForPretraining
from .config import CanViTConfig

log = logging.getLogger(__name__)

DEFAULT_HF_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-mlx"


def _load_weights(weights_path: Path) -> list[tuple[str, mx.array]]:
    from safetensors import safe_open
    weights: list[tuple[str, mx.array]] = []
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            weights.append((key, mx.array(f.get_tensor(key))))
    return weights


def _config_from_json(config_path: Path) -> CanViTConfig:
    mc = json.loads(config_path.read_text())["model_config"]
    return CanViTConfig(**mc)


def load_from_local(weights_path: Path, config_path: Path) -> CanViTForPretraining:
    """Load CanViTForPretraining from local .safetensors + config.json."""
    cfg = _config_from_json(config_path)
    model = CanViTForPretraining(cfg)

    weights = _load_weights(weights_path)
    model.load_weights(weights)
    log.info("Loaded %d tensors from %s", len(weights), weights_path)
    return model


def _download_hf_from_hub(repo_id: str) -> tuple[Path, Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            f"huggingface_hub is required to load from HF Hub ({repo_id!r}). Install the [hub] extra to enable this."
        ) from e
    log.info("Downloading from HF Hub: %s", repo_id)
    weights = Path(hf_hub_download(repo_id, "model.safetensors"))
    config = Path(hf_hub_download(repo_id, "config.json"))
    return weights, config


def load_from_hf_hub(repo_id: str) -> CanViTForPretraining:
    """Download CanViTForPretraining weights from HF Hub and load."""
    weights, config = _download_hf_from_hub(repo_id)
    return load_from_local(weights, config)
