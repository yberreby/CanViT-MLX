"""CanViT for image classification: backbone + LN → Linear head."""

__all__ = ["CanViTForImageClassification", "fuse_probe"]

import logging
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from .canvit import CanViT, CanViTForPretraining, CanViTOutput, RecurrentState
from .checkpoint import _config_from_json, _load_weights, load_from_local
from .config import CanViTConfig
from .viewpoint import Viewpoint

log = logging.getLogger(__name__)

# Pretraining-head key prefixes — filtered out when copying backbone weights
_PRETRAINING_PREFIXES = (
    "scene_patches_ln.", "scene_patches_proj.",
    "scene_cls_ln.", "scene_cls_proj.",
    "cls_std_mean", "cls_std_var",
    "scene_std_mean", "scene_std_var",
)


def fuse_probe(
    *,
    W_proj: mx.array,
    b_proj: mx.array,
    mu: mx.array,
    sigma: mx.array,
    W_probe: mx.array,
    b_probe: mx.array,
) -> tuple[mx.array, mx.array]:
    """Fuse proj → destandardize → probe into a single linear transform.

    The pretrained eval chain after LayerNorm is three affine transforms::

        s = W_proj @ z + b_proj          (projection)
        d = sigma * s + mu               (destandardization)
        logits = W_probe @ d + b_probe   (probe)

    Since affine . affine = affine, these collapse into::

        W_fused = W_probe @ diag(sigma) @ W_proj
        b_fused = W_probe @ (sigma * b_proj + mu) + b_probe

    Returns (W_fused [n_classes, D], b_fused [n_classes]).
    """
    teacher_dim, D = W_proj.shape
    n_classes = W_probe.shape[0]
    assert b_proj.shape == (teacher_dim,), f"b_proj shape {b_proj.shape} != ({teacher_dim},)"
    assert mu.shape == (teacher_dim,) and sigma.shape == (teacher_dim,)
    assert W_probe.shape == (n_classes, teacher_dim) and b_probe.shape == (n_classes,)

    B_mat = mx.expand_dims(sigma, 1) * W_proj
    b_mid = sigma * b_proj + mu
    W_fused = W_probe @ B_mat
    b_fused = W_probe @ b_mid + b_probe
    return W_fused, b_fused


class CanViTForImageClassification(nn.Module):
    """CanViT backbone + LN -> Linear classification head.

    The backbone is a base CanViT (no pretraining heads).
    The classification head is always LN(D) -> Linear(D, n_classes).
    """

    def __init__(self, cfg: CanViTConfig, n_classes: int):
        super().__init__()
        self.canvit = CanViT(cfg)
        D = cfg.embed_dim
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, n_classes)

    @property
    def cfg(self) -> CanViTConfig:
        return self.canvit.cfg

    @property
    def n_classes(self) -> int:
        return self.head.weight.shape[0]

    def init_state(self, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        return self.canvit.init_state(batch_size, canvas_grid_size)

    def __call__(
        self, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint,
    ) -> tuple[mx.array, RecurrentState]:
        """Returns (logits [B, n_classes], new_state)."""
        out = self.canvit(glimpse, state, viewpoint)
        cls = out.state.recurrent_cls[:, 0]
        return self.head(self.norm(cls)), out.state

    def canvit_forward(
        self, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint,
    ) -> CanViTOutput:
        """Run CanViT only (for training with separate head_forward)."""
        return self.canvit(glimpse, state, viewpoint)

    def head_forward(self, cls: mx.array) -> mx.array:
        """LN -> Linear on [B, D] CLS token."""
        assert cls.ndim == 2 and cls.shape[1] == self.cfg.embed_dim
        return self.head(self.norm(cls))

    @classmethod
    def from_pretrained_with_probe(
        cls,
        *,
        pretrained_weights: Path,
        pretrained_config: Path,
        probe_weights: Path,
        canvas_grid: int = 32,
    ) -> "CanViTForImageClassification":
        """Load pretrained backbone, fuse proj -> destandardize -> probe into LN -> Linear.

        Args:
            pretrained_weights: Path to pretrained .safetensors (MLX format)
            pretrained_config: Path to pretrained config.json
            probe_weights: Path to probe .safetensors (with 'weight' and 'bias' keys)
            canvas_grid: Canvas grid size used during pretraining standardizer calibration
        """
        log.info("Loading pretrained model from %s", pretrained_weights)
        pretrained = load_from_local(pretrained_weights, pretrained_config)
        cfg = pretrained.cfg
        assert cfg.teacher_dim is not None

        log.info("Loading probe from %s", probe_weights)
        probe_raw = _load_weights(probe_weights)
        probe_sd = {k: v for k, v in probe_raw}

        eps = 1e-6
        W_fused, b_fused = fuse_probe(
            W_proj=pretrained.scene_cls_proj.weight,
            b_proj=pretrained.scene_cls_proj.bias,
            mu=pretrained.cls_std_mean.squeeze(0),
            sigma=(pretrained.cls_std_var.squeeze(0) + eps).sqrt(),
            W_probe=probe_sd["weight"],
            b_probe=probe_sd["bias"],
        )
        mx.eval(W_fused, b_fused)

        n_classes = W_fused.shape[0]
        model = cls(cfg, n_classes)

        # Copy backbone weights (filter out pretraining-specific params)
        backbone_weights = [
            (k, v) for k, v in _load_weights(pretrained_weights)
            if not any(k.startswith(pfx) for pfx in _PRETRAINING_PREFIXES)
        ]
        model.canvit.load_weights(backbone_weights)

        # Set fused head + LN weights
        model.head.weight = W_fused
        model.head.bias = b_fused
        model.norm.weight = pretrained.scene_cls_ln.weight
        model.norm.bias = pretrained.scene_cls_ln.bias

        D = cfg.embed_dim
        log.info("Fused classifier: LN(%d) -> Linear(%d, %d), pretraining heads discarded", D, D, n_classes)
        return model

    @classmethod
    def from_finetuned_local(
        cls,
        *,
        weights_path: Path,
        config_path: Path,
    ) -> "CanViTForImageClassification":
        """Load a finetuned classifier from local MLX safetensors + config.json.

        Config must have 'n_classes' and 'model_config' keys.
        """
        import json
        meta = json.loads(config_path.read_text())
        cfg = CanViTConfig(**meta["model_config"])
        n_classes = meta["n_classes"]

        model = cls(cfg, n_classes)
        weights = _load_weights(weights_path)
        model.load_weights(weights)
        log.info("Loaded finetuned classifier: %d classes from %s", n_classes, weights_path)
        return model
