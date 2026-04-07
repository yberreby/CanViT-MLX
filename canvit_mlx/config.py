"""Model hyperparameters and architecture configuration."""

__all__ = ["CanViTConfig"]

from dataclasses import dataclass
from typing import Literal


@dataclass
class CanViTConfig:
    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size: int
    ffn_ratio: float
    n_backbone_registers: int
    rw_stride: int
    n_canvas_registers: int
    canvas_num_heads: int
    canvas_head_dim: int
    enable_vpe: bool
    enable_reads: bool = True
    canvas_update_mode: Literal["additive", "convex"] = "additive"
    gate_bias_init: float | None = None
    # Pretraining-specific (required by CanViTForPretraining, ignored by base CanViT)
    teacher_dim: int | None = None
    std_grid_size: int | None = None

    def __post_init__(self):
        assert (self.canvas_update_mode == "convex") == (self.gate_bias_init is not None), \
            f"convex requires gate_bias_init, got mode={self.canvas_update_mode}, gate_bias_init={self.gate_bias_init}"

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim

    @property
    def head_dim(self) -> int:
        assert self.embed_dim % self.num_heads == 0
        return self.embed_dim // self.num_heads
