"""Model hyperparameters and architecture configuration."""

__all__ = ["CanViTConfig"]

from dataclasses import dataclass


@dataclass
class CanViTConfig:
    embed_dim: int
    num_heads: int
    n_blocks: int
    patch_size: int
    ffn_ratio: float
    n_register_tokens: int
    rw_stride: int
    n_canvas_registers: int
    canvas_num_heads: int
    canvas_head_dim: int
    enable_vpe: bool
    teacher_dim: int
    std_grid_size: int

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim

    @property
    def head_dim(self) -> int:
        assert self.embed_dim % self.num_heads == 0
        return self.embed_dim // self.num_heads
