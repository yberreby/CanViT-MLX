"""Image-to-patch-tokens via strided convolution."""

__all__ = ["PatchEmbed"]

import mlx.core as mx
import mlx.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, int, int]:
        x = self.proj(x)
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D), H, W
