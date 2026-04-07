"""Viewpoint Position Encoding via random Fourier features."""

__all__ = ["VPEEncoder"]

import mlx.core as mx
import mlx.nn as nn


class VPEEncoder(nn.Module):
    def __init__(self, rff_dim: int):
        super().__init__()
        self.B_mat = mx.zeros((rff_dim // 2, 3))
        self.norm = nn.LayerNorm(rff_dim)

    def __call__(self, y: mx.array, x: mx.array, s: mx.array) -> mx.array:
        z = mx.stack([y / s, x / s, mx.log(s)], axis=-1)
        proj = z @ self.B_mat.T
        return self.norm(mx.concatenate([mx.cos(proj), mx.sin(proj)], axis=-1))
