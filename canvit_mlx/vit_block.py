__all__ = ["ViTBlock"]

import mlx.core as mx
import mlx.nn as nn

from .layer_scale import LayerScale
from .rope import apply_rope_with_prefix


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q = apply_rope_with_prefix(q, sin, cos)
        k = apply_rope_with_prefix(k, sin, cos)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class ViTBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), sin, cos))
        return x + self.ls2(self.mlp(self.norm2(x)))
