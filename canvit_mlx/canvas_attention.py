"""Canvas cross-attention: local reads from / writes to the persistent canvas."""

__all__ = ["CanvasReadAttention", "CanvasWriteAttention"]

import mlx.core as mx
import mlx.nn as nn

from .rope import apply_rope_with_prefix


def _to_mh(x: mx.array, h: int) -> mx.array:
    B, N, D = x.shape
    assert D % h == 0
    return x.reshape(B, N, h, D // h).transpose(0, 2, 1, 3)


def _from_mh(x: mx.array) -> mx.array:
    B, H, N, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * hd)


class CanvasReadAttention(nn.Module):
    """Local queries canvas. Dense projections on local side only."""
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        assert canvas_dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(local_dim)
        self.pre_kv_ln = nn.LayerNorm(canvas_dim)
        self.q_transform = nn.Linear(local_dim, canvas_dim)
        self.out_transform = nn.Linear(canvas_dim, local_dim)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array, kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        q = apply_rope_with_prefix(_to_mh(self.q_transform(self.pre_q_ln(query)), self.num_heads), q_sin, q_cos)
        kv_n = self.pre_kv_ln(kv)
        k = apply_rope_with_prefix(_to_mh(kv_n, self.num_heads), kv_sin, kv_cos)
        v = _to_mh(kv_n, self.num_heads)
        return self.out_transform(_from_mh(mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)))


class CanvasWriteAttention(nn.Module):
    """Canvas queries local. Dense projections on local side only."""
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        assert canvas_dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(canvas_dim)
        self.pre_kv_ln = nn.LayerNorm(local_dim)
        self.k_transform = nn.Linear(local_dim, canvas_dim)
        self.v_transform = nn.Linear(local_dim, canvas_dim)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array, kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        q = apply_rope_with_prefix(_to_mh(self.pre_q_ln(query), self.num_heads), q_sin, q_cos)
        kv_n = self.pre_kv_ln(kv)
        k = apply_rope_with_prefix(_to_mh(self.k_transform(kv_n), self.num_heads), kv_sin, kv_cos)
        v = _to_mh(self.v_transform(kv_n), self.num_heads)
        return _from_mh(mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale))
