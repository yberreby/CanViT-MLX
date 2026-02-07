"""2D Rotary Position Embeddings.

Two equivalent styles (proven in LOG.md):
- DINOv3 rotate-half: backbone self-attention
- Explicit sin/cos split: canvas cross-attention (saves one allocation)
"""

import math

import mlx.core as mx


def make_rope_periods(head_dim: int, base: float = 100.0) -> mx.array:
    n_freqs = head_dim // 4
    return base ** (mx.arange(n_freqs) / n_freqs)


def compute_rope(positions: mx.array, periods: mx.array) -> tuple[mx.array, mx.array]:
    """positions: [B, N, 2], periods: [n_freqs] → sin, cos each [B, 1, N, head_dim]."""
    angles = 2 * math.pi * mx.expand_dims(positions, -1) / periods
    B, N = angles.shape[:2]
    angles = mx.concatenate([angles.reshape(B, N, -1)] * 2, axis=-1)
    return mx.expand_dims(mx.sin(angles), 1), mx.expand_dims(mx.cos(angles), 1)


def apply_with_prefix(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Explicit sin/cos split (canvas attention). x: [B, H, N, hd], sin/cos: [B, 1, N_spatial, hd]."""
    n_prefix = x.shape[2] - sin.shape[2]
    half = x.shape[3] // 2
    prefix = x[:, :, :n_prefix] if n_prefix > 0 else None
    x_s = x[:, :, n_prefix:]
    x1, x2 = x_s[..., :half], x_s[..., half:]
    spatial = mx.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1)
    return mx.concatenate([prefix, spatial], axis=2) if prefix is not None else spatial


def apply_dinov3_with_prefix(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Rotate-half (backbone self-attention). Same args as apply_with_prefix."""
    n_prefix = x.shape[2] - sin.shape[2]
    spatial = x[:, :, n_prefix:]
    x1, x2 = mx.split(spatial, 2, axis=-1)
    spatial = spatial * cos + mx.concatenate([-x2, x1], axis=-1) * sin
    return mx.concatenate([x[:, :, :n_prefix], spatial], axis=2) if n_prefix > 0 else spatial
