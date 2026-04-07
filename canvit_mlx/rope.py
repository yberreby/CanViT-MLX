"""2D Rotary Position Embeddings."""

__all__ = ["make_rope_periods", "compute_rope", "apply_rope_with_prefix"]

import math

import mlx.core as mx


def make_rope_periods(head_dim: int, base: float = 100.0) -> mx.array:
    assert head_dim % 4 == 0, f"2D RoPE requires head_dim divisible by 4, got {head_dim}"
    n_freqs = head_dim // 4
    return base ** (mx.arange(n_freqs) / n_freqs)


def compute_rope(positions: mx.array, periods: mx.array) -> tuple[mx.array, mx.array]:
    """positions: [B, N, 2], periods: [n_freqs] -> sin, cos each [B, 1, N, head_dim].

    Always computes and returns float32 — caller must NOT downcast.
    """
    assert positions.ndim == 3 and positions.shape[2] == 2, f"expected [B, N, 2], got {positions.shape}"
    angles = 2 * math.pi * mx.expand_dims(positions.astype(mx.float32), -1) / periods.astype(mx.float32)
    B, N = angles.shape[:2]
    angles = mx.concatenate([angles.reshape(B, N, -1)] * 2, axis=-1)
    return mx.expand_dims(mx.sin(angles), 1), mx.expand_dims(mx.cos(angles), 1)


def apply_rope_with_prefix(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Apply 2D RoPE to spatial tokens, leaving prefix tokens unchanged.

    Rotation is computed in float32 for precision, then cast back to x's dtype.
    sin/cos MUST be float32 (from compute_rope).
    """
    assert sin.dtype == mx.float32, f"RoPE sin must be float32, got {sin.dtype}"
    out_dtype = x.dtype
    n_prefix = x.shape[2] - sin.shape[2]
    half = x.shape[3] // 2
    prefix = x[:, :, :n_prefix] if n_prefix > 0 else None
    x_s = x[:, :, n_prefix:].astype(mx.float32)
    x1, x2 = x_s[..., :half], x_s[..., half:]
    spatial = mx.concatenate([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x2 * cos[..., half:] + x1 * sin[..., half:],
    ], axis=-1).astype(out_dtype)
    return mx.concatenate([prefix, spatial], axis=2) if prefix is not None else spatial
