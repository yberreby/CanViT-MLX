__all__ = ["grid_coords", "canvas_coords_for_glimpse"]

import mlx.core as mx


def grid_coords(H: int, W: int) -> mx.array:
    """Normalized [-1, 1] grid. Returns [H, W, 2] with (y, x)."""
    y = (mx.arange(H) + 0.5) / H * 2 - 1
    x = (mx.arange(W) + 0.5) / W * 2 - 1
    yy, xx = mx.meshgrid(y, x, indexing="ij")
    return mx.stack([yy, xx], axis=-1)


def canvas_coords_for_glimpse(center: mx.array, scale: mx.array, H: int, W: int) -> mx.array:
    """Canvas-space coordinates for glimpse patches. Returns [B, H*W, 2]."""
    B = center.shape[0]
    retino = grid_coords(H, W).reshape(1, H * W, 2)
    return center.reshape(B, 1, 2) + scale.reshape(B, 1, 1) * retino
