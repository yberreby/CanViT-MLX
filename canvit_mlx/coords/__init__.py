"""Coordinate systems and viewpoint-based sampling.

All coordinates in [-1, 1]², (y, x) order. (0, 0) = center.
"""

from dataclasses import dataclass

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


@dataclass
class Viewpoint:
    centers: mx.array  # [B, 2]
    scales: mx.array   # [B]

    @staticmethod
    def full_scene(batch_size: int) -> "Viewpoint":
        return Viewpoint(centers=mx.zeros((batch_size, 2)), scales=mx.ones((batch_size,)))


def sample_at_viewpoint(image: mx.array, viewpoint: Viewpoint, glimpse_size_px: int) -> mx.array:
    """Bilinear sampling. image: [B, H, W, C] → [B, gH, gW, C]. Matches F.grid_sample(align_corners=False)."""
    B, H, W, C = image.shape
    g = glimpse_size_px

    grid = (viewpoint.centers.reshape(B, 1, 1, 2) +
            viewpoint.scales.reshape(B, 1, 1, 1) * grid_coords(g, g).reshape(1, g, g, 2))

    gy = (grid[..., 0] + 1) / 2 * H - 0.5
    gx = (grid[..., 1] + 1) / 2 * W - 0.5

    y0 = mx.floor(gy).astype(mx.int32)
    x0 = mx.floor(gx).astype(mx.int32)
    wy = mx.expand_dims(gy - y0.astype(mx.float32), -1)
    wx = mx.expand_dims(gx - x0.astype(mx.float32), -1)

    y0c = mx.clip(y0, 0, H - 1)
    y1c = mx.clip(y0 + 1, 0, H - 1)
    x0c = mx.clip(x0, 0, W - 1)
    x1c = mx.clip(x0 + 1, 0, W - 1)

    flat = image.reshape(B, H * W, C)

    def gather(yy: mx.array, xx: mx.array) -> mx.array:
        idx = (yy * W + xx).reshape(B, g * g)
        return mx.take_along_axis(flat, mx.broadcast_to(mx.expand_dims(idx, -1), (B, g * g, C)), axis=1).reshape(B, g, g, C)

    return (gather(y0c, x0c) * (1 - wy) * (1 - wx) +
            gather(y0c, x1c) * (1 - wy) * wx +
            gather(y1c, x0c) * wy * (1 - wx) +
            gather(y1c, x1c) * wy * wx)
