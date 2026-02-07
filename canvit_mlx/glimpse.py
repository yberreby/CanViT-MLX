__all__ = ["extract_glimpse_at_viewpoint"]

import mlx.core as mx

from .grid import grid_coords
from .viewpoint import Viewpoint


def extract_glimpse_at_viewpoint(image: mx.array, viewpoint: Viewpoint, glimpse_size_px: int) -> mx.array:
    """Bilinear sampling. image: [B, H, W, C] -> [B, gH, gW, C]. Matches F.grid_sample(align_corners=False)."""
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
