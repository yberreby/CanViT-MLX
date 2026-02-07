__all__ = ["Viewpoint"]

from dataclasses import dataclass

import mlx.core as mx


@dataclass
class Viewpoint:
    centers: mx.array  # [B, 2]
    scales: mx.array   # [B]

    @staticmethod
    def full_scene(batch_size: int) -> "Viewpoint":
        return Viewpoint(centers=mx.zeros((batch_size, 2)), scales=mx.ones((batch_size,)))
