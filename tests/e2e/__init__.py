"""End-to-end test constants and shared data loading."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from conftest import CANVAS_GRID

from canvit_mlx.preprocess import load_and_preprocess

IMAGE_PATH = Path("test_data/Cat03.jpg")
PATCH_SIZE = 16
IMG_SIZE = CANVAS_GRID * PATCH_SIZE  # 512

# (center_y, center_x, scale)
TRAJECTORY = [
    (0.0, 0.0, 1.0),    # full scene
    (0.3, -0.2, 0.5),   # zoom upper-left
    (-0.3, 0.3, 0.6),   # pan lower-right
]

OUTPUTS = ("canvas", "recurrent_cls", "ephemeral_cls", "local_patches")

# f32 SDPA accumulation error grows with sequence length and through recurrence.
RTOL = 5e-3


def load_image() -> tuple[mx.array, torch.Tensor]:
    """Preprocess via canvit_mlx.preprocess, derive PT tensor from same data."""
    image_mlx = load_and_preprocess(str(IMAGE_PATH), target_size=IMG_SIZE)
    # NHWC → NCHW for PyTorch, share the same pixel data
    image_pt = torch.from_numpy(np.array(image_mlx).transpose(0, 3, 1, 2))
    return image_mlx, image_pt
