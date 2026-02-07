"""End-to-end test constants and shared data loading."""

from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

from conftest import CANVAS_GRID

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
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
    """PIL → resize → center crop → ImageNet normalize. Returns (NHWC, NCHW)."""
    img = Image.open(IMAGE_PATH).convert("RGB")
    w, h = img.size
    scale = IMG_SIZE / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    img = img.crop(((w - IMG_SIZE) // 2, (h - IMG_SIZE) // 2,
                     (w + IMG_SIZE) // 2, (h + IMG_SIZE) // 2))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return mx.array(arr[np.newaxis]), torch.from_numpy(arr.transpose(2, 0, 1)[np.newaxis])
