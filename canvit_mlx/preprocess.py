"""Image preprocessing: resize shortest edge, center crop, ImageNet normalize."""

import mlx.core as mx
import numpy as np
from PIL import Image

__all__ = ["load_and_preprocess"]

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


def load_and_preprocess(path: str, target_size: int) -> mx.array:
    """Load image, resize shortest edge, center crop, ImageNet normalize.

    Returns [1, target_size, target_size, 3] float32.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w < h:
        new_w, new_h = target_size, int(h * target_size / w)
    else:
        new_h, new_w = target_size, int(w * target_size / h)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    # Center crop
    x0 = (new_w - target_size) // 2
    y0 = (new_h - target_size) // 2
    img = img.crop((x0, y0, x0 + target_size, y0 + target_size))
    x = mx.array(np.asarray(img))
    x = x.astype(mx.float32) * (1 / 255.0)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD)[None]
