"""Image preprocessing: resize shortest edge, center crop, ImageNet normalize."""

import mlx.core as mx
import mlx.data as dx
__all__ = ["load_and_preprocess"]

IMAGENET_MEAN = mx.array([0.485, 0.456, 0.406])
IMAGENET_STD = mx.array([0.229, 0.224, 0.225])


def _resize_crop_normalize(buf: dx.Buffer, target_size: int) -> mx.array:
    dset = (
        buf
        .image_resize_smallest_side("image", target_size)
        .image_center_crop("image", target_size, target_size)
    )
    x = mx.array(list(dset)[0]["image"])
    x = x.astype(mx.float32) * (1 / 255.0)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD)[None]


def load_and_preprocess(path: str, target_size: int) -> mx.array:
    """Load image from file, resize shortest edge, center crop, ImageNet normalize.

    Returns [1, target_size, target_size, 3] float32.
    """
    buf = (
        dx.buffer_from_vector([{"file": path.encode()}])
        .load_image("file", output_key="image")
    )
    return _resize_crop_normalize(buf, target_size)
