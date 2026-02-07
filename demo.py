#!/usr/bin/env python3
"""CanViT MLX end-to-end verification: real image, full pipeline, explicit pass/fail.

Runs MLX inference on a real image and compares every output tensor against
PyTorch f32 CPU reference. Prints explicit PASS/FAIL for each check.

Usage:
    uv run python demo.py
    uv run python demo.py --image path/to/image.jpg
    uv run python demo.py --no-compare-pytorch  # MLX only, no verification
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import tyro
from PIL import Image

from canvit_mlx import (
    Viewpoint as MlxViewpoint,
    load_canvit,
    sample_at_viewpoint as mlx_sample,
)

log = logging.getLogger(__name__)

# ImageNet normalization constants (shared, single source of truth)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Acceptance criteria for f32 agreement.
# Justification: f32 SDPA accumulation over ~1000 tokens with values in the
# thousands produces max absolute errors ~2-4 at extreme values. Relative
# error stays below 1e-3. These thresholds are derived from empirical error
# analysis (see test_modules.py and LOG.md).
TOLERANCES: dict[str, tuple[float, float]] = {
    # name: (atol, rtol)
    "glimpse":       (1e-4,  1e-5),   # preprocessing only, should be near-exact
    "canvas":        (5.0,   2e-3),    # largest tensor, highest magnitude, most SDPA accumulation
    "recurrent_cls": (1.0,   1e-3),    # single token, high magnitude
    "ephemeral_cls": (1.0,   1e-3),    # single token
    "local_patches": (1.0,   1e-3),    # 64 tokens
    "scene_pred":    (0.1,   2e-3),    # after LayerNorm+Linear, smaller range
    "cls_pred":      (0.01,  1e-3),    # after LayerNorm+Linear, small range
}


@dataclass
class Config:
    image: Path = Path("test_data/Cat03.jpg")
    weights: Path = Path("weights.safetensors")
    canvas_grid: int = 32
    glimpse_px: int = 128
    compare_pytorch: bool = True


def load_and_preprocess(path: Path, size: int) -> tuple[mx.array, torch.Tensor]:
    """Load image, preprocess to both MLX (NHWC) and PyTorch (NCHW) tensors.

    Uses PIL for loading + crop so both frameworks get identical pixels.
    """
    img = Image.open(path.expanduser()).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    w, h = img.size
    img = img.crop(((w - size) // 2, (h - size) // 2,
                     (w + size) // 2, (h + size) // 2))

    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD  # [H, W, 3]

    mlx_img = mx.array(arr[np.newaxis])  # [1, H, W, 3]
    pt_img = torch.from_numpy(arr.transpose(2, 0, 1)[np.newaxis])  # [1, 3, H, W]
    return mlx_img, pt_img


def check(name: str, ref: np.ndarray, got: np.ndarray) -> bool:
    """Compare arrays against acceptance criteria. Returns True if PASS."""
    atol, rtol = TOLERANCES[name]
    diff = np.abs(ref - got)
    ref_scale = np.abs(ref).max() + 1e-8
    max_abs = float(diff.max())
    max_rel = max_abs / ref_scale
    mean_abs = float(diff.mean())
    passed = max_abs < atol or max_rel < rtol
    status = "PASS" if passed else "FAIL"
    log.info("  [%s] %-16s  max_abs=%.4f (atol=%.1e)  max_rel=%.2e (rtol=%.1e)  mean=%.4f  range=[%.1f, %.1f]",
             status, name, max_abs, atol, max_rel, rtol, mean_abs, ref.min(), ref.max())
    return passed


def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    patch_size = 16
    img_size = cfg.canvas_grid * patch_size

    log.info("=== CanViT MLX End-to-End Verification ===")
    log.info("Image: %s, canvas_grid=%d, glimpse=%dpx, img_size=%dpx",
             cfg.image, cfg.canvas_grid, cfg.glimpse_px, img_size)

    # --- Load models ---
    mlx_model = load_canvit(str(cfg.weights))

    # --- Load and preprocess image (shared pixels) ---
    image_mlx, image_pt = load_and_preprocess(cfg.image, img_size)
    log.info("Preprocessed: MLX %s, PyTorch %s", image_mlx.shape, tuple(image_pt.shape))

    # --- MLX forward ---
    vp_mlx = MlxViewpoint.full_scene(batch_size=1)
    state_mlx = mlx_model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    glimpse_mlx = mlx_sample(image_mlx, vp_mlx, cfg.glimpse_px)

    t0 = time.perf_counter()
    out_mlx = mlx_model(glimpse_mlx, state_mlx, vp_mlx)
    mx.eval(out_mlx.state.canvas, out_mlx.state.recurrent_cls,
            out_mlx.ephemeral_cls, out_mlx.local_patches)
    dt_mlx = time.perf_counter() - t0

    scene_mlx = mlx_model.predict_teacher_scene(out_mlx.state.canvas)
    cls_mlx = mlx_model.predict_scene_teacher_cls(out_mlx.state.recurrent_cls)
    mx.eval(scene_mlx, cls_mlx)

    log.info("MLX forward: %.1fms", dt_mlx * 1000)
    log.info("  canvas %s  range=[%.1f, %.1f]", out_mlx.state.canvas.shape,
             float(out_mlx.state.canvas.min()), float(out_mlx.state.canvas.max()))

    if not cfg.compare_pytorch:
        log.info("PyTorch comparison skipped. Cannot verify correctness.")
        return

    # --- PyTorch forward ---
    from canvit import CanViTForPretrainingHFHub
    from canvit.viewpoint import Viewpoint as PtViewpoint, sample_at_viewpoint as pt_sample

    pt_model = CanViTForPretrainingHFHub.from_pretrained(
        "canvit/canvit-vitb16-pretrain-512px-in21k"
    ).eval()

    vp_pt = PtViewpoint.full_scene(batch_size=1, device=torch.device("cpu"))
    state_pt = pt_model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    glimpse_pt = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=cfg.glimpse_px)

    t0 = time.perf_counter()
    with torch.inference_mode():
        out_pt = pt_model(glimpse=glimpse_pt, state=state_pt, viewpoint=vp_pt)
        scene_pt = pt_model.predict_teacher_scene(out_pt.state.canvas)
        cls_pt = pt_model.predict_scene_teacher_cls(out_pt.state.recurrent_cls)
    dt_pt = time.perf_counter() - t0

    log.info("PyTorch forward: %.1fms", dt_pt * 1000)

    # --- Systematic comparison ---
    log.info("Checking %d outputs against f32 CPU PyTorch:", len(TOLERANCES))

    pairs: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("glimpse",       glimpse_pt.numpy().transpose(0, 2, 3, 1), np.array(glimpse_mlx)),
        ("canvas",        out_pt.state.canvas.numpy(),               np.array(out_mlx.state.canvas)),
        ("recurrent_cls", out_pt.state.recurrent_cls.numpy(),        np.array(out_mlx.state.recurrent_cls)),
        ("ephemeral_cls", out_pt.ephemeral_cls.numpy(),              np.array(out_mlx.ephemeral_cls)),
        ("local_patches", out_pt.local_patches.numpy(),              np.array(out_mlx.local_patches)),
        ("scene_pred",    scene_pt.numpy(),                          np.array(scene_mlx)),
        ("cls_pred",      cls_pt.numpy(),                            np.array(cls_mlx)),
    ]

    all_pass = all(check(name, ref, got) for name, ref, got in pairs)

    log.info("Speed: MLX %.1fms, PyTorch CPU %.1fms", dt_mlx * 1000, dt_pt * 1000)

    if all_pass:
        log.info("=== ALL %d CHECKS PASSED ===", len(pairs))
    else:
        log.error("=== SOME CHECKS FAILED ===")
        raise SystemExit(1)


if __name__ == "__main__":
    main(tyro.cli(Config))
