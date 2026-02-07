#!/usr/bin/env python3
"""CanViT MLX end-to-end verification: real image, multi-step, side-by-side plots.

What this tests:
  - Real image (Cat03.jpg) preprocessed identically for both frameworks
  - 3 forward steps with different viewpoints, feeding state back each time
  - Every recurrent output (canvas, recurrent_cls) checked at every step
  - Side-by-side PCA visualization of canvas features (PyTorch vs MLX)

What this does NOT test:
  - Multiple images (single image only)
  - Batch size > 1
  - grid_sample on real images (glimpses are pre-sampled by each framework)

Usage:
    uv run python demo.py
    uv run python demo.py --image path/to/image.jpg
    uv run python demo.py --no-compare-pytorch  # MLX only, no verification
"""

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
import tyro
from PIL import Image
from sklearn.decomposition import PCA

from canvit_mlx import (
    RecurrentState as MlxState,
    Viewpoint as MlxViewpoint,
    load_canvit,
    sample_at_viewpoint as mlx_sample,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (separated from logic)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Viewpoint trajectory: 3 steps with different positions/scales.
# (center_y, center_x, scale) — full scene, then two partial views.
VIEWPOINT_TRAJECTORY = [
    (0.0,  0.0,  1.0),   # step 0: full scene
    (0.3, -0.2,  0.5),   # step 1: zoom into upper-left region
    (-0.3, 0.3,  0.6),   # step 2: different region
]

# Acceptance criteria per output tensor.
# atol: max acceptable absolute error
# rtol: max acceptable relative error (max_abs / max_|ref|)
# Logic: PASS if max_abs < atol OR max_rel < rtol (either suffices).
#
# Justification: f32 SDPA accumulation over ~1000 tokens with values in
# the thousands produces absolute errors ~2-4 at extreme values. Error
# grows across steps since inputs diverge slightly. See LOG.md.
TOLERANCES: dict[str, tuple[float, float]] = {
    "canvas":        (5.0,  2e-3),
    "recurrent_cls": (1.0,  1e-3),
    "ephemeral_cls": (1.0,  1e-3),
    "local_patches": (1.0,  1e-3),
    "scene_pred":    (0.1,  2e-3),
    "cls_pred":      (0.01, 1e-3),
}

# Step 2+ gets looser tolerances (error accumulates through recurrence)
TOLERANCES_STEP2: dict[str, tuple[float, float]] = {
    "canvas":        (10.0, 5e-3),
    "recurrent_cls": (2.0,  2e-3),
    "ephemeral_cls": (2.0,  2e-3),
    "local_patches": (2.0,  2e-3),
    "scene_pred":    (0.5,  5e-3),
    "cls_pred":      (0.05, 2e-3),
}

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
PATCH_SIZE = 16


@dataclass
class Config:
    image: Path = Path("test_data/Cat03.jpg")
    weights: Path = Path("weights.safetensors")
    canvas_grid: int = 32
    glimpse_px: int = 128
    compare_pytorch: bool = True
    plot: Path = Path("outputs/comparison.png")


# ---------------------------------------------------------------------------
# Shared image loading (guarantees identical pixels for both frameworks)
# ---------------------------------------------------------------------------

def load_and_preprocess(path: Path, size: int) -> tuple[mx.array, torch.Tensor]:
    """PIL load → resize → center crop → ImageNet normalize.

    Returns NHWC for MLX, NCHW for PyTorch, from the same numpy array.
    """
    img = Image.open(path.expanduser()).convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
    w, h = img.size
    img = img.crop(((w - size) // 2, (h - size) // 2,
                     (w + size) // 2, (h + size) // 2))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return mx.array(arr[np.newaxis]), torch.from_numpy(arr.transpose(2, 0, 1)[np.newaxis])


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def check(name: str, ref: np.ndarray, got: np.ndarray, tols: dict[str, tuple[float, float]]) -> bool:
    """Compare arrays, log result with all context. Returns True if PASS."""
    assert ref.shape == got.shape, f"{name}: shape mismatch {ref.shape} vs {got.shape}"
    atol, rtol = tols[name]
    diff = np.abs(ref - got)
    ref_max = float(np.abs(ref).max()) + 1e-8
    max_abs = float(diff.max())
    max_rel = max_abs / ref_max
    passed = max_abs < atol or max_rel < rtol
    status = "PASS" if passed else "FAIL"
    log.info("    [%s] %-16s max_abs=%.4f (atol=%.1e) max_rel=%.2e (rtol=%.1e) range=[%.1f, %.1f]",
             status, name, max_abs, atol, max_rel, rtol, float(ref.min()), float(ref.max()))
    return passed


def canvas_to_pca_rgb(canvas: np.ndarray, n_registers: int, pca: PCA | None = None) -> tuple[np.ndarray, PCA]:
    """Extract spatial tokens from canvas, PCA to RGB.

    canvas: [1, n_regs + G², D]
    Returns: ([G, G, 3] uint8, fitted PCA)
    """
    spatial = canvas[0, n_registers:]  # [G², D]
    g = int(math.sqrt(spatial.shape[0]))
    assert g * g == spatial.shape[0]
    if pca is None:
        pca = PCA(n_components=3, whiten=True)
        pca.fit(spatial)
    proj = pca.transform(spatial)[:, :3]
    rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(proj, -10, 10)))
    return (rgb.reshape(g, g, 3) * 255).astype(np.uint8), pca


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: Config) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    img_size = cfg.canvas_grid * PATCH_SIZE
    n_steps = len(VIEWPOINT_TRAJECTORY)

    log.info("=== CanViT MLX End-to-End Verification ===")
    log.info("Image: %s (%dpx), canvas_grid=%d, glimpse=%dpx, steps=%d",
             cfg.image, img_size, cfg.canvas_grid, cfg.glimpse_px, n_steps)
    log.info("Viewpoints: %s", VIEWPOINT_TRAJECTORY)

    # Load models
    mlx_model = load_canvit(str(cfg.weights))
    n_canvas_regs = mlx_model.cfg.n_canvas_registers

    image_mlx, image_pt = load_and_preprocess(cfg.image, img_size)
    log.info("Preprocessed: MLX %s (NHWC), PyTorch %s (NCHW)", image_mlx.shape, tuple(image_pt.shape))

    pt_model = None
    if cfg.compare_pytorch:
        from canvit import CanViTForPretrainingHFHub
        pt_model = CanViTForPretrainingHFHub.from_pretrained(HF_REPO).eval()

    # Init states
    state_mlx = mlx_model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid)
    state_pt = pt_model.init_state(batch_size=1, canvas_grid_size=cfg.canvas_grid) if pt_model else None

    # Storage for plots
    canvases_mlx: list[np.ndarray] = []
    canvases_pt: list[np.ndarray] = []
    all_pass = True

    for step, (cy, cx, s) in enumerate(VIEWPOINT_TRAJECTORY):
        tols = TOLERANCES if step == 0 else TOLERANCES_STEP2
        log.info("--- Step %d: center=(%.1f, %.1f), scale=%.1f (tols=%s) ---",
                 step, cy, cx, s, "base" if step == 0 else "accumulated")

        # MLX
        vp_mlx = MlxViewpoint(centers=mx.array([[cy, cx]]), scales=mx.array([s]))
        glimpse_mlx = mlx_sample(image_mlx, vp_mlx, cfg.glimpse_px)

        t0 = time.perf_counter()
        out_mlx = mlx_model(glimpse_mlx, state_mlx, vp_mlx)
        mx.eval(out_mlx.state.canvas, out_mlx.state.recurrent_cls,
                out_mlx.ephemeral_cls, out_mlx.local_patches)
        dt_mlx = time.perf_counter() - t0

        scene_mlx = mlx_model.predict_teacher_scene(out_mlx.state.canvas)
        cls_mlx = mlx_model.predict_scene_teacher_cls(out_mlx.state.recurrent_cls)
        mx.eval(scene_mlx, cls_mlx)

        log.info("  MLX: %.1fms, canvas range [%.1f, %.1f]",
                 dt_mlx * 1000,
                 float(out_mlx.state.canvas.min()), float(out_mlx.state.canvas.max()))

        canvases_mlx.append(np.array(out_mlx.state.canvas))

        if pt_model is not None and state_pt is not None:
            from canvit.viewpoint import Viewpoint as PtViewpoint, sample_at_viewpoint as pt_sample

            vp_pt = PtViewpoint(
                centers=torch.tensor([[cy, cx]], dtype=torch.float32),
                scales=torch.tensor([s], dtype=torch.float32),
            )
            glimpse_pt = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=cfg.glimpse_px)

            t0 = time.perf_counter()
            with torch.inference_mode():
                out_pt = pt_model(glimpse=glimpse_pt, state=state_pt, viewpoint=vp_pt)
                scene_pt = pt_model.predict_teacher_scene(out_pt.state.canvas)
                cls_pt = pt_model.predict_scene_teacher_cls(out_pt.state.recurrent_cls)
            dt_pt = time.perf_counter() - t0

            log.info("  PyTorch: %.1fms, canvas range [%.1f, %.1f]",
                     dt_pt * 1000,
                     float(out_pt.state.canvas.min()), float(out_pt.state.canvas.max()))

            canvases_pt.append(out_pt.state.canvas.numpy())

            pairs: list[tuple[str, np.ndarray, np.ndarray]] = [
                ("canvas",        out_pt.state.canvas.numpy(),        np.array(out_mlx.state.canvas)),
                ("recurrent_cls", out_pt.state.recurrent_cls.numpy(), np.array(out_mlx.state.recurrent_cls)),
                ("ephemeral_cls", out_pt.ephemeral_cls.numpy(),       np.array(out_mlx.ephemeral_cls)),
                ("local_patches", out_pt.local_patches.numpy(),       np.array(out_mlx.local_patches)),
                ("scene_pred",    scene_pt.numpy(),                   np.array(scene_mlx)),
                ("cls_pred",      cls_pt.numpy(),                     np.array(cls_mlx)),
            ]
            for name, ref, got in pairs:
                if not check(name, ref, got, tols):
                    all_pass = False

            # Feed state forward for next step
            state_pt = out_pt.state

        # Feed MLX state forward
        state_mlx = out_mlx.state

    # --- Plot ---
    if canvases_pt:
        log.info("Generating side-by-side PCA plots...")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, n_steps + 1, figsize=(4 * (n_steps + 1), 8))

        # Input image (denormalized)
        img_display = np.array(image_mlx[0])  # NHWC, normalized
        img_display = img_display * IMAGENET_STD + IMAGENET_MEAN
        img_display = np.clip(img_display, 0, 1)
        axes[0, 0].imshow(img_display)
        axes[0, 0].set_title("Input image")
        axes[0, 0].axis("off")
        axes[1, 0].imshow(img_display)
        axes[1, 0].set_title("Input image")
        axes[1, 0].axis("off")

        for step in range(n_steps):
            cy, cx, s = VIEWPOINT_TRAJECTORY[step]
            # Fit PCA on PyTorch canvas, apply same projection to both
            pt_rgb, pca = canvas_to_pca_rgb(canvases_pt[step], n_canvas_regs)
            mlx_rgb, _ = canvas_to_pca_rgb(canvases_mlx[step], n_canvas_regs, pca=pca)

            label = f"step {step}: c=({cy},{cx}) s={s}"
            axes[0, step + 1].imshow(pt_rgb)
            axes[0, step + 1].set_title(f"PyTorch\n{label}")
            axes[0, step + 1].axis("off")
            axes[1, step + 1].imshow(mlx_rgb)
            axes[1, step + 1].set_title(f"MLX\n{label}")
            axes[1, step + 1].axis("off")

        fig.suptitle("Canvas PCA: PyTorch (top) vs MLX (bottom)\nSame PCA basis per step", fontsize=12)
        plt.tight_layout()
        cfg.plot.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(cfg.plot, dpi=150)
        log.info("Plot saved: %s", cfg.plot)

    # --- Verdict ---
    if pt_model is not None:
        if all_pass:
            log.info("=== ALL CHECKS PASSED (%d steps × %d outputs) ===", n_steps, len(TOLERANCES))
        else:
            log.error("=== SOME CHECKS FAILED ===")
            raise SystemExit(1)
    else:
        log.info("MLX ran successfully (no PyTorch comparison)")


if __name__ == "__main__":
    main(tyro.cli(Config))
