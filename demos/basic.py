#!/usr/bin/env python3
"""CanViT-MLX demo: load model from HF Hub, classify image, visualize canvas PCA.

Usage:
    uv run python demos/basic.py
    uv run python demos/basic.py --image test_data/Cat03.jpg --canvas-grid 64
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn
import numpy as np
import tyro
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from safetensors import safe_open
from sklearn.decomposition import PCA

from canvit_mlx import Viewpoint, load_from_hf_hub, load_and_preprocess

HF_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx"
PROBE_REPO = "yberreby/dinov3-vitb16-lvd1689m-in1k-512x512-linear-clf-probe"


@dataclass
class Config:
    hf_repo: str = HF_REPO
    image: Path = Path("test_data/Cat03.jpg")
    canvas_grid: int = 32
    glimpse_px: int = 128
    output: Path = Path("outputs/demo_pca.png")


def load_probe(repo: str) -> tuple[NDArray, NDArray]:
    """Load linear probe (weight, bias) from HF Hub safetensors."""
    path = hf_hub_download(repo, "model.safetensors")
    with safe_open(path, framework="numpy") as f:
        return f.get_tensor("weight"), f.get_tensor("bias")


def spatial_to_rgb(spatial: NDArray) -> NDArray[np.uint8]:
    """PCA-project spatial tokens to RGB uint8 image."""
    n = spatial.shape[0]
    grid = int(math.sqrt(n))
    assert grid * grid == n
    pca = PCA(n_components=3, whiten=True)
    proj = pca.fit_transform(spatial)
    rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(proj, -10, 10)))
    return (rgb.reshape(grid, grid, 3) * 255).astype(np.uint8)


def main(cfg: Config) -> None:
    # Load model
    print(f"Loading {cfg.hf_repo}...")
    model = load_from_hf_hub(cfg.hf_repo)
    mx.eval(model.parameters())
    n_params = sum(v.size for _, v in mlx.nn.utils.tree_flatten(model.parameters()))
    print(f"  {n_params / 1e6:.1f}M params")

    # Load probe
    print(f"Loading probe from {PROBE_REPO}...")
    probe_w, probe_b = load_probe(PROBE_REPO)

    # Load + preprocess image
    print(f"Loading {cfg.image}...")
    glimpse = load_and_preprocess(str(cfg.image), cfg.glimpse_px)

    # Forward pass
    print("Running inference...")
    state = model.init_state(1, cfg.canvas_grid)
    vp = Viewpoint.full_scene(1)
    out = model(glimpse, state, vp)
    mx.eval(out.state.canvas, out.state.recurrent_cls)

    n_regs = model.cfg.n_canvas_registers
    n_spatial = cfg.canvas_grid * cfg.canvas_grid
    print(f"  canvas: {out.state.canvas.shape} ({n_regs} regs + {n_spatial} spatial)")

    # --- Classification ---
    cls_pred_std = model.predict_scene_teacher_cls(out.state.recurrent_cls)
    cls_raw = np.array(cls_pred_std * mx.sqrt(model.cls_std_var) + model.cls_std_mean)
    logits = cls_raw @ probe_w.T + probe_b
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    try:
        import timm
        ini = timm.data.ImageNetInfo()
        get_label = lambda idx: ini.index_to_description(idx)
    except ImportError:
        get_label = lambda idx: f"class {idx}"

    print("\nClassification:")
    top5 = np.argsort(logits[0])[::-1][:5]
    for i, idx in enumerate(top5, 1):
        print(f"  {i}. {get_label(idx):40s} {probs[0, idx] * 100:5.2f}%")

    # --- PCA visualization ---
    print("\nGenerating PCA visualization...")
    canvas_spatial = out.state.canvas[:, n_regs:]  # [1, N, canvas_dim]
    scene_pred_std = model.predict_teacher_scene(out.state.canvas)
    scene_destd = scene_pred_std * mx.sqrt(model.scene_std_var) + model.scene_std_mean

    panels = [
        ("Canvas (raw)",         spatial_to_rgb(np.array(canvas_spatial[0]))),
        ("Canvas (LN)",          spatial_to_rgb(np.array(mlx.nn.LayerNorm(model.cfg.canvas_dim)(canvas_spatial)[0]))),
        ("Scene pred (std)",     spatial_to_rgb(np.array(scene_pred_std[0]))),
        ("Scene pred (destd)",   spatial_to_rgb(np.array(scene_destd[0]))),
    ]

    from PIL import Image
    img_display = np.array(Image.open(cfg.image).convert("RGB"))

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 4, hspace=0.15, wspace=0.1)

    # Input image: left half
    ax_img = fig.add_subplot(gs[:, :2])
    ax_img.imshow(img_display)
    ax_img.set_title("Input")
    ax_img.axis("off")

    # 2x2 PCA grid: right half
    for i, (title, img) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, 2 + i % 2])
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(cfg.output, dpi=150, bbox_inches="tight")
    print(f"  saved: {cfg.output}")


if __name__ == "__main__":
    main(tyro.cli(Config))
