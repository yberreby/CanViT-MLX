"""PCA comparison and difference heatmap visualization."""

import math
from pathlib import Path

import mlx.core as mx
import numpy as np

from canvit_mlx.preprocess import IMAGENET_MEAN, IMAGENET_STD

from . import TRAJECTORY

OUT_PATH = Path("outputs/comparison.png")


def save(pt_results: list[dict[str, np.ndarray]],
         mlx_results: list[dict[str, np.ndarray]],
         image_mlx: mx.array, n_regs: int) -> None:
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_steps = len(pt_results)
    # 3 rows: PyTorch PCA, MLX PCA, absolute difference heatmap
    fig, axes = plt.subplots(3, n_steps + 1, figsize=(4 * (n_steps + 1), 11))

    disp = np.array(image_mlx[0]) * IMAGENET_STD + IMAGENET_MEAN
    disp = np.clip(disp, 0, 1)
    for row in range(3):
        axes[row, 0].imshow(disp)
        axes[row, 0].set_title("Input")
        axes[row, 0].axis("off")

    for step in range(n_steps):
        cy, cx, s = TRAJECTORY[step]
        spatial_pt = pt_results[step]["canvas"][0, n_regs:]
        spatial_mlx = mlx_results[step]["canvas"][0, n_regs:]
        g = int(math.sqrt(spatial_pt.shape[0]))

        pca = PCA(n_components=3, whiten=True).fit(spatial_pt)

        for row, (spatial, label) in enumerate([(spatial_pt, "PyTorch"), (spatial_mlx, "MLX")]):
            proj = pca.transform(spatial)[:, :3]
            rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(proj, -10, 10)))
            axes[row, step + 1].imshow(rgb.reshape(g, g, 3))
            axes[row, step + 1].set_title(f"{label}\nc=({cy},{cx}) s={s}")
            axes[row, step + 1].axis("off")

        # Difference heatmap: mean absolute error per spatial token
        diff = np.abs(spatial_pt - spatial_mlx).mean(axis=-1).reshape(g, g)
        im = axes[2, step + 1].imshow(diff, cmap="hot", interpolation="nearest")
        axes[2, step + 1].set_title(f"MAE (max={diff.max():.2f})")
        axes[2, step + 1].axis("off")
        fig.colorbar(im, ax=axes[2, step + 1], fraction=0.046, pad=0.04)

    axes[0, 0].set_ylabel("PyTorch", fontsize=12)
    axes[1, 0].set_ylabel("MLX", fontsize=12)
    axes[2, 0].set_ylabel("Difference", fontsize=12)
    fig.suptitle("Canvas: PyTorch vs MLX (PCA) + per-token MAE", fontsize=13)
    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
