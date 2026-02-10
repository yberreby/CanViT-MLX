"""Canvas linear probes: location + scale decodability from frozen CanViT.

A MNIST digit (random scale 14-56px) is placed at a random location on a
256x256 image. A frozen CanViT processes the full scene once. Tiny linear
probes on the canvas spatial tokens predict digit center and scale.

Demonstrates that the canvas linearly encodes object location and size
without any task-specific training.

Usage:
    uv run python rl/canvas_probes.py
    uv run python rl/canvas_probes.py --n-steps 2000 --batch-size 8
"""

import logging
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tyro
from mlx.utils import tree_flatten
from tqdm import tqdm

from canvit_mlx import Viewpoint, extract_glimpse_at_viewpoint, load_canvit
from canvit_mlx.grid import grid_coords

log = logging.getLogger(__name__)

_MEAN_NP = np.array([0.485, 0.456, 0.406])
_STD_NP = np.array([0.229, 0.224, 0.225])


@dataclass
class Config:
    weights: str = "weights/canvit-vitb16-pretrain-512px-in21k.safetensors"
    image_size: int = 256
    canvas_grid: int = 32
    glimpse_px: int = 64
    batch_size: int = 4
    n_steps: int = 1000
    lr: float = 1e-2
    digit_min_px: int = 14
    digit_max_px: int = 56
    ema_alpha: float = 0.95
    log_interval: int = 10
    viz_interval: int = 200
    out_dir: str = "outputs/canvas_probes"


# -- Data --------------------------------------------------------------------


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST
    ds = MNIST("/tmp/mnist", download=True, train=True)
    return ds.data.numpy(), ds.targets.numpy()


def make_batch(
    mnist_images: np.ndarray,
    B: int, canvas_size: int,
    digit_min_px: int, digit_max_px: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (images_norm, centers_norm, scales_norm).

    centers_norm: [B, 2] in [-1, 1] (y, x).
    scales_norm: [B] digit size / canvas_size, in (0, 1).
    """
    idx = np.random.randint(0, len(mnist_images), size=B)

    # Random digit size per sample
    digit_sizes = np.random.randint(digit_min_px, digit_max_px + 1, size=B)

    canvas = np.random.normal(0.0, 0.05, (B, canvas_size, canvas_size, 3)).astype(np.float32)
    centers = np.zeros((B, 2), dtype=np.float32)
    scales = np.zeros(B, dtype=np.float32)

    for i in range(B):
        sz = digit_sizes[i]
        # Resize digit to sz x sz via nearest-neighbor
        src = mnist_images[idx[i]].astype(np.float32) / 255.0
        # Simple resize: repeat pixels
        ys = np.linspace(0, 27, sz).astype(int)
        xs = np.linspace(0, 27, sz).astype(int)
        digit = src[np.ix_(ys, xs)]

        max_pos = canvas_size - sz
        py = np.random.randint(0, max(1, max_pos))
        px = np.random.randint(0, max(1, max_pos))

        for c in range(3):
            canvas[i, py:py+sz, px:px+sz, c] = digit

        centers[i, 0] = (py + sz / 2) / canvas_size * 2 - 1
        centers[i, 1] = (px + sz / 2) / canvas_size * 2 - 1
        scales[i] = sz / canvas_size

    images = (canvas - _MEAN_NP) / _STD_NP
    return mx.array(images), mx.array(centers), mx.array(scales)


# -- Probes ------------------------------------------------------------------


class LocationProbe(nn.Module):
    """Spatial softmax: LN + Linear(D,1) + softmax -> weighted coords."""

    def __init__(self, canvas_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(canvas_dim)
        self.proj = nn.Linear(canvas_dim, 1)

    def __call__(self, spatial: mx.array, grid_size: int) -> mx.array:
        logits = self.proj(self.ln(spatial)).squeeze(-1)
        weights = mx.softmax(logits, axis=-1)
        coords = grid_coords(grid_size, grid_size).reshape(1, -1, 2)
        return mx.sum(weights[:, :, None] * coords, axis=1)

    def attention_map(self, spatial: mx.array, grid_size: int) -> mx.array:
        """Returns [B, G, G] softmax attention weights."""
        logits = self.proj(self.ln(spatial)).squeeze(-1)
        return mx.softmax(logits, axis=-1).reshape(-1, grid_size, grid_size)


class ScaleProbe(nn.Module):
    """Mean-pool + Linear(D,1) + sigmoid -> scalar in [0, 1]."""

    def __init__(self, canvas_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(canvas_dim)
        self.proj = nn.Linear(canvas_dim, 1)

    def __call__(self, spatial: mx.array) -> mx.array:
        pooled = mx.mean(self.ln(spatial), axis=1)
        return mx.sigmoid(self.proj(pooled)).squeeze(-1)


class Probes(nn.Module):
    def __init__(self, canvas_dim: int):
        super().__init__()
        self.location = LocationProbe(canvas_dim)
        self.scale = ScaleProbe(canvas_dim)


# -- Training ----------------------------------------------------------------


def train(cfg: Config) -> None:
    import mlflow

    mnist_images, _ = load_mnist()
    log.info("MNIST: %d images", len(mnist_images))

    model = load_canvit(cfg.weights)
    model.freeze()
    n_regs = model.cfg.n_canvas_registers
    grid = cfg.canvas_grid

    probes = Probes(model.cfg.canvas_dim)
    n_params = sum(p.size for _, p in tree_flatten(probes.trainable_parameters()))
    log.info("Probe params: %d", n_params)

    probe_opt = optim.Adam(learning_rate=cfg.lr)

    def loss_fn(probes: Probes, spatial: mx.array, target_centers: mx.array, target_scales: mx.array) -> mx.array:
        loc_pred = probes.location(spatial, grid)
        scale_pred = probes.scale(spatial)
        loc_mse = mx.mean(mx.sum((loc_pred - target_centers) ** 2, axis=-1))
        scale_mse = mx.mean((scale_pred - target_scales) ** 2)
        return loc_mse + scale_mse

    loss_and_grad = nn.value_and_grad(probes, loss_fn)

    mlflow.log_params({**vars(cfg), "n_probe_params": n_params})

    ema_loss = 0.0
    ema_loc = 0.0
    ema_scale = 0.0
    alpha = cfg.ema_alpha

    history: dict[str, list[float]] = {"loss": [], "loc_mse": [], "scale_mse": [], "step": []}
    t0 = time.monotonic()

    for step in tqdm(range(cfg.n_steps), desc="canvas-probes"):
        images, centers, scales = make_batch(
            mnist_images, cfg.batch_size, cfg.image_size,
            cfg.digit_min_px, cfg.digit_max_px,
        )
        vp = Viewpoint.full_scene(cfg.batch_size)
        glimpse = extract_glimpse_at_viewpoint(images, vp, cfg.glimpse_px)
        out = model(glimpse, model.init_state(cfg.batch_size, grid), vp)
        spatial = mx.stop_gradient(out.state.canvas[:, n_regs:])

        loss, grads = loss_and_grad(probes, spatial, centers, scales)
        probe_opt.update(probes, grads)

        # Also compute individual losses for logging
        loc_pred = probes.location(spatial, grid)
        scale_pred = probes.scale(spatial)
        loc_mse = mx.mean(mx.sum((loc_pred - centers) ** 2, axis=-1))
        scale_mse = mx.mean((scale_pred - scales) ** 2)

        is_log = step % cfg.log_interval == 0
        if is_log:
            mx.eval(probes.parameters(), probe_opt.state, loss, loc_mse, scale_mse)
        else:
            mx.eval(probes.parameters(), probe_opt.state)

        if is_log:
            l, lm, sm = loss.item(), loc_mse.item(), scale_mse.item()
            bc = 1 - alpha ** (step + 1)
            ema_loss = alpha * ema_loss + (1 - alpha) * l
            ema_loc = alpha * ema_loc + (1 - alpha) * lm
            ema_scale = alpha * ema_scale + (1 - alpha) * sm

            history["loss"].append(ema_loss / bc)
            history["loc_mse"].append(ema_loc / bc)
            history["scale_mse"].append(ema_scale / bc)
            history["step"].append(step)

            mlflow.log_metrics({
                "ema/loss": ema_loss / bc,
                "ema/loc_mse": ema_loc / bc,
                "ema/scale_mse": ema_scale / bc,
            }, step=step)

            if step % (cfg.log_interval * 10) == 0:
                tqdm.write(
                    f"step={step:04d}  loss={ema_loss/bc:.6f}  "
                    f"loc={ema_loc/bc:.6f}  scale={ema_scale/bc:.6f}"
                )

        if step % cfg.viz_interval == 0 or step == cfg.n_steps - 1:
            fig = _make_viz(model, probes, cfg, step, mnist_images)
            mlflow.log_figure(fig, f"viz/step_{step:05d}.png")
            plt.close(fig)

    elapsed = time.monotonic() - t0
    log.info("Done: %d steps in %.1fs (%.1f step/s)", cfg.n_steps, elapsed, cfg.n_steps / elapsed)

    # Final convergence plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history["step"], history["loc_mse"])
    ax1.set_xlabel("step")
    ax1.set_ylabel("location MSE")
    ax1.set_title("Location probe")
    ax1.set_yscale("log")
    ax2.plot(history["step"], history["scale_mse"])
    ax2.set_xlabel("step")
    ax2.set_ylabel("scale MSE")
    ax2.set_title("Scale probe")
    ax2.set_yscale("log")
    fig.tight_layout()
    mlflow.log_figure(fig, "convergence.png")
    plt.close(fig)


def _make_viz(model, probes: Probes, cfg: Config, step: int,
              mnist_images: np.ndarray) -> plt.Figure:
    N = 4
    grid = cfg.canvas_grid
    n_regs = model.cfg.n_canvas_registers
    images, centers, scales = make_batch(
        mnist_images, N, cfg.image_size, cfg.digit_min_px, cfg.digit_max_px,
    )
    vp = Viewpoint.full_scene(N)
    glimpse = extract_glimpse_at_viewpoint(images, vp, cfg.glimpse_px)
    out = model(glimpse, model.init_state(N, grid), vp)
    spatial = mx.stop_gradient(out.state.canvas[:, n_regs:])

    loc_pred = probes.location(spatial, grid)
    scale_pred = probes.scale(spatial)
    attn_map = probes.location.attention_map(spatial, grid)
    mx.eval(images, centers, scales, loc_pred, scale_pred, attn_map)

    attn_np = np.array(attn_map)  # [N, G, G]

    fig, axes = plt.subplots(2, N, figsize=(3 * N, 6))
    if N == 1:
        axes = axes[:, None]
    for i in range(N):
        img = np.clip(np.array(images[i]) * _STD_NP + _MEAN_NP, 0, 1)
        H, W = img.shape[:2]
        gy, gx = float(centers[i, 0]), float(centers[i, 1])
        py, px = float(loc_pred[i, 0]), float(loc_pred[i, 1])

        # Row 0: image + markers
        axes[0, i].imshow(img)
        axes[0, i].plot((gx+1)/2*W, (gy+1)/2*H, "o", color="lime", markersize=8,
                        markeredgewidth=1.5, markeredgecolor="white")
        axes[0, i].plot((px+1)/2*W, (py+1)/2*H, "x", color="yellow", markersize=8,
                        markeredgewidth=2)
        gs = float(scales[i]) * cfg.image_size
        axes[0, i].add_patch(plt.Rectangle(
            ((gx+1)/2*W - gs/2, (gy+1)/2*H - gs/2), gs, gs,
            linewidth=1.5, edgecolor="lime", facecolor="none", linestyle="--"))
        ps = float(scale_pred[i]) * cfg.image_size
        axes[0, i].add_patch(plt.Rectangle(
            ((px+1)/2*W - ps/2, (py+1)/2*H - ps/2), ps, ps,
            linewidth=1.5, edgecolor="yellow", facecolor="none", linestyle=":"))
        axes[0, i].set_title(f"s_gt={float(scales[i]):.2f} s_pred={float(scale_pred[i]):.2f}", fontsize=8)
        axes[0, i].axis("off")

        # Row 1: attention heatmap overlaid on image
        axes[1, i].imshow(img)
        # Upsample heatmap to image size
        from scipy.ndimage import zoom as ndzoom
        hmap = ndzoom(attn_np[i], H / grid, order=0)
        axes[1, i].imshow(hmap, cmap="hot", alpha=0.6, extent=(0, W, H, 0))
        axes[1, i].set_title("attention", fontsize=8)
        axes[1, i].axis("off")

    fig.suptitle(f"Step {step} — green=GT, yellow=pred", fontsize=10)
    fig.tight_layout()
    return fig


def main(cfg: Config) -> None:
    import mlflow
    mlflow.set_experiment("canvas-probes")
    with mlflow.start_run(run_name="loc+scale"):
        train(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(tyro.cli(Config))
