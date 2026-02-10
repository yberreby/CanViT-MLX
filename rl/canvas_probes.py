"""Canvas per-class location probes on frozen CanViT.

Multiple MNIST digits (unique classes, fixed scale) placed on a 256x256
canvas. Frozen CanViT processes the full scene once. A single linear probe
(D -> 10) on the canvas spatial tokens predicts each digit class's center
independently via per-class spatial softmax.

Demonstrates class-selective spatial attention: each probe output attends to
its digit and ignores distractors, using only frozen pretrained features.

Usage:
    uv run python rl/canvas_probes.py
    uv run python rl/canvas_probes.py --n-steps 2000 --digits-per-image 5
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tyro
from mlx.utils import tree_flatten
from scipy.ndimage import zoom as ndzoom
from tqdm import tqdm

from canvit_mlx import Viewpoint, extract_glimpse_at_viewpoint, load_canvit
from canvit_mlx.grid import grid_coords

log = logging.getLogger(__name__)

_MEAN_NP = np.array([0.485, 0.456, 0.406])
_STD_NP = np.array([0.229, 0.224, 0.225])
N_CLASSES = 10


@dataclass
class Config:
    weights: str = "weights/canvit-vitb16-pretrain-512px-in21k.safetensors"
    image_size: int = 256
    canvas_grid: int = 32
    glimpse_px: int = 64
    batch_size: int = 8
    n_steps: int = 5000
    lr: float = 3e-3
    digits_per_image: int = 10
    digit_px: int = 32
    ema_alpha: float = 0.95
    log_interval: int = 10
    viz_interval: int = 500
    out_dir: str = "outputs/canvas_probes"


# -- Data --------------------------------------------------------------------


def load_mnist() -> tuple[np.ndarray, np.ndarray]:
    from torchvision.datasets import MNIST
    ds = MNIST("/tmp/mnist", download=True, train=True)
    return ds.data.numpy(), ds.targets.numpy()


def indices_by_class(labels: np.ndarray) -> list[np.ndarray]:
    return [np.where(labels == c)[0] for c in range(N_CLASSES)]


def make_batch(
    mnist_images: np.ndarray,
    by_class: list[np.ndarray],
    B: int, canvas_size: int,
    digits_per_image: int,
    digit_px: int,
) -> tuple[mx.array, mx.array, mx.array]:
    """Returns (images_norm, centers, present).

    centers: [B, 10, 2] in [-1, 1] (y, x). Valid only where present=True.
    present: [B, 10] float mask (1.0 where class is on the canvas).
    """
    canvas = np.random.normal(0.0, 0.05, (B, canvas_size, canvas_size, 3)).astype(np.float32)
    centers = np.zeros((B, N_CLASSES, 2), dtype=np.float32)
    present = np.zeros((B, N_CLASSES), dtype=np.float32)

    for i in range(B):
        classes = np.random.choice(N_CLASSES, size=digits_per_image, replace=False)
        for cls in classes:
            idx = by_class[cls][np.random.randint(len(by_class[cls]))]
            src = mnist_images[idx].astype(np.float32) / 255.0

            ys = np.linspace(0, 27, digit_px).astype(int)
            xs = np.linspace(0, 27, digit_px).astype(int)
            digit = src[np.ix_(ys, xs)]

            max_pos = canvas_size - digit_px
            py = np.random.randint(0, max(1, max_pos))
            px = np.random.randint(0, max(1, max_pos))

            for c in range(3):
                canvas[i, py:py+digit_px, px:px+digit_px, c] = digit

            centers[i, cls, 0] = (py + digit_px / 2) / canvas_size * 2 - 1
            centers[i, cls, 1] = (px + digit_px / 2) / canvas_size * 2 - 1
            present[i, cls] = 1.0

    images = (canvas - _MEAN_NP) / _STD_NP
    return mx.array(images), mx.array(centers), mx.array(present)


# -- Probe -------------------------------------------------------------------


class PerClassLocationProbe(nn.Module):
    """Shared LN + Linear(D, C): per-class spatial softmax -> per-class centers.

    Equivalent to C independent LocationProbe(D,1), but vectorized.
    """

    def __init__(self, canvas_dim: int, n_classes: int):
        super().__init__()
        self.ln = nn.LayerNorm(canvas_dim)
        self.proj = nn.Linear(canvas_dim, n_classes)

    def __call__(self, spatial: mx.array, grid_size: int) -> mx.array:
        """[B, S, D] -> [B, C, 2] predicted centers per class."""
        logits = self.proj(self.ln(spatial))  # [B, S, C]
        weights = mx.softmax(logits, axis=1)  # softmax over spatial dim
        coords = grid_coords(grid_size, grid_size).reshape(1, -1, 1, 2)  # [1, S, 1, 2]
        return mx.sum(weights[:, :, :, None] * coords, axis=1)  # [B, C, 2]

    def attention_maps(self, spatial: mx.array, grid_size: int) -> mx.array:
        """[B, S, D] -> [B, C, G, G] per-class attention heatmaps."""
        logits = self.proj(self.ln(spatial))  # [B, S, C]
        weights = mx.softmax(logits, axis=1)
        B = weights.shape[0]
        return weights.reshape(B, grid_size, grid_size, -1).transpose(0, 3, 1, 2)


# -- Training ----------------------------------------------------------------


def train(cfg: Config) -> None:
    import mlflow

    mnist_images, mnist_labels = load_mnist()
    by_class = indices_by_class(mnist_labels)
    log.info("MNIST: %d images", len(mnist_images))

    model = load_canvit(cfg.weights)
    model.freeze()
    n_regs = model.cfg.n_canvas_registers
    grid = cfg.canvas_grid

    probe = PerClassLocationProbe(model.cfg.canvas_dim, N_CLASSES)
    n_params = sum(p.size for _, p in tree_flatten(probe.trainable_parameters()))
    log.info("Probe params: %d", n_params)

    opt = optim.Adam(learning_rate=cfg.lr)

    def loss_fn(probe: PerClassLocationProbe, spatial: mx.array,
                target_centers: mx.array, present: mx.array) -> mx.array:
        pred = probe(spatial, grid)  # [B, C, 2]
        err = mx.sum((pred - target_centers) ** 2, axis=-1)  # [B, C]
        return mx.sum(err * present) / mx.sum(present)

    loss_and_grad = nn.value_and_grad(probe, loss_fn)

    out_dir = Path(cfg.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log.info("Artifacts → %s", out_dir)

    mlflow.log_params({**vars(cfg), "n_probe_params": n_params, "out_dir": str(out_dir)})

    ema_loss = 0.0
    alpha = cfg.ema_alpha
    history: dict[str, list[float]] = {"ema_loss": [], "raw_loss": [], "grad_norm": [], "step": []}
    # Per-class histories: list of (step, class_mse_array) for classes present
    per_class_history: dict[str, list[float]] = {f"class_{c}": [] for c in range(N_CLASSES)}
    per_class_steps: dict[str, list[int]] = {f"class_{c}": [] for c in range(N_CLASSES)}
    t0 = time.monotonic()

    for step in tqdm(range(cfg.n_steps), desc="canvas-probes"):
        images, centers, present = make_batch(
            mnist_images, by_class, cfg.batch_size, cfg.image_size,
            cfg.digits_per_image, cfg.digit_px,
        )
        vp = Viewpoint.full_scene(cfg.batch_size)
        glimpse = extract_glimpse_at_viewpoint(images, vp, cfg.glimpse_px)
        out = model(glimpse, model.init_state(cfg.batch_size, grid), vp)
        spatial = mx.stop_gradient(out.state.canvas[:, n_regs:])

        loss, grads = loss_and_grad(probe, spatial, centers, present)
        grad_norm = mx.sqrt(sum(mx.sum(g ** 2) for _, g in tree_flatten(grads)))
        opt.update(probe, grads)

        is_log = step % cfg.log_interval == 0
        if is_log:
            # Per-class MSE (recompute — probe is tiny, cheap)
            pred = probe(spatial, grid)  # [B, C, 2]
            err = mx.sum((pred - centers) ** 2, axis=-1)  # [B, C]
            class_count = mx.sum(present, axis=0)  # [C]
            per_class_mse = mx.sum(err * present, axis=0) / mx.maximum(class_count, 1.0)  # [C]
            mx.eval(probe.parameters(), opt.state, loss, grad_norm, per_class_mse, class_count)
        else:
            mx.eval(probe.parameters(), opt.state)

        if is_log:
            raw = loss.item()
            gn = grad_norm.item()
            bc = 1 - alpha ** (step + 1)
            ema_loss = alpha * ema_loss + (1 - alpha) * raw

            history["ema_loss"].append(ema_loss / bc)
            history["raw_loss"].append(raw)
            history["grad_norm"].append(gn)
            history["step"].append(step)

            metrics: dict[str, float] = {
                "ema/loss": ema_loss / bc,
                "raw/loss": raw,
                "grad_norm": gn,
            }
            # Per-class MSE for present classes
            pc_mse = np.array(per_class_mse)
            cc = np.array(class_count)
            for c in range(N_CLASSES):
                if cc[c] > 0:
                    metrics[f"class/{c}_mse"] = float(pc_mse[c])
                    per_class_history[f"class_{c}"].append(float(pc_mse[c]))
                    per_class_steps[f"class_{c}"].append(step)
            mlflow.log_metrics(metrics, step=step)

            if step % (cfg.log_interval * 10) == 0:
                present_mse = [f"{c}:{pc_mse[c]:.4f}" for c in range(N_CLASSES) if cc[c] > 0]
                tqdm.write(
                    f"step={step:04d}  ema={ema_loss/bc:.6f}  raw={raw:.6f}  "
                    f"gnorm={gn:.4f}  [{', '.join(present_mse)}]"  # noqa: E501
                )

        if step % cfg.viz_interval == 0 or step == cfg.n_steps - 1:
            fig = _make_viz(model, probe, cfg, step, mnist_images, by_class)
            fig.savefig(str(out_dir / f"viz_{step:05d}.png"), dpi=150)
            mlflow.log_figure(fig, f"viz/step_{step:05d}.png")
            plt.close(fig)

    elapsed = time.monotonic() - t0
    log.info("Done: %d steps in %.1fs (%.1f step/s)", cfg.n_steps, elapsed, cfg.n_steps / elapsed)

    # Save probe checkpoint
    ckpt_path = out_dir / "probe.npz"
    probe.save_weights(str(ckpt_path))
    log.info("Checkpoint → %s (%.1f KB)", ckpt_path, ckpt_path.stat().st_size / 1024)

    # Save full metrics history
    metrics_path = out_dir / "metrics.npz"
    save_dict: dict[str, np.ndarray] = {k: np.array(v) for k, v in history.items()}
    for c in range(N_CLASSES):
        key = f"class_{c}"
        save_dict[f"{key}_mse"] = np.array(per_class_history[key])
        save_dict[f"{key}_steps"] = np.array(per_class_steps[key])
    np.savez(str(metrics_path), **save_dict)
    log.info("Metrics → %s", metrics_path)

    # Convergence plot
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["step"], history["ema_loss"], label="EMA")
    axes[0].plot(history["step"], history["raw_loss"], alpha=0.3, label="raw")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("location MSE")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].set_title("Loss")
    axes[1].plot(history["step"], history["grad_norm"])
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("grad norm")
    axes[1].set_title("Gradient norm")
    # Per-class EMA
    for c in range(N_CLASSES):
        key = f"class_{c}"
        steps = per_class_steps[key]
        vals = per_class_history[key]
        if len(vals) > 10:
            axes[2].plot(steps, vals, alpha=0.4, label=str(c), color=_COLORS[c])
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("per-class MSE")
    axes[2].set_title("Per-class location MSE")
    axes[2].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    conv_path = out_dir / "convergence.png"
    fig.savefig(str(conv_path), dpi=150)
    mlflow.log_figure(fig, "convergence.png")
    plt.close(fig)
    log.info("Plot → %s", conv_path)


_COLORS = plt.cm.tab10(np.arange(N_CLASSES))[:, :3]


def _make_viz(model, probe: PerClassLocationProbe, cfg: Config, step: int,
              mnist_images: np.ndarray, by_class: list[np.ndarray]) -> plt.Figure:
    N = 4
    K = cfg.digits_per_image
    grid = cfg.canvas_grid
    n_regs = model.cfg.n_canvas_registers

    images, centers, present = make_batch(
        mnist_images, by_class, N, cfg.image_size,
        cfg.digits_per_image, cfg.digit_px,
    )
    vp = Viewpoint.full_scene(N)
    glimpse = extract_glimpse_at_viewpoint(images, vp, cfg.glimpse_px)
    out = model(glimpse, model.init_state(N, grid), vp)
    spatial = mx.stop_gradient(out.state.canvas[:, n_regs:])

    pred = probe(spatial, grid)  # [N, C, 2]
    attn = probe.attention_maps(spatial, grid)  # [N, C, G, G]
    mx.eval(images, centers, present, pred, attn)

    present_np = np.array(present)
    pred_np = np.array(pred)
    centers_np = np.array(centers)
    attn_np = np.array(attn)

    n_rows = 1 + K
    fig, axes = plt.subplots(n_rows, N, figsize=(3.5 * N, 3 * n_rows))
    if N == 1:
        axes = axes[:, None]

    for i in range(N):
        img = np.clip(np.array(images[i]) * _STD_NP + _MEAN_NP, 0, 1)
        H, W = img.shape[:2]
        present_classes = np.where(present_np[i] > 0.5)[0]

        # Row 0: image + colored markers per class
        axes[0, i].imshow(img)
        for cls in present_classes:
            gy, gx = centers_np[i, cls, 0], centers_np[i, cls, 1]
            py, px = pred_np[i, cls, 0], pred_np[i, cls, 1]
            color = _COLORS[cls]
            axes[0, i].plot((gx+1)/2*W, (gy+1)/2*H, "o", color=color,
                            markersize=8, markeredgewidth=1.5, markeredgecolor="white")
            axes[0, i].plot((px+1)/2*W, (py+1)/2*H, "x", color=color,
                            markersize=8, markeredgewidth=2)
        axes[0, i].set_title(",".join(str(c) for c in present_classes), fontsize=9)
        axes[0, i].axis("off")

        # Rows 1..K: per-class attention heatmap
        for j, cls in enumerate(present_classes[:K]):
            ax = axes[1 + j, i]
            ax.imshow(img)
            hmap = ndzoom(attn_np[i, cls], H / grid, order=0)
            ax.imshow(hmap, cmap="hot", alpha=0.6, extent=(0, W, H, 0))
            ax.set_title(f"probe {cls}", fontsize=8, color=_COLORS[cls])
            ax.axis("off")

        for j in range(len(present_classes), K):
            axes[1 + j, i].set_visible(False)

    fig.suptitle(f"Step {step} — o=GT, x=pred (color=class)", fontsize=10)
    fig.tight_layout()
    return fig


def main(cfg: Config) -> None:
    import mlflow
    mlflow.set_experiment("canvas-probes")
    with mlflow.start_run(run_name="multi-digit"):
        train(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main(tyro.cli(Config))
