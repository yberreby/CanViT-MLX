"""Real-time webcam PCA visualization with CanViT on MLX.

Single full-scene glimpse per frame, fresh canvas each time (no recurrence).
The model is not yet trained on video — this is a non-recurrent visualization
of what a single t=0 step produces from each webcam frame.

Controls:
    q               quit
    left/right      shift PCA components (0-2, 3-5, ...)
    up/down         change canvas grid size

Usage:
    uv run --group demos python demos/realtime_pca.py
    uv run --group demos python demos/realtime_pca.py --canvas-grid 16
    uv run --group demos python demos/realtime_pca.py --hf-repo canvit/my-model-mlx
"""

import math
import time
from collections import deque
from dataclasses import dataclass
import cv2
import mlx.core as mx
import mlx.nn
import numpy as np
import tyro
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from canvit_mlx import load_canvit, Viewpoint, extract_glimpse_at_viewpoint

DISPLAY_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Config:
    hf_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx"
    scene_size: int = 512
    canvas_grid: int = 8
    glimpse_px: int = 128
    camera: int = 0


class TemporalPCA:
    """PCA with sign alignment across frames to reduce flicker."""

    def __init__(self) -> None:
        self._prev_components: NDArray[np.floating] | None = None
        self._prev_offset: int | None = None

    def reset(self) -> None:
        self._prev_components = None
        self._prev_offset = None

    def __call__(self, spatial: NDArray[np.floating], *, component_offset: int = 0) -> NDArray[np.uint8]:
        if self._prev_offset is not None and self._prev_offset != component_offset:
            self._prev_components = None
        self._prev_offset = component_offset

        n_tokens = spatial.shape[0]
        grid = int(math.sqrt(n_tokens))
        assert grid * grid == n_tokens
        n_comp = component_offset + 3

        pca = PCA(n_components=n_comp, whiten=True)
        proj = pca.fit_transform(spatial)

        if self._prev_components is not None and self._prev_components.shape == pca.components_.shape:
            for i in range(n_comp):
                if np.dot(pca.components_[i], self._prev_components[i]) < 0:
                    pca.components_[i] *= -1
                    proj[:, i] *= -1

        self._prev_components = pca.components_.copy()
        rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(proj[:, component_offset:component_offset + 3], -10, 10)))
        return (rgb.reshape(grid, grid, 3) * 255).astype(np.uint8)


def capture_scene(cap: cv2.VideoCapture, scene_size: int) -> tuple[np.ndarray, mx.array]:
    """Capture webcam frame → (display_rgb uint8, scene float32 [1,H,W,3]).

    Resize shortest edge, center crop to square, ImageNet normalize.
    """
    ret, frame = cap.read()
    assert ret, "Failed to capture"
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    # Resize shortest edge to scene_size
    scale = scene_size / min(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    # Center crop
    h, w = frame.shape[:2]
    y0, x0 = (h - scene_size) // 2, (w - scene_size) // 2
    frame = frame[y0:y0 + scene_size, x0:x0 + scene_size]
    # Normalize
    scene = frame.astype(np.float32) / 255.0
    scene = (scene - IMAGENET_MEAN) / IMAGENET_STD
    return frame, mx.array(scene)[None]


def main(cfg: Config) -> None:
    print(f"Loading {cfg.hf_repo}...")
    model = load_canvit(cfg.hf_repo)
    mx.eval(model.parameters())
    n_canvas_regs = model.cfg.n_canvas_registers
    n_params = sum(v.size for _, v in mlx.nn.utils.tree_flatten(model.parameters()))
    print(f"  {n_params / 1e6:.1f}M params | {model.cfg.embed_dim}d, {model.cfg.n_blocks} blocks, canvas {model.cfg.canvas_dim}d")

    vp = Viewpoint.full_scene(1)
    temporal_pca = TemporalPCA()

    cap = cv2.VideoCapture(cfg.camera)
    assert cap.isOpened(), "Cannot open webcam"

    print("Running... q=quit, left/right=PCA offset, up/down=canvas grid")
    pc_offset = 0
    canvas_grid = cfg.canvas_grid
    fps_buf: deque[float] = deque(maxlen=60)

    while True:
        t0 = time.monotonic()

        display_rgb, scene = capture_scene(cap, cfg.scene_size)
        glimpse = extract_glimpse_at_viewpoint(scene, vp, glimpse_size_px=cfg.glimpse_px)

        state = model.init_state(1, canvas_grid)
        out = model(glimpse, state, vp)
        mx.eval(out.state.canvas)

        spatial = np.array(out.state.canvas[0, n_canvas_regs:])
        pca_rgb = temporal_pca(spatial, component_offset=pc_offset)

        elapsed = time.monotonic() - t0
        fps_buf.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = np.mean(fps_buf)

        # Show the glimpse (what the model actually sees) and PCA side by side
        glimpse_rgb = np.array(glimpse[0] * mx.array(IMAGENET_STD) + mx.array(IMAGENET_MEAN))
        glimpse_rgb = np.clip(glimpse_rgb * 255, 0, 255).astype(np.uint8)
        input_vis = cv2.resize(cv2.cvtColor(glimpse_rgb, cv2.COLOR_RGB2BGR),
                               (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)
        pca_vis = cv2.resize(cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR),
                             (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack([input_vis, pca_vis])
        cv2.putText(combined, f"FPS:{fps:.1f} | PC {pc_offset}-{pc_offset+2} | canvas {canvas_grid}x{canvas_grid}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("CanViT PCA (MLX)", combined)

        key = cv2.waitKeyEx(1)
        if key & 0xFF == ord("q"):
            break
        elif key in {63234, 65361, 2424832, 2}:  # LEFT
            pc_offset = max(0, pc_offset - 1)
            temporal_pca.reset()
        elif key in {63235, 65363, 2555904, 3}:  # RIGHT
            pc_offset += 1
            temporal_pca.reset()
        elif key in {63232, 65362, 2490368, 0}:  # UP
            canvas_grid = min(128, canvas_grid * 2)
            temporal_pca.reset()
        elif key in {63233, 65364, 2621440, 1}:  # DOWN
            canvas_grid = max(4, canvas_grid // 2)
            temporal_pca.reset()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(tyro.cli(Config))
