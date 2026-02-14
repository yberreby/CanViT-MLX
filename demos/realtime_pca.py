"""Real-time webcam PCA visualization with CanViT on MLX.

Fresh state each frame (t=0), full-scene viewpoint.
Displays: webcam feed | canvas PCA features.

Controls:
    q               quit
    left/right      shift PCA components (0-2, 3-5, ...)
    up/down         change canvas grid size

Usage:
    uv run python demos/realtime_pca.py
    uv run python demos/realtime_pca.py --canvas-grid 16
    uv run python demos/realtime_pca.py --hf-repo canvit/my-model-mlx
"""

import math
import time
from collections import deque
from dataclasses import dataclass
import cv2
import mlx.core as mx
import numpy as np
import tyro
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from canvit_mlx import load_canvit, Viewpoint

DISPLAY_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class Config:
    hf_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx"
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


def capture_frame(cap: cv2.VideoCapture, size: int) -> np.ndarray:
    """Center-crop square, resize, return uint8 RGB [H,W,3]."""
    ret, frame = cap.read()
    assert ret, "Failed to capture"
    h, w = frame.shape[:2]
    crop = min(h, w)
    y0, x0 = (h - crop) // 2, (w - crop) // 2
    frame = frame[y0:y0+crop, x0:x0+crop]
    frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def preprocess(frame_rgb: np.ndarray) -> mx.array:
    """RGB uint8 [H,W,3] -> ImageNet-normalized [1,H,W,3] float32."""
    x = frame_rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return mx.array(x)[None]


def main(cfg: Config) -> None:
    print(f"Loading {cfg.hf_repo}...")
    model = load_canvit(cfg.hf_repo)
    mx.eval(model.parameters())
    n_canvas_regs = model.cfg.n_canvas_registers
    n_params = sum(v.size for _, v in mx.utils.tree_flatten(model.parameters()))
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

        frame_rgb = capture_frame(cap, cfg.glimpse_px)
        glimpse = preprocess(frame_rgb)

        state = model.init_state(1, canvas_grid)
        out = model(glimpse, state, vp)
        mx.eval(out.state.canvas)

        spatial = np.array(out.state.canvas[0, n_canvas_regs:])
        pca_rgb = temporal_pca(spatial, component_offset=pc_offset)

        elapsed = time.monotonic() - t0
        fps_buf.append(1.0 / elapsed if elapsed > 0 else 0)
        fps = np.mean(fps_buf)

        input_vis = cv2.resize(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), (DISPLAY_SIZE, DISPLAY_SIZE))
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
