"""Real-time webcam PCA visualization with CanViT on MLX.

Single full-scene glimpse per frame, fresh canvas each time (no recurrence).
The model is not yet trained on video — this is a non-recurrent visualization
of what a single t=0 step produces from each webcam frame.

Controls:
    q               quit
    left/right      shift PCA components (0-2, 3-5, ...)
    up/down         change canvas grid size

Usage:
    uv run --group demos python demos/webcam.py
    uv run --group demos python demos/webcam.py --canvas-grid 16
    uv run --group demos python demos/webcam.py --model-repo canvit/my-model-mlx
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

import cv2
import mlx.core as mx
import mlx.nn
import numpy as np
import tyro
from numpy.typing import NDArray
from sklearn.decomposition import PCA

from canvit_mlx import load_from_hf_hub, Viewpoint, extract_glimpse_at_viewpoint

DISPLAY_SIZE = 512
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Arrow(Enum):
    """Arrow key codes across platforms (macOS, Linux GTK, Windows, Linux Qt)."""

    # fmt: off
    LEFT  = {63234, 65361, 2424832, 81, 2}
    RIGHT = {63235, 65363, 2555904, 83, 3}
    UP    = {63232, 65362, 2490368, 82, 0}
    DOWN  = {63233, 65364, 2621440, 84, 1}
    # fmt: on

    def matches(self, key: int) -> bool:
        return key in self.value


@dataclass
class Config:
    model_repo: str = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-mlx"
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


class TimingStats:
    """Rolling statistics for timing measurements."""

    def __init__(self, window: int = 100) -> None:
        self._data: dict[str, deque[float]] = {
            k: deque(maxlen=window) for k in ("pre", "inf", "pca", "tot")
        }

    def log(self, pre: float, inf: float, pca: float, tot: float) -> None:
        self._data["pre"].append(pre)
        self._data["inf"].append(inf)
        self._data["pca"].append(pca)
        self._data["tot"].append(tot)

    def summary(self) -> str:
        def stats(d: deque[float]) -> tuple[float, float]:
            return (float(np.mean(d)), float(np.std(d))) if d else (0.0, 0.0)

        d = {k: stats(v) for k, v in self._data.items()}
        fps = 1000.0 / d["tot"][0] if d["tot"][0] > 0 else 0
        return (
            f"FPS:{fps:5.1f} | "
            f"pre:{d['pre'][0]:4.1f}+-{d['pre'][1]:3.1f} | "
            f"inf:{d['inf'][0]:4.1f}+-{d['inf'][1]:3.1f} | "
            f"pca:{d['pca'][0]:4.1f}+-{d['pca'][1]:3.1f} | "
            f"tot:{d['tot'][0]:5.1f}+-{d['tot'][1]:4.1f}ms"
        )


def capture_scene(frame: np.ndarray, scene_size: int) -> mx.array:
    """BGR webcam frame → [1, scene_size, scene_size, 3] ImageNet-normalized scene (NHWC)."""
    h, w = frame.shape[:2]
    scale = scene_size / min(h, w)
    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = frame.shape[:2]
    y0, x0 = (h - scene_size) // 2, (w - scene_size) // 2
    frame = frame[y0:y0 + scene_size, x0:x0 + scene_size]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scene = rgb.astype(np.float32) / 255.0
    scene = (scene - IMAGENET_MEAN) / IMAGENET_STD
    return mx.array(scene)[None]


def glimpse_to_display(glimpse: mx.array) -> np.ndarray:
    """[1, H, W, 3] ImageNet-normalized → [DISPLAY_SIZE, DISPLAY_SIZE, 3] uint8 BGR for cv2."""
    img = np.array(glimpse[0]) * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                      (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)


def main(cfg: Config) -> None:
    print(f"Loading {cfg.model_repo}...")
    model = load_from_hf_hub(cfg.model_repo)
    mx.eval(model.parameters())
    n_canvas_regs = model.cfg.n_canvas_registers
    n_params = sum(v.size for _, v in mlx.nn.utils.tree_flatten(model.parameters()))
    print(f"  {n_params / 1e6:.1f}M params | {model.cfg.embed_dim}d, {model.cfg.n_blocks} blocks, canvas {model.cfg.canvas_dim}d")

    vp = Viewpoint.full_scene(1)
    stats = TimingStats()
    temporal_pca = TemporalPCA()

    print(f"Opening camera {cfg.camera}...")
    cap = cv2.VideoCapture(cfg.camera)
    assert cap.isOpened(), "Cannot open webcam"

    print("Running... q=quit, left/right=PCA, up/down=canvas size")
    pc_offset = 0
    canvas_grid = cfg.canvas_grid
    frame_count = 0

    while True:
        t_start = time.perf_counter()

        ret, frame = cap.read()
        assert ret, "Failed to capture"

        # Scene preprocessing + glimpse extraction
        t0 = time.perf_counter()
        scene = capture_scene(frame, cfg.scene_size)
        glimpse = extract_glimpse_at_viewpoint(scene, vp, glimpse_size_px=cfg.glimpse_px)
        mx.eval(glimpse)
        t_pre = (time.perf_counter() - t0) * 1000

        # Fresh state + inference
        t0 = time.perf_counter()
        state = model.init_state(1, canvas_grid)
        out = model(glimpse, state, vp)
        mx.eval(out.state.canvas)
        t_inf = (time.perf_counter() - t0) * 1000

        # PCA visualization
        t0 = time.perf_counter()
        spatial = np.array(out.state.canvas[0, n_canvas_regs:])
        pca_rgb = temporal_pca(spatial, component_offset=pc_offset)
        t_pca = (time.perf_counter() - t0) * 1000

        t_tot = (time.perf_counter() - t_start) * 1000
        stats.log(t_pre, t_inf, t_pca, t_tot)

        # Display: glimpse (what the model sees) | PCA
        input_vis = glimpse_to_display(glimpse)
        pca_vis = cv2.resize(cv2.cvtColor(pca_rgb, cv2.COLOR_RGB2BGR),
                             (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST)

        combined = np.hstack([input_vis, pca_vis])
        cv2.putText(combined, stats.summary(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(combined, f"PC {pc_offset}-{pc_offset+2} | canvas {canvas_grid}x{canvas_grid}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("CanViT PCA (MLX)", combined)

        frame_count += 1
        if frame_count % 100 == 0:
            print(stats.summary())

        key = cv2.waitKeyEx(1)
        if key & 0xFF == ord("q"):
            break
        elif Arrow.LEFT.matches(key):
            pc_offset = max(0, pc_offset - 1)
            temporal_pca.reset()
            print(f"PC {pc_offset}-{pc_offset+2}")
        elif Arrow.RIGHT.matches(key):
            pc_offset += 1
            temporal_pca.reset()
            print(f"PC {pc_offset}-{pc_offset+2}")
        elif Arrow.UP.matches(key):
            canvas_grid = min(128, canvas_grid * 2)
            temporal_pca.reset()
            print(f"Canvas: {canvas_grid}x{canvas_grid}")
        elif Arrow.DOWN.matches(key):
            canvas_grid = max(4, canvas_grid // 2)
            temporal_pca.reset()
            print(f"Canvas: {canvas_grid}x{canvas_grid}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal: {stats.summary()}")


if __name__ == "__main__":
    main(tyro.cli(Config))
