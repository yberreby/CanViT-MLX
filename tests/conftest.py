"""Shared test utilities. Only this module imports canvit (PyTorch reference)."""

import logging

import mlx.core as mx
import numpy as np
import pytest
import torch

log = logging.getLogger(__name__)

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
_MODEL_NAME = HF_REPO.split("/")[-1]  # org/model → model
WEIGHTS = f"weights/{_MODEL_NAME}.safetensors"
SEED = 42
B = 1
CANVAS_GRID = 32
GLIMPSE_PX = 128


def assert_close(
    name: str, ref: np.ndarray, got: mx.array, *, atol: float, rtol: float = 1e-4
) -> None:
    got = np.array(got)
    assert ref.shape == got.shape, f"{name}: shape {ref.shape} vs {got.shape}"
    diff = np.abs(ref - got)
    scale = np.abs(ref).max() + 1e-8
    max_abs = float(diff.max())
    max_rel = max_abs / scale
    if not (max_abs < atol or max_rel < rtol):
        idx = np.unravel_index(diff.argmax(), diff.shape)
        raise AssertionError(
            f"{name}: max_abs={max_abs:.2e} (atol={atol:.0e}), "
            f"max_rel={max_rel:.2e} (rtol={rtol:.0e}), "
            f"worst at {idx}: ref={ref[idx]:.6f}, got={got[idx]:.6f}"
        )


# -- PyTorch helpers (only used in tests) --

def load_pt_model():
    from canvit import CanViTForPretrainingHFHub
    log.info("Loading PyTorch model from HF: %s", HF_REPO)
    return CanViTForPretrainingHFHub.from_pretrained(HF_REPO).eval()


def pt_viewpoint_full(batch_size: int = B):
    from canvit.viewpoint import Viewpoint
    return Viewpoint.full_scene(batch_size=batch_size, device=torch.device("cpu"))


def pt_viewpoint(centers: list[list[float]], scales: list[float]):
    from canvit.viewpoint import Viewpoint
    return Viewpoint(centers=torch.tensor(centers), scales=torch.tensor(scales))


def pt_forward(model, glimpse_pt: torch.Tensor, viewpoint_pt, state_pt) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        out = model(glimpse=glimpse_pt, state=state_pt, viewpoint=viewpoint_pt)
    return {
        "canvas": out.state.canvas.numpy(),
        "recurrent_cls": out.state.recurrent_cls.numpy(),
        "ephemeral_cls": out.ephemeral_cls.numpy(),
        "local_patches": out.local_patches.numpy(),
    }


def pt_recurrent_state(canvas: np.ndarray, recurrent_cls: np.ndarray):
    from canvit.model.base.impl import RecurrentState
    return RecurrentState(canvas=torch.tensor(canvas), recurrent_cls=torch.tensor(recurrent_cls))


def pt_sample(img_pt: torch.Tensor, viewpoint_pt, glimpse_px: int) -> np.ndarray:
    from canvit.viewpoint import sample_at_viewpoint
    out = sample_at_viewpoint(spatial=img_pt, viewpoint=viewpoint_pt, glimpse_size_px=glimpse_px)
    return out.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC


# -- Fixtures --

@pytest.fixture(scope="session")
def pt_model():
    return load_pt_model()


@pytest.fixture(scope="session")
def mlx_model():
    from canvit_mlx import load_canvit
    log.info("Loading MLX model from %s", WEIGHTS)
    return load_canvit(WEIGHTS)


@pytest.fixture(scope="session")
def glimpse_pair():
    """Returns (pt_glimpse [B,3,H,W], mlx_glimpse [B,H,W,3]) from same seed."""
    torch.manual_seed(SEED)
    pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
    mlx = mx.array(pt.numpy().transpose(0, 2, 3, 1))
    return pt, mlx
