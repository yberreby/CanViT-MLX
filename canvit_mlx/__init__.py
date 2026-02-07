"""CanViT inference on Apple Silicon via MLX."""

from .coords import Viewpoint, grid_coords, sample_at_viewpoint
from .model import CanViT, CanViTConfig, CanViTOutput, RecurrentState
from .rope import compute_rope, make_rope_periods
from .weights import load_canvit

__all__ = [
    "CanViT", "CanViTConfig", "CanViTOutput", "RecurrentState",
    "Viewpoint", "grid_coords", "sample_at_viewpoint",
    "compute_rope", "make_rope_periods",
    "load_canvit",
]
