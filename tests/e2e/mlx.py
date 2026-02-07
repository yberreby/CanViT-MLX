"""Run MLX trajectory."""

import mlx.core as mx
import numpy as np

from canvit_mlx import RecurrentState, Viewpoint, sample_at_viewpoint
from conftest import CANVAS_GRID, GLIMPSE_PX
from . import OUTPUTS, TRAJECTORY


def run(model, image: mx.array) -> list[dict[str, np.ndarray]]:
    state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
    results: list[dict[str, np.ndarray]] = []
    for cy, cx, s in TRAJECTORY:
        vp = Viewpoint(centers=mx.array([[cy, cx]]), scales=mx.array([s]))
        glimpse = sample_at_viewpoint(image, vp, GLIMPSE_PX)
        out = model(glimpse, state, vp)
        mx.eval(out.state.canvas, out.state.recurrent_cls, out.ephemeral_cls, out.local_patches)
        results.append({k: np.array(getattr(out.state, k) if k in ("canvas", "recurrent_cls")
                                    else getattr(out, k)) for k in OUTPUTS})
        state = RecurrentState(canvas=out.state.canvas, recurrent_cls=out.state.recurrent_cls)
    return results
