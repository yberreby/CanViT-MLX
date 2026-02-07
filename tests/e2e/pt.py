"""Run PT reference trajectory."""

import numpy as np
import torch

from conftest import pt_forward, pt_viewpoint, CANVAS_GRID, GLIMPSE_PX
from . import TRAJECTORY


def run(model, image: torch.Tensor) -> list[dict[str, np.ndarray]]:
    from canvit.model.base.impl import RecurrentState as PtState
    from canvit.viewpoint import sample_at_viewpoint as pt_sample

    state = model.init_state(batch_size=1, canvas_grid_size=CANVAS_GRID)
    results: list[dict[str, np.ndarray]] = []
    for cy, cx, s in TRAJECTORY:
        vp = pt_viewpoint([[cy, cx]], [s])
        glimpse = pt_sample(spatial=image, viewpoint=vp, glimpse_size_px=GLIMPSE_PX)
        ref = pt_forward(model, glimpse, vp, state)
        results.append(ref)
        state = PtState(canvas=torch.tensor(ref["canvas"]),
                        recurrent_cls=torch.tensor(ref["recurrent_cls"]))
    return results
