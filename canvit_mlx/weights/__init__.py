"""Weight loading from safetensors (converted by convert.py).

The checkpoint is expected to have MLX-native key names — no remapping needed.
"""

import json
from pathlib import Path

import mlx.core as mx

from ..model import CanViT, CanViTConfig


def load_canvit(weights_path: str, config_path: str | None = None) -> CanViT:
    wp = Path(weights_path)
    cp = Path(config_path) if config_path else wp.with_suffix(".json")

    mc = json.loads(cp.read_text())["model_config"]
    model = CanViT(CanViTConfig(
        rw_stride=mc["rw_stride"],
        n_canvas_registers=mc["n_canvas_registers"],
        canvas_num_heads=mc["canvas_num_heads"],
        canvas_head_dim=mc["canvas_head_dim"],
        enable_vpe=mc["enable_vpe"],
        teacher_dim=mc["teacher_dim"],
    ))

    from safetensors import safe_open
    weights: list[tuple[str, mx.array]] = []
    with safe_open(str(wp), framework="numpy") as f:
        for key in f.keys():
            weights.append((key, mx.array(f.get_tensor(key))))
    model.load_weights(weights)
    return model
