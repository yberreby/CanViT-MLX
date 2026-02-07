"""Weight loading from safetensors (converted by convert.py)."""

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
    raw: dict[str, mx.array] = {}
    with safe_open(str(wp), framework="numpy") as f:
        for key in f.keys():
            raw[key] = mx.array(f.get_tensor(key))

    weights = {k: v for key, v in raw.items() if (k := _map_key(key)) is not None}
    weights["rope_periods_backbone"] = raw["backbone.vit.rope_embed.periods"]
    model.load_weights(list(weights.items()))
    return model


def _map_key(pt_key: str) -> str | None:
    if any(pt_key.startswith(p) for p in ("backbone.vit.rope_embed.", "backbone.vit.mask_token")):
        return None
    if pt_key.endswith("._initialized") or "qkv.bias_mask" in pt_key:
        return None

    k = pt_key
    k = k.replace("backbone.vit.patch_embed.proj.", "patch_embed.proj.")
    k = k.replace("backbone.vit.norm.", "backbone_norm.")
    k = k.replace("backbone.vit.blocks.", "blocks.")
    k = k.replace("vpe_encoder.B", "vpe_encoder.B_mat")
    k = k.replace("scene_patches_head.0.", "scene_patches_ln.")
    k = k.replace("scene_patches_head.1.", "scene_patches_proj.")
    k = k.replace("scene_cls_head.0.", "scene_cls_ln.")
    k = k.replace("scene_cls_head.1.", "scene_cls_proj.")

    for old, new in [
        ("backbone.vit.cls_token", "cls_token"),
        ("backbone.vit.storage_tokens", "storage_tokens"),
    ]:
        if k == old: return new

    for prefix, target in [
        ("cls_standardizers.32.mean", "cls_std_mean"),
        ("cls_standardizers.32.var", "cls_std_var"),
        ("scene_standardizers.32.mean", "scene_std_mean"),
        ("scene_standardizers.32.var", "scene_std_var"),
    ]:
        if k.startswith(prefix): return target

    return k
