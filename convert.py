"""Convert CanViT HF Hub weights to MLX-native safetensors.

Performs: key remapping (PT → MLX names), Conv2d weight permutation, filtering.
The output checkpoint has keys matching canvit_mlx nn.Module parameter names;
load_canvit() loads them directly with zero key surgery.

Usage:
    uv run python convert.py
    uv run python convert.py --out /path/to/canvit-vitb16.safetensors
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import tyro
from safetensors.torch import save_file

log = logging.getLogger(__name__)

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
DEFAULT_OUT = Path("weights/canvit-vitb16.safetensors")


@dataclass
class Args:
    repo: str = HF_REPO
    out: Path = DEFAULT_OUT


def _map_key(pt_key: str) -> str | None:
    """Map a PyTorch state_dict key to the corresponding canvit_mlx parameter name.

    Returns None for keys that should be dropped (inference-irrelevant).
    """
    if pt_key.endswith("._initialized") or "qkv.bias_mask" in pt_key:
        return None
    if pt_key == "backbone.vit.mask_token":
        return None
    # rope_embed: only keep periods, renamed
    if pt_key.startswith("backbone.vit.rope_embed."):
        if pt_key == "backbone.vit.rope_embed.periods":
            return "rope_periods_backbone"
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
        if k == old:
            return new

    for prefix, target in [
        ("cls_standardizers.32.mean", "cls_std_mean"),
        ("cls_standardizers.32.var", "cls_std_var"),
        ("scene_standardizers.32.mean", "scene_std_mean"),
        ("scene_standardizers.32.var", "scene_std_var"),
    ]:
        if k.startswith(prefix):
            return target

    return k


def convert(args: Args) -> None:
    from canvit import CanViTForPretrainingHFHub

    log.info("Loading from HuggingFace: %s", args.repo)
    model = CanViTForPretrainingHFHub.from_pretrained(args.repo)
    sd = model.state_dict()
    log.info("State dict: %d keys", len(sd))

    out: dict[str, torch.Tensor] = {}
    skipped = 0
    for pt_key, val in sd.items():
        mlx_key = _map_key(pt_key)
        if mlx_key is None:
            skipped += 1
            continue
        val = val.to(torch.float32).contiguous()
        if pt_key == "backbone.vit.patch_embed.proj.weight":
            val = val.permute(0, 2, 3, 1).contiguous()
        out[mlx_key] = val

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(args.out))
    total = sum(v.numel() for v in out.values()) / 1e6
    log.info("Saved %d tensors (%.1fM params), skipped %d → %s", len(out), total, skipped, args.out)

    meta = {
        "hf_repo": args.repo,
        "model_config": {
            k: getattr(model.cfg, k)
            for k in ["rw_stride", "n_canvas_registers", "canvas_num_heads",
                       "canvas_head_dim", "canvas_read_layer_scale_init", "enable_vpe", "teacher_dim"]
        },
        "backbone_name": model.backbone_name,
        "grid_sizes": model.grid_sizes,
    }
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata to %s", meta_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert(tyro.cli(Args))
