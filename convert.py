"""Convert CanViT from HuggingFace Hub to MLX-compatible safetensors.

Usage:
    uv run python convert.py
    uv run python convert.py --repo canvit/canvit-vitb16-pretrain-512px-in21k --out weights.safetensors
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


@dataclass
class Args:
    repo: str = HF_REPO
    out: Path = Path("weights.safetensors")


def convert(args: Args) -> None:
    from canvit import CanViTForPretrainingHFHub

    log.info("Loading from HuggingFace: %s", args.repo)
    model = CanViTForPretrainingHFHub.from_pretrained(args.repo)
    sd = model.state_dict()

    log.info("State dict: %d keys", len(sd))

    out: dict[str, torch.Tensor] = {}
    for key, val in sd.items():
        val = val.to(torch.float32).contiguous()

        # MLX Conv2d expects [out_channels, kH, kW, in_channels]
        if key == "backbone.vit.patch_embed.proj.weight":
            val = val.permute(0, 2, 3, 1).contiguous()

        out[key] = val

    save_file(out, str(args.out))
    total = sum(v.numel() for v in out.values()) / 1e6
    log.info("Saved %d tensors (%.1fM params) to %s", len(out), total, args.out)

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
