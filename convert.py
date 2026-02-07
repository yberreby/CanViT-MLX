"""Convert CanViT HF Hub weights → MLX-native safetensors.

Performs key remapping (PT → MLX names), Conv2d weight permutation (OIHW→OHWI),
filtering of inference-irrelevant tensors. Verifies the result by comparing a
single forward pass against the PyTorch reference.

Usage:
    uv run python convert.py
    uv run python convert.py --out /path/to/canvit-vitb16.safetensors
    uv run python convert.py --no-verify
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tyro
from safetensors.torch import save_file

log = logging.getLogger(__name__)

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
DEFAULT_OUT = Path("weights/canvit-vitb16.safetensors")

VERIFY_SEED = 42
VERIFY_GLIMPSE_PX = 128
VERIFY_ATOL = 5.0   # f32 SDPA accumulation over ~1040 tokens
VERIFY_RTOL = 2e-3


@dataclass
class Args:
    repo: str = HF_REPO
    out: Path = DEFAULT_OUT
    verify: bool = True


# ---------------------------------------------------------------------------
# Key mapping
# ---------------------------------------------------------------------------

def _make_key_mapper(grid_size: int):
    """Build PT→MLX key mapping function. Returns None for inference-irrelevant keys."""
    gs = str(grid_size)

    def map_key(pt_key: str) -> str | None:
        if pt_key.endswith("._initialized") or "qkv.bias_mask" in pt_key:
            return None
        if pt_key == "backbone.vit.mask_token":
            return None
        if pt_key.startswith("backbone.vit.rope_embed.") or pt_key.startswith("backbone.vit.norm."):
            return None

        k = pt_key
        k = k.replace("backbone.vit.patch_embed.proj.", "patch_embed.proj.")
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

        for suffix, target in [
            ("mean", "cls_std_mean"), ("var", "cls_std_var"),
        ]:
            if k == f"cls_standardizers.{gs}.{suffix}":
                return target

        for suffix, target in [
            ("mean", "scene_std_mean"), ("var", "scene_std_var"),
        ]:
            if k == f"scene_standardizers.{gs}.{suffix}":
                return target

        return k

    return map_key


# ---------------------------------------------------------------------------
# Weight remapping
# ---------------------------------------------------------------------------

def _remap_state_dict(sd: dict[str, torch.Tensor], map_key) -> dict[str, torch.Tensor]:
    """Remap PT state_dict → MLX-native keys and weight layouts."""
    out: dict[str, torch.Tensor] = {}
    skipped = 0
    for pt_key, val in sd.items():
        mlx_key = map_key(pt_key)
        if mlx_key is None:
            log.info("  skip  %-55s  %s", pt_key, tuple(val.shape))
            skipped += 1
            continue
        val = val.to(torch.float32).contiguous()
        if pt_key == "backbone.vit.patch_embed.proj.weight":
            val = val.permute(0, 2, 3, 1).contiguous()
            log.info("  conv  %-55s → %-35s %s [OIHW->OHWI]", pt_key, mlx_key, tuple(val.shape))
        else:
            log.info("  map   %-55s → %-35s %s", pt_key, mlx_key, tuple(val.shape))
        out[mlx_key] = val

    total_params = sum(v.numel() for v in out.values())
    log.info("Remapped: %d tensors (%.1fM params), skipped: %d", len(out), total_params / 1e6, skipped)
    return out


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def _extract_config(model) -> dict:
    """Build model_config dict from PT model attributes."""
    bb = model.backbone
    return {
        "embed_dim": bb.embed_dim,
        "num_heads": bb.vit.num_heads,
        "n_blocks": bb.n_blocks,
        "patch_size": bb.vit.patch_size,
        "ffn_ratio": bb.ffn_ratio,
        "n_register_tokens": bb.n_register_tokens,
        "rw_stride": model.cfg.rw_stride,
        "n_canvas_registers": model.cfg.n_canvas_registers,
        "canvas_num_heads": model.cfg.canvas_num_heads,
        "canvas_head_dim": model.cfg.canvas_head_dim,
        "enable_vpe": model.cfg.enable_vpe,
        "teacher_dim": model.cfg.teacher_dim,
        "std_grid_size": model.grid_sizes[0],
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def _verify(pt_model, weights_path: str, grid_size: int) -> None:
    """Load converted weights, run one forward pass, compare against PT reference."""
    import mlx.core as mx
    from canvit.viewpoint import Viewpoint as PtViewpoint
    from canvit_mlx import Viewpoint as MlxViewpoint, load_canvit

    log.info("--- Verification: full-scene forward pass ---")
    log.info("  seed=%d  glimpse=%dpx  canvas_grid=%d  atol=%.1f  rtol=%.0e",
             VERIFY_SEED, VERIFY_GLIMPSE_PX, grid_size, VERIFY_ATOL, VERIFY_RTOL)

    torch.manual_seed(VERIFY_SEED)
    glimpse_pt = torch.randn(1, 3, VERIFY_GLIMPSE_PX, VERIFY_GLIMPSE_PX)
    glimpse_mlx = mx.array(glimpse_pt.numpy().transpose(0, 2, 3, 1))

    vp_pt = PtViewpoint.full_scene(batch_size=1, device=torch.device("cpu"))
    state_pt = pt_model.init_state(batch_size=1, canvas_grid_size=grid_size)
    with torch.inference_mode():
        out_pt = pt_model(glimpse=glimpse_pt, state=state_pt, viewpoint=vp_pt)

    mlx_model = load_canvit(weights_path)
    vp_mlx = MlxViewpoint.full_scene(batch_size=1)
    state_mlx = mlx_model.init_state(batch_size=1, canvas_grid_size=grid_size)
    out_mlx = mlx_model(glimpse_mlx, state_mlx, vp_mlx)
    mx.eval(out_mlx.state.canvas, out_mlx.state.recurrent_cls,
            out_mlx.ephemeral_cls, out_mlx.local_patches)

    pairs = [
        ("canvas", out_pt.state.canvas.numpy(), np.array(out_mlx.state.canvas)),
        ("recurrent_cls", out_pt.state.recurrent_cls.numpy(), np.array(out_mlx.state.recurrent_cls)),
        ("ephemeral_cls", out_pt.ephemeral_cls.numpy(), np.array(out_mlx.ephemeral_cls)),
        ("local_patches", out_pt.local_patches.numpy(), np.array(out_mlx.local_patches)),
    ]
    all_ok = True
    for name, ref, got in pairs:
        assert ref.shape == got.shape, f"{name}: shape {ref.shape} vs {got.shape}"
        max_abs = float(np.abs(ref - got).max())
        scale = float(np.abs(ref).max()) + 1e-8
        max_rel = max_abs / scale
        ok = max_abs < VERIFY_ATOL or max_rel < VERIFY_RTOL
        log.info("  %-16s  max_abs=%.2e  max_rel=%.2e  scale=%.1f  [%s]",
                 name, max_abs, max_rel, scale, "OK" if ok else "FAIL")
        if not ok:
            all_ok = False

    assert all_ok, "Verification FAILED: MLX outputs do not match PyTorch reference"
    log.info("Verification passed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(args: Args) -> None:
    from canvit import CanViTForPretrainingHFHub

    log.info("=== CanViT conversion: %s → %s ===", args.repo, args.out)

    pt_model = CanViTForPretrainingHFHub.from_pretrained(args.repo)
    sd = pt_model.state_dict()
    log.info("Loaded PT model: %d state_dict keys", len(sd))

    assert len(pt_model.grid_sizes) == 1, f"expected single grid size, got {pt_model.grid_sizes}"
    grid_size = pt_model.grid_sizes[0]
    log.info("Grid size: %d", grid_size)

    weights = _remap_state_dict(sd, _make_key_mapper(grid_size))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(weights, str(args.out))
    log.info("Saved weights → %s", args.out)

    config = _extract_config(pt_model)
    meta = {"hf_repo": args.repo, "model_config": config, "backbone_name": pt_model.backbone_name}
    meta_path = args.out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("Saved metadata → %s", meta_path)
    for k, v in config.items():
        log.info("  %-20s %s", k, v)

    if args.verify:
        _verify(pt_model, str(args.out), grid_size)
    else:
        log.warning("Skipping verification (--no-verify)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    convert(tyro.cli(Args))
