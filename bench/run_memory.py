"""Memory benchmark: MLX peak vs MPS current_allocated (absolute).

MLX: mx.reset_peak_memory() → forward (inference) → mx.get_peak_memory().
     Peak of lazy-graph evaluation including model weights and transient intermediates.

MPS: current_allocated_memory() AFTER forward WITHOUT inference_mode.
     Absolute value — includes model weights + autograd-retained intermediates.
     No peak memory API on MPS, so this is the best available proxy.

Usage:
    uv run python -m bench.run_memory                     # default grids
    uv run python -m bench.run_memory --grids 8 16 32 64  # custom
    uv run python -m bench.run_memory --teacher-max-px 512
"""

import gc
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import polars as pl
import torch

from .shared import (
    B, BENCH_DIR, DINOV3_REPO, DINOV3S_REPO, GLIMPSE_PX, HF_REPO,
    IMAGE_PATH, WEIGHTS, git_sha,
)

log = logging.getLogger(__name__)

MiB = 1024 * 1024


def main(grids: tuple[int, ...] = (8, 16, 32, 64, 128),
         teacher_max_px: int = 1024,
         out_dir: str | None = None) -> None:
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint as pt_sample
    from canvit.viewpoint import Viewpoint as PTVP

    from canvit_mlx import Viewpoint as MLXVP, load_canvit
    from canvit_mlx.glimpse import extract_glimpse_at_viewpoint as mlx_extract
    from canvit_mlx.preprocess import load_and_preprocess

    device = torch.device("mps")

    ts = datetime.now(timezone.utc)
    run_dir = Path(out_dir) if out_dir else BENCH_DIR / ts.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(exist_ok=True)
    log.info("Output: %s", run_dir)

    # ── Load models ──
    log.info("Loading models...")
    pt_m = CanViTForPretrainingHFHub.from_pretrained(HF_REPO).to(device).eval()
    mlx_m = load_canvit(WEIGHTS)
    mlx_m_bf16 = load_canvit(WEIGHTS)
    mlx_m_bf16.apply(lambda p: p.astype(mx.bfloat16))
    cfg = mlx_m.cfg

    d3b = AutoModel.from_pretrained(DINOV3_REPO, dtype=torch.float32).to(device).eval()
    d3s = AutoModel.from_pretrained(DINOV3S_REPO, dtype=torch.float32).to(device).eval()
    pil_image = Image.open(IMAGE_PATH)
    log.info("Models loaded. Grids: %s, teacher_max_px: %d", grids, teacher_max_px)

    # ── Warmup all models once ──
    log.info("Warming up...")
    with torch.inference_mode():
        img_pt = torch.from_numpy(
            np.array(load_and_preprocess(str(IMAGE_PATH), target_size=128)).transpose(0, 3, 1, 2)
        ).to(device)
        vp_pt = PTVP.full_scene(batch_size=B, device=device)
        g_pt = pt_sample(spatial=img_pt, viewpoint=vp_pt, glimpse_size_px=GLIMPSE_PX)
        _ = pt_m(glimpse=g_pt, state=pt_m.init_state(batch_size=B, canvas_grid_size=8), viewpoint=vp_pt)
        torch.mps.synchronize()

        proc_b = AutoImageProcessor.from_pretrained(DINOV3_REPO, size={"height": 128, "width": 128})
        proc_s = AutoImageProcessor.from_pretrained(DINOV3S_REPO, size={"height": 128, "width": 128})
        _ = d3b(proc_b(images=pil_image, return_tensors="pt")["pixel_values"].to(device))
        _ = d3s(proc_s(images=pil_image, return_tensors="pt")["pixel_values"].to(device))
        torch.mps.synchronize()

    img_mlx = load_and_preprocess(str(IMAGE_PATH), target_size=128)
    vp_mlx = MLXVP.full_scene(batch_size=B)
    o = mlx_m(mlx_extract(img_mlx, vp_mlx, GLIMPSE_PX),
              mlx_m.init_state(batch_size=B, canvas_grid_size=8), vp_mlx)
    mx.eval(o.state.canvas)

    mlx_ref: list = [None]
    def mlx_sync():
        if mlx_ref[0] is not None:
            mx.eval(mlx_ref[0])

    # ── Measure ──
    records = []
    for grid in grids:
        image_px = max(grid * cfg.patch_size, GLIMPSE_PX)
        log.info("=== grid=%d, image=%dpx ===", grid, image_px)

        # gc between resolutions to start clean
        gc.collect(); torch.mps.synchronize()

        # Prepare inputs for this resolution
        image_mlx = load_and_preprocess(str(IMAGE_PATH), target_size=image_px)
        mx.eval(image_mlx)
        image_bf = image_mlx.astype(mx.bfloat16)
        mx.eval(image_bf)
        image_pt = torch.from_numpy(np.array(image_mlx).transpose(0, 3, 1, 2)).to(device)
        vp_pt = PTVP.full_scene(batch_size=B, device=device)
        vp_mlx = MLXVP.full_scene(batch_size=B)
        vp_bf = MLXVP(centers=mx.zeros((B, 2), dtype=mx.bfloat16),
                       scales=mx.ones((B,), dtype=mx.bfloat16))
        torch.mps.synchronize()

        rec = {"grid": grid, "image_px": image_px}

        # ── CanViT MLX fp32 ──
        def canvit_mlx_fp32():
            g = mlx_extract(image_mlx, vp_mlx, GLIMPSE_PX)
            o = mlx_m(g, mlx_m.init_state(batch_size=B, canvas_grid_size=grid), vp_mlx)
            mx.eval(o.state.recurrent_cls, o.ephemeral_cls, o.local_patches)
            mlx_ref[0] = o.state.canvas
            return mlx_ref[0]

        canvit_mlx_fp32(); mlx_sync()  # warmup this grid size
        mx.reset_peak_memory()
        canvit_mlx_fp32(); mlx_sync()
        rec["canvit_mlx_fp32_peak"] = mx.get_peak_memory()
        log.info("  CanViT MLX fp32 peak: %.1f MiB", rec["canvit_mlx_fp32_peak"] / MiB)

        # ── CanViT MLX bf16 ──
        def canvit_mlx_bf16():
            g = mlx_extract(image_bf, vp_bf, GLIMPSE_PX)
            o = mlx_m_bf16(g, mlx_m_bf16.init_state(batch_size=B, canvas_grid_size=grid), vp_bf)
            mx.eval(o.state.recurrent_cls, o.ephemeral_cls, o.local_patches)
            mlx_ref[0] = o.state.canvas
            return mlx_ref[0]

        canvit_mlx_bf16(); mlx_sync()
        mx.reset_peak_memory()
        canvit_mlx_bf16(); mlx_sync()
        rec["canvit_mlx_bf16_peak"] = mx.get_peak_memory()
        log.info("  CanViT MLX bf16 peak: %.1f MiB", rec["canvit_mlx_bf16_peak"] / MiB)

        # ── CanViT PT MPS (no inference_mode → autograd retains intermediates) ──
        def canvit_pt():
            g = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=GLIMPSE_PX)
            return pt_m(glimpse=g, state=pt_m.init_state(batch_size=B, canvas_grid_size=grid),
                        viewpoint=vp_pt).state.canvas

        with torch.inference_mode():
            canvit_pt(); torch.mps.synchronize()  # warmup (inference_mode ok here)
        gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

        out = canvit_pt(); torch.mps.synchronize()
        rec["canvit_pt_mps"] = torch.mps.current_allocated_memory()
        del out
        log.info("  CanViT PT MPS: %.1f MiB", rec["canvit_pt_mps"] / MiB)
        gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

        # ── Teachers (no inference_mode → autograd retains intermediates) ──
        if image_px <= teacher_max_px:
            for name, model, repo in [("dinov3", d3b, DINOV3_REPO),
                                      ("dinov3s", d3s, DINOV3S_REPO)]:
                proc = AutoImageProcessor.from_pretrained(
                    repo, size={"height": image_px, "width": image_px})
                inp = proc(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
                torch.mps.synchronize()

                with torch.inference_mode():
                    _ = model(inp).last_hidden_state; torch.mps.synchronize()  # warmup
                del _
                gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

                out = model(inp).last_hidden_state; torch.mps.synchronize()
                rec[f"{name}_mps"] = torch.mps.current_allocated_memory()
                del out, inp
                log.info("  %s MPS: %.1f MiB", name, rec[f"{name}_mps"] / MiB)
                gc.collect(); torch.mps.empty_cache(); torch.mps.synchronize()

        records.append(rec)

    # ── Save ──
    df = pl.DataFrame(records)
    meta = {"timestamp": ts.isoformat(), "git_sha": git_sha(),
            "torch_version": torch.__version__, "mlx_version": mx.__version__,
            "grids": list(grids), "teacher_max_px": teacher_max_px}

    pq_path = run_dir / "memory.parquet"
    df.write_parquet(pq_path)
    (run_dir / "memory_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Saved %s (%d rows)", pq_path, len(records))
    print(f"\n→ {run_dir}/memory.parquet")
    print(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import tyro
    tyro.cli(main)
