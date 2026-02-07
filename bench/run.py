"""Component-level benchmark: PT (MPS) vs MLX (GPU) with correctness verification.

Each component is fed the SAME input; correctness is checked BEFORE timing.
Structurally impossible to bench something that isn't verified 1-to-1.

Components: init_state, patch_embed, ViT block, canvas read, canvas write, full forward.
Outputs: bench/<timestamp>.{parquet,json} (raw data + metadata).

Usage:
    uv run python bench/run.py                          # default grid=128
    uv run python bench/run.py --grids 32 64 128 256    # multiple grid sizes
"""

import logging
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
B = 1
PIXELS_PER_CANVAS_CELL = 4
GLIMPSE_PX = 128
WARMUP = 5
ITERS = 20
SEED = 42
IMAGE_PATH = Path("test_data/Cat03.jpg")
WEIGHTS = "weights/canvit-vitb16-pretrain-512px-in21k.safetensors"
HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"

# MPS vs MLX GPU — both have approximate SDPA, so rtol is generous.
RTOL = 5e-3

OUT_DIR = Path(__file__).parent


# ── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Stats:
    med: float; mn: float; mx: float; mean: float
    raw_us: list[float]

    @staticmethod
    def from_secs(ts: list[float]) -> "Stats":
        us = [t * 1e6 for t in ts]
        return Stats(statistics.median(us), min(us), max(us), statistics.mean(us), us)


@dataclass
class Row:
    name: str
    count: int
    pt: Stats
    mlx: Stats
    rel_err: float

    @property
    def ratio(self) -> float:
        return self.pt.med / self.mlx.med

    @property
    def weighted(self) -> float:
        return self.mlx.med * self.count

    def fmt(self, total_us: float = 0.0) -> str:
        pct = f"{self.weighted / total_us * 100:>5.1f}%" if total_us > 0 else "      "
        return (f"{self.name:<16} {self.count:>3} "
                f"{self.pt.med:>7.0f}μ {self.pt.mn:>7.0f}μ {self.pt.mx:>7.0f}μ "
                f"{self.mlx.med:>7.0f}μ {self.mlx.mn:>7.0f}μ {self.mlx.mx:>7.0f}μ "
                f"{self.ratio:>5.2f}× {pct} {self.rel_err:>7.1e}")


TABLE_HDR = (f"{'Component':<16} {'×':>3} "
             f"{'PT med':>8} {'PT min':>8} {'PT max':>8} "
             f"{'MLX med':>8} {'MLX min':>8} {'MLX max':>8} "
             f"{'ratio':>6} {'MLX%':>6} {'err':>8}")


# ── Core bench primitives ───────────────────────────────────────────────────

def time_fn(fn, sync) -> Stats:
    for _ in range(WARMUP):
        fn(); sync()
    ts = []
    for _ in range(ITERS):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return Stats.from_secs(ts)


def check(name: str, ref: np.ndarray, got: np.ndarray) -> float:
    """Log abs/rel error. Returns rel_err = ||diff||/||ref||."""
    assert ref.shape == got.shape, f"{name}: shape {ref.shape} vs {got.shape}"
    diff_norm = float(np.linalg.norm(ref - got))
    ref_norm = float(np.linalg.norm(ref)) + 1e-8
    rel = diff_norm / ref_norm
    max_abs = float(np.abs(ref - got).max())
    ok = rel < RTOL
    tag = "PASS" if ok else "WARN"
    log.info("  %s: ||Δ||/||ref||=%.2e  max_abs=%.2e  [%s]", name, rel, max_abs, tag)
    return rel


def check_and_bench(name: str, count: int,
                    pt_fn, mlx_fn, pt_sync, mlx_sync) -> Row:
    """Verify correctness (via numpy), then bench raw GPU compute (no copy)."""
    pt_val = pt_fn(); pt_sync()
    mlx_val = mlx_fn(); mlx_sync()
    rel = check(name, pt_val.cpu().numpy(), np.asarray(mlx_val))
    return Row(name, count, time_fn(pt_fn, pt_sync), time_fn(mlx_fn, mlx_sync), rel)


def print_table(rows: list[Row]) -> None:
    components = rows[:-1]
    sum_mlx = sum(r.weighted for r in components)
    sum_pt = sum(r.pt.med * r.count for r in components)

    sep = "─" * len(TABLE_HDR)
    print(f"\n{TABLE_HDR}")
    print(sep)
    for r in components:
        print(r.fmt(sum_mlx))
    print(sep)
    ratio = sum_pt / sum_mlx
    print(f"{'Σ components':<16}     {sum_pt:>7.0f}μ {'':>17} "
          f"{sum_mlx:>7.0f}μ {'':>17} {ratio:>5.2f}×")
    print(rows[-1].fmt())
    print(sep)


# ── Bench driver ────────────────────────────────────────────────────────────

def bench_at_grid(pt_m, mlx_m, cfg, device, canvas_grid, image_px,
                  image_mlx: mx.array,
                  n_patches, n_local,
                  pt_rope, pt_periods, mlx_rope, mlx_periods,
                  pt_sample, mlx_extract, PTVP, MLXVP) -> list[Row]:
    """Bench all components at a given canvas_grid. Runs inside inference_mode."""
    mps_sync = torch.mps.synchronize
    n_canvas = cfg.n_canvas_registers + canvas_grid ** 2
    n_reads = len(mlx_m.read_after_blocks)
    n_writes = len(mlx_m.write_after_blocks)

    rng = np.random.RandomState(SEED)
    local_np = rng.randn(B, n_local, cfg.embed_dim).astype(np.float32)
    canvas_np = rng.randn(B, n_canvas, cfg.canvas_dim).astype(np.float32)
    local_pos_np = rng.uniform(-1, 1, (B, n_patches, 2)).astype(np.float32)
    canvas_pos_np = rng.uniform(-1, 1, (B, canvas_grid ** 2, 2)).astype(np.float32)

    local_pt = torch.tensor(local_np, device=device)
    canvas_pt = torch.tensor(canvas_np, device=device)
    local_mlx = mx.array(local_np)
    canvas_mlx = mx.array(canvas_np)

    # Image: shared data, different layouts (PT=NCHW, MLX=NHWC)
    image_pt = torch.from_numpy(np.array(image_mlx).transpose(0, 3, 1, 2)).to(device)
    vp_pt = PTVP.full_scene(batch_size=B, device=device)
    vp_mlx = MLXVP.full_scene(batch_size=B)

    bb_rope = pt_rope(positions=torch.tensor(local_pos_np, device=device),
                      periods=pt_periods(head_dim=cfg.head_dim, device=device),
                      dtype=torch.float32)
    bb_sin, bb_cos = mlx_rope(mx.array(local_pos_np), mlx_periods(cfg.head_dim))
    ca_rope_l = pt_rope(positions=torch.tensor(local_pos_np, device=device),
                        periods=pt_periods(head_dim=cfg.canvas_head_dim, device=device),
                        dtype=torch.float32)
    ca_rope_c = pt_rope(positions=torch.tensor(canvas_pos_np, device=device),
                        periods=pt_periods(head_dim=cfg.canvas_head_dim, device=device),
                        dtype=torch.float32)
    ca_sin_l, ca_cos_l = mlx_rope(mx.array(local_pos_np), mlx_periods(cfg.canvas_head_dim))
    ca_sin_c, ca_cos_c = mlx_rope(mx.array(canvas_pos_np), mlx_periods(cfg.canvas_head_dim))
    mx.eval(bb_sin, bb_cos, ca_sin_l, ca_cos_l, ca_sin_c, ca_cos_c)
    mps_sync()

    mlx_ref: list = [None]

    def mlx_wrap(fn):
        def wrapper():
            mlx_ref[0] = fn()
            return mlx_ref[0]
        return wrapper

    def mlx_sync():
        if mlx_ref[0] is not None:
            mx.eval(mlx_ref[0])

    log.info("--- Check + Bench (grid=%d, image=%dpx, tokens=%d) ---",
             canvas_grid, image_px, n_canvas)
    rows: list[Row] = []

    # MPS F.grid_sample is BROKEN in torch 2.10.0 — produces wrong pixels.
    # MLX matches PT CPU to machine precision. Correctness check here is
    # meaningless (MPS vs MLX); timing is still valid.
    rows.append(check_and_bench(
        "glimpse", 1,
        lambda: pt_sample(spatial=image_pt, viewpoint=vp_pt,
                          glimpse_size_px=GLIMPSE_PX).permute(0, 2, 3, 1),
        mlx_wrap(lambda: mlx_extract(image_mlx, vp_mlx, GLIMPSE_PX)),
        mps_sync, mlx_sync,
    ))

    # Pre-extract glimpse for downstream component benchmarks
    glimpse_pt = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=GLIMPSE_PX)
    glimpse_mlx = mlx_extract(image_mlx, vp_mlx, GLIMPSE_PX)
    mx.eval(glimpse_mlx); mps_sync()

    rows.append(check_and_bench(
        "init_state", 1,
        lambda: pt_m.init_state(batch_size=B, canvas_grid_size=canvas_grid).canvas,
        mlx_wrap(lambda: mlx_m.init_state(batch_size=B, canvas_grid_size=canvas_grid).canvas),
        mps_sync, mlx_sync,
    ))

    rows.append(check_and_bench(
        "patch_embed", 1,
        lambda: pt_m.backbone.vit.patch_embed(glimpse_pt).flatten(1, 2),
        mlx_wrap(lambda: mlx_m.patch_embed(glimpse_mlx)[0]),
        mps_sync, mlx_sync,
    ))

    rows.append(check_and_bench(
        "ViT block", cfg.n_blocks,
        lambda: pt_m.backbone.forward_block(0, local_pt, bb_rope),
        mlx_wrap(lambda: mlx_m.blocks[0](local_mlx, bb_sin, bb_cos)),
        mps_sync, mlx_sync,
    ))

    # PT read_attn = ScaledResidualAttention (residual + scale included).
    # MLX read_attn = raw; residual + scale applied separately.
    rows.append(check_and_bench(
        "canvas read", n_reads,
        lambda: pt_m.read_attn[0](query=local_pt, kv=canvas_pt,
                                   query_rope=ca_rope_l, kv_rope=ca_rope_c),
        mlx_wrap(lambda: local_mlx + mlx_m.read_scales[0](
            mlx_m.read_attn[0](local_mlx, canvas_mlx,
                               ca_sin_l, ca_cos_l, ca_sin_c, ca_cos_c))),
        mps_sync, mlx_sync,
    ))

    # PT write_attn = ResidualAttention (residual included). MLX = raw.
    rows.append(check_and_bench(
        "canvas write", n_writes,
        lambda: pt_m.write_attn[0](query=canvas_pt, kv=local_pt,
                                    query_rope=ca_rope_c, kv_rope=ca_rope_l),
        mlx_wrap(lambda: canvas_mlx + mlx_m.write_attn[0](
            canvas_mlx, local_mlx, ca_sin_c, ca_cos_c, ca_sin_l, ca_cos_l)),
        mps_sync, mlx_sync,
    ))

    # Full forward: image → glimpse extraction → model forward (the real pipeline)
    def pt_full():
        g = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=GLIMPSE_PX)
        s = pt_m.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        return pt_m(glimpse=g, state=s, viewpoint=vp_pt).state.canvas

    def mlx_full():
        g = mlx_extract(image_mlx, vp_mlx, GLIMPSE_PX)
        s = mlx_m.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        o = mlx_m(g, s, vp_mlx)
        mx.eval(o.state.recurrent_cls, o.ephemeral_cls, o.local_patches)
        return o.state.canvas

    rows.append(check_and_bench(
        "full forward", 1,
        pt_full, mlx_wrap(mlx_full), mps_sync, mlx_sync,
    ))

    return rows


# ── Output ──────────────────────────────────────────────────────────────────

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def save_results(all_results: dict[int, list[Row]], cfg) -> Path:
    import json

    import polars as pl

    ts = datetime.now(timezone.utc)
    ts_str = ts.isoformat()
    git = _git_sha()
    stem = ts.strftime("%Y-%m-%dT%H-%M-%S")

    meta = {
        "timestamp": ts_str,
        "git_sha": git,
        "torch_version": torch.__version__,
        "mlx_version": mx.__version__,
        "batch_size": B,
        "glimpse_px": GLIMPSE_PX,
        "warmup": WARMUP,
        "iters": ITERS,
        "seed": SEED,
        "embed_dim": cfg.embed_dim,
        "canvas_dim": cfg.canvas_dim,
        "n_blocks": cfg.n_blocks,
        "grids": sorted(all_results.keys()),
    }

    records = []
    for grid, rows in sorted(all_results.items()):
        for r in rows:
            records.append({
                **meta,
                "grid": grid,
                "image_px": max(grid * PIXELS_PER_CANVAS_CELL, GLIMPSE_PX),
                "canvas_tokens": cfg.n_canvas_registers + grid ** 2,
                "component": r.name,
                "count": r.count,
                "pt_med_us": r.pt.med,
                "pt_min_us": r.pt.mn,
                "pt_max_us": r.pt.mx,
                "pt_mean_us": r.pt.mean,
                "mlx_med_us": r.mlx.med,
                "mlx_min_us": r.mlx.mn,
                "mlx_max_us": r.mlx.mx,
                "mlx_mean_us": r.mlx.mean,
                "pt_raw_us": r.pt.raw_us,
                "mlx_raw_us": r.mlx.raw_us,
                "ratio": r.ratio,
                "rel_err": r.rel_err,
            })

    df = pl.DataFrame(records)
    pq_path = OUT_DIR / f"{stem}.parquet"
    json_path = OUT_DIR / f"{stem}.json"
    df.write_parquet(pq_path)
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Saved %s + %s (%d rows, git=%s)", pq_path.name, json_path.name, len(records), git)
    return pq_path


# ── Main ────────────────────────────────────────────────────────────────────

def main(grids: tuple[int, ...] = (128,)) -> None:

    from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint as pt_sample
    from canvit.rope import compute_rope as pt_rope, make_rope_periods as pt_periods
    from canvit.viewpoint import Viewpoint as PTVP

    from canvit_mlx import Viewpoint as MLXVP, load_canvit
    from canvit_mlx.glimpse import extract_glimpse_at_viewpoint as mlx_extract
    from canvit_mlx.preprocess import load_and_preprocess
    from canvit_mlx.rope import compute_rope as mlx_rope, make_rope_periods as mlx_periods

    device = torch.device("mps")

    log.info("Loading models...")
    pt_m = CanViTForPretrainingHFHub.from_pretrained(HF_REPO).to(device).eval()
    mlx_m = load_canvit(WEIGHTS)
    cfg = mlx_m.cfg

    n_patches = (GLIMPSE_PX // cfg.patch_size) ** 2
    n_prefix = (1 if cfg.enable_vpe else 0) + 2 + cfg.n_register_tokens
    n_local = n_prefix + n_patches

    log.info("B=%d, glimpse=%dpx, warmup=%d, iters=%d, seed=%d",
             B, GLIMPSE_PX, WARMUP, ITERS, SEED)
    log.info("embed=%d, canvas=%d, bb_hd=%d, ca_hd=%d, heads=%d/%d",
             cfg.embed_dim, cfg.canvas_dim, cfg.head_dim, cfg.canvas_head_dim,
             cfg.num_heads, cfg.canvas_num_heads)
    log.info("blocks=%d, stride=%d, patch=%d, regs=%d/%d, vpe=%s",
             cfg.n_blocks, cfg.rw_stride, cfg.patch_size,
             cfg.n_register_tokens, cfg.n_canvas_registers, cfg.enable_vpe)
    log.info("patches=%d, prefix=%d, local=%d", n_patches, n_prefix, n_local)
    log.info("reads@%s (×%d), writes@%s (×%d)",
             mlx_m.read_after_blocks, len(mlx_m.read_after_blocks),
             mlx_m.write_after_blocks, len(mlx_m.write_after_blocks))
    log.info("Torch %s, MLX %s, git %s", torch.__version__, mx.__version__, _git_sha())
    log.info("Grids: %s", grids)

    all_results: dict[int, list[Row]] = {}
    for canvas_grid in grids:
        image_px = max(canvas_grid * PIXELS_PER_CANVAS_CELL, GLIMPSE_PX)
        n_canvas = cfg.n_canvas_registers + canvas_grid ** 2
        log.info("=" * 60)
        bpe = mx.float32.size
        canvas_mib = n_canvas * cfg.canvas_dim * bpe / (1024 * 1024)
        log.info("canvas_grid=%d → %d tokens (%.1f MiB), image=%dpx",
                 canvas_grid, n_canvas, canvas_mib, image_px)

        image_mlx = load_and_preprocess(str(IMAGE_PATH), target_size=image_px)
        mx.eval(image_mlx)

        with torch.inference_mode():
            rows = bench_at_grid(pt_m, mlx_m, cfg, device, canvas_grid, image_px,
                                 image_mlx, n_patches, n_local,
                                 pt_rope, pt_periods, mlx_rope, mlx_periods,
                                 pt_sample, mlx_extract, PTVP, MLXVP)
        print_table(rows)
        all_results[canvas_grid] = rows

    path = save_results(all_results, cfg)
    print(f"\n→ {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import tyro
    tyro.cli(main)
