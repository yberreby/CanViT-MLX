"""Component-level latency benchmark: PT vs MLX (fp32 + bf16).

Each component is fed the SAME input; correctness is checked BEFORE timing.
Structurally impossible to bench something that isn't verified 1-to-1.

Components: init_state, patch_embed, ViT block, canvas read, canvas write, full forward.
Teacher baselines: DINOv3 ViT-B/16, DINOv3 ViT-S/16 (timing only, no MLX equivalent).
Outputs: bench/<timestamp>/{results.parquet, meta.json, pca.png}.

Usage:
    uv run python -m bench.run_latency                          # default grids, PT on MPS
    uv run python -m bench.run_latency --grids 32 64 128        # custom grids
    uv run python -m bench.run_latency --pt-device cpu           # PT on CPU
    uv run python -m bench.run_latency --dinov3-bf16             # enable DINOv3 ViT-B bf16
    uv run python -m bench.run_latency --teacher-max-px 512      # teacher baselines up to 512px

Notes:
    DINOv3 bf16 is off by default: on CPU ARM has no native bf16 compute
    (~2-5× slower than fp32), on MPS the speedup is marginal (<10%).
"""

import logging
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from .shared import (
    B, BENCH_DIR, DINOV3_REPO, DINOV3S_REPO, GLIMPSE_PX, HF_REPO,
    IMAGE_PATH, IMAGENET_MEAN, IMAGENET_STD, TEACHER_COMPONENTS, WEIGHTS,
    git_sha, make_pt_sync,
)

log = logging.getLogger(__name__)

WARMUP = 1
TIME_BUDGET_SECS = 1.0
MIN_ITERS = 2
MAX_ITERS = 100
SEED = 42
RTOL = 5e-3


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
    bf16: Stats | None = None
    bf16_errs: dict[str, float] | None = None

    @property
    def ratio(self) -> float:
        return self.pt.med / self.mlx.med

    @property
    def weighted(self) -> float:
        return self.mlx.med * self.count

    @property
    def bf16_weighted(self) -> float:
        return self.bf16.med * self.count if self.bf16 else 0.0


# ── Core bench primitives ───────────────────────────────────────────────────

def time_fn(fn, sync) -> Stats:
    for _ in range(WARMUP):
        fn(); sync()
    # Calibrate: time one iteration to decide how many to run
    sync(); t0 = time.perf_counter(); fn(); sync()
    one = time.perf_counter() - t0
    n = int(TIME_BUDGET_SECS / max(one, 1e-9))
    n = max(MIN_ITERS, min(n, MAX_ITERS))
    log.info("    calibrated: %.1fms/iter → %d iters (budget=%.1fs)", one * 1e3, n, TIME_BUDGET_SECS)
    ts = []
    for _ in range(n):
        sync(); t0 = time.perf_counter(); fn(); sync()
        ts.append(time.perf_counter() - t0)
    return Stats.from_secs(ts)


def check(name: str, ref: np.ndarray, got: np.ndarray) -> float:
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
    pt_val = pt_fn(); pt_sync()
    mlx_val = mlx_fn(); mlx_sync()
    rel = check(name, pt_val.cpu().numpy(), np.asarray(mlx_val))
    return Row(name, count, time_fn(pt_fn, pt_sync), time_fn(mlx_fn, mlx_sync), rel)


def bench_teacher(name: str, model: torch.nn.Module, repo: str,
                  pil_image, image_px: int, device: torch.device,
                  pt_sync, *, model_bf: torch.nn.Module | None = None) -> Row:
    from transformers import AutoImageProcessor

    proc = AutoImageProcessor.from_pretrained(
        repo, size={"height": image_px, "width": image_px})
    inp = proc(images=pil_image, return_tensors="pt")["pixel_values"].to(device)
    log.info("  %s input: %s", name, list(inp.shape))

    fp32_stats = time_fn(lambda: model(inp).last_hidden_state, pt_sync)

    bf16_stats = None
    bf16_errs = None
    if model_bf is not None:
        inp_bf = inp.to(torch.bfloat16)
        bf16_stats = time_fn(
            lambda: model_bf(inp_bf).last_hidden_state, pt_sync)
        ref = model(inp).last_hidden_state.float()
        got = model_bf(inp_bf).last_hidden_state.float()
        pt_sync()
        err = check(f"{name} bf16 vs fp32", ref.cpu().numpy(), got.cpu().numpy())
        bf16_errs = {"last_hidden_state": err}
        log.info("  %s fp32: %.0fμs, bf16: %.0fμs (%.2f×)",
                 name, fp32_stats.med, bf16_stats.med,
                 fp32_stats.med / bf16_stats.med)
    else:
        log.info("  %s fp32: %.0fμs", name, fp32_stats.med)

    row = Row(name, 1, fp32_stats, fp32_stats, 0.0, bf16_stats)
    row.bf16_errs = bf16_errs
    return row


def _fmt_row(name, count, pt_med, f32_med, bf16_med):
    bf = f"{bf16_med:>8.0f}" if bf16_med else f"{'—':>8}"
    f32_pt = f"{pt_med / f32_med:>7.2f}×"
    bf_pt = f"{pt_med / bf16_med:>7.2f}×" if bf16_med else f"{'—':>7}"
    bf_f32 = f"{f32_med / bf16_med:>8.2f}×" if bf16_med else f"{'—':>8}"
    return (f"{name:<16} {count:>3} "
            f"{pt_med:>8.0f} {f32_med:>8.0f} {bf} "
            f"{f32_pt} {bf_pt} {bf_f32}")


def print_table(rows: list[Row]) -> None:
    hdr = (f"{'Component':<16} {'×':>3} "
           f"{'PT med':>8} {'f32 med':>8} {'bf16 med':>8} "
           f"{'f32/PT':>7} {'bf16/PT':>7} {'bf16/f32':>8}")
    sep = "─" * len(hdr)
    components = [r for r in rows if r.name != "full forward" and r.name not in TEACHER_COMPONENTS]
    sum_pt = sum(r.pt.med * r.count for r in components)
    sum_f32 = sum(r.weighted for r in components)
    sum_bf = sum(r.bf16_weighted for r in components)

    print(f"\n{hdr}")
    print(sep)
    for r in components:
        print(_fmt_row(r.name, r.count, r.pt.med, r.mlx.med,
                       r.bf16.med if r.bf16 else None))
    print(sep)
    print(_fmt_row("Σ components", 0, sum_pt, sum_f32, sum_bf or None)
          .replace("  0 ", "    "))
    for r in rows:
        if r.name == "full forward" or r.name in TEACHER_COMPONENTS:
            print(_fmt_row(r.name, r.count, r.pt.med, r.mlx.med,
                           r.bf16.med if r.bf16 else None))
    print(sep)


# ── Bench driver ────────────────────────────────────────────────────────────

def bench_at_grid(pt_m, mlx_m, mlx_m_bf16, cfg, device, canvas_grid, image_px,
                  image_mlx: mx.array, n_patches, n_local,
                  pt_rope, pt_periods, mlx_rope, mlx_periods,
                  pt_sample, mlx_extract, PTVP, MLXVP) -> list[Row]:
    pt_sync = make_pt_sync(device)
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
    pt_sync()

    mlx_ref: list = [None]
    def mlx_wrap(fn):
        def wrapper():
            mlx_ref[0] = fn()
            return mlx_ref[0]
        return wrapper
    def mlx_sync():
        if mlx_ref[0] is not None:
            mx.eval(mlx_ref[0])

    # ── fp32: correctness + timing ──────────────────────────────────────────
    log.info("--- fp32: Check + Bench (grid=%d, image=%dpx, tokens=%d) ---",
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
        pt_sync, mlx_sync,
    ))

    glimpse_pt = pt_sample(spatial=image_pt, viewpoint=vp_pt, glimpse_size_px=GLIMPSE_PX)
    glimpse_mlx = mlx_extract(image_mlx, vp_mlx, GLIMPSE_PX)
    mx.eval(glimpse_mlx); pt_sync()

    rows.append(check_and_bench(
        "init_state", 1,
        lambda: pt_m.init_state(batch_size=B, canvas_grid_size=canvas_grid).canvas,
        mlx_wrap(lambda: mlx_m.init_state(batch_size=B, canvas_grid_size=canvas_grid).canvas),
        pt_sync, mlx_sync,
    ))
    rows.append(check_and_bench(
        "patch_embed", 1,
        lambda: pt_m.backbone.vit.patch_embed(glimpse_pt).flatten(1, 2),
        mlx_wrap(lambda: mlx_m.patch_embed(glimpse_mlx)[0]),
        pt_sync, mlx_sync,
    ))
    rows.append(check_and_bench(
        "ViT block", cfg.n_blocks,
        lambda: pt_m.backbone.forward_block(0, local_pt, bb_rope),
        mlx_wrap(lambda: mlx_m.blocks[0](local_mlx, bb_sin, bb_cos)),
        pt_sync, mlx_sync,
    ))
    rows.append(check_and_bench(
        "canvas read", n_reads,
        lambda: pt_m.read_attn[0](query=local_pt, kv=canvas_pt,
                                   query_rope=ca_rope_l, kv_rope=ca_rope_c),
        mlx_wrap(lambda: local_mlx + mlx_m.read_scales[0](
            mlx_m.read_attn[0](local_mlx, canvas_mlx,
                               ca_sin_l, ca_cos_l, ca_sin_c, ca_cos_c))),
        pt_sync, mlx_sync,
    ))
    rows.append(check_and_bench(
        "canvas write", n_writes,
        lambda: pt_m.write_attn[0](query=canvas_pt, kv=local_pt,
                                    query_rope=ca_rope_c, kv_rope=ca_rope_l),
        mlx_wrap(lambda: canvas_mlx + mlx_m.write_attn[0](
            canvas_mlx, local_mlx, ca_sin_c, ca_cos_c, ca_sin_l, ca_cos_l)),
        pt_sync, mlx_sync,
    ))

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
        pt_full, mlx_wrap(mlx_full), pt_sync, mlx_sync,
    ))

    # ── bf16: timing ────────────────────────────────────────────────────────
    log.info("--- bf16: Bench (grid=%d) ---", canvas_grid)

    local_bf = mx.array(local_np).astype(mx.bfloat16)
    canvas_bf = mx.array(canvas_np).astype(mx.bfloat16)
    image_bf = image_mlx.astype(mx.bfloat16)
    vp_bf = MLXVP(centers=mx.zeros((B, 2), dtype=mx.bfloat16),
                  scales=mx.ones((B,), dtype=mx.bfloat16))

    bb_sin_bf, bb_cos_bf = mlx_rope(
        mx.array(local_pos_np).astype(mx.bfloat16), mlx_periods(cfg.head_dim))
    ca_sl_bf, ca_cl_bf = mlx_rope(
        mx.array(local_pos_np).astype(mx.bfloat16), mlx_periods(cfg.canvas_head_dim))
    ca_sc_bf, ca_cc_bf = mlx_rope(
        mx.array(canvas_pos_np).astype(mx.bfloat16), mlx_periods(cfg.canvas_head_dim))

    glimpse_bf = mlx_extract(image_bf, vp_bf, GLIMPSE_PX)
    mx.eval(local_bf, canvas_bf, image_bf, glimpse_bf,
            bb_sin_bf, bb_cos_bf, ca_sl_bf, ca_cl_bf, ca_sc_bf, ca_cc_bf)

    bf_ref: list = [None]
    def bf_wrap(fn):
        def w():
            bf_ref[0] = fn()
            return bf_ref[0]
        return w
    def bf_sync():
        if bf_ref[0] is not None:
            mx.eval(bf_ref[0])

    bf_fns = [
        ("glimpse", bf_wrap(lambda: mlx_extract(image_bf, vp_bf, GLIMPSE_PX))),
        ("init_state", bf_wrap(lambda: mlx_m_bf16.init_state(
            batch_size=B, canvas_grid_size=canvas_grid).canvas)),
        ("patch_embed", bf_wrap(lambda: mlx_m_bf16.patch_embed(glimpse_bf)[0])),
        ("ViT block", bf_wrap(lambda: mlx_m_bf16.blocks[0](
            local_bf, bb_sin_bf, bb_cos_bf))),
        ("canvas read", bf_wrap(lambda: local_bf + mlx_m_bf16.read_scales[0](
            mlx_m_bf16.read_attn[0](local_bf, canvas_bf,
                                     ca_sl_bf, ca_cl_bf, ca_sc_bf, ca_cc_bf)))),
        ("canvas write", bf_wrap(lambda: canvas_bf + mlx_m_bf16.write_attn[0](
            canvas_bf, local_bf, ca_sc_bf, ca_cc_bf, ca_sl_bf, ca_cl_bf))),
    ]

    def mlx_full_bf16():
        g = mlx_extract(image_bf, vp_bf, GLIMPSE_PX)
        s = mlx_m_bf16.init_state(batch_size=B, canvas_grid_size=canvas_grid)
        o = mlx_m_bf16(g, s, vp_bf)
        mx.eval(o.state.recurrent_cls, o.ephemeral_cls, o.local_patches)
        return o.state.canvas

    bf_fns.append(("full forward", bf_wrap(mlx_full_bf16)))

    for name, fn in bf_fns:
        row = next(r for r in rows if r.name == name)
        row.bf16 = time_fn(fn, bf_sync)
        speedup = row.mlx.med / row.bf16.med
        log.info("  %s bf16: %.0fμs (fp32: %.0fμs, speedup: %.2f×)",
                 name, row.bf16.med, row.mlx.med, speedup)

    # bf16 vs fp32 precision on multiple representations
    log.info("--- bf16 vs fp32: precision ---")
    f32_c = mlx_full(); mlx_sync()
    bf_c = mlx_full_bf16(); bf_sync()
    n_regs = cfg.n_canvas_registers

    f32_ln = mlx_m.scene_patches_ln(f32_c[:, n_regs:])
    bf_ln = mlx_m_bf16.scene_patches_ln(bf_c[:, n_regs:])
    f32_proj = mlx_m.scene_patches_proj(f32_ln)
    bf_proj = mlx_m_bf16.scene_patches_proj(bf_ln)
    mx.eval(f32_ln, bf_ln, f32_proj, bf_proj)

    def _to_np(x): return np.asarray(x.astype(mx.float32) if x.dtype != mx.float32 else x)

    fwd_row = next(r for r in rows if r.name == "full forward")
    fwd_row.bf16_errs = {
        "raw_canvas": check("raw canvas", _to_np(f32_c), _to_np(bf_c)),
        "ln_canvas": check("LN canvas", _to_np(f32_ln), _to_np(bf_ln)),
        "scene_proj": check("scene proj", _to_np(f32_proj), _to_np(bf_proj)),
    }

    return rows


# ── PCA + artifacts ─────────────────────────────────────────────────────────

def _denorm(x: np.ndarray) -> np.ndarray:
    return np.clip(x * IMAGENET_STD + IMAGENET_MEAN, 0, 1)


def save_pca_and_artifacts(run_dir: Path, mlx_m, mlx_m_bf16, image_mlx, cfg,
                           canvas_grid: int, MLXVP, mlx_extract):
    """Single step from init canvas, full-scene VP. Saves PCA + glimpse + input."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    n_regs = cfg.n_canvas_registers

    vp32 = MLXVP.full_scene(batch_size=B)
    g32 = mlx_extract(image_mlx, vp32, GLIMPSE_PX)
    out32 = mlx_m(g32, mlx_m.init_state(B, canvas_grid), vp32)
    mx.eval(out32.state.canvas, g32)

    image_bf = image_mlx.astype(mx.bfloat16)
    vp_bf = MLXVP(centers=mx.zeros((B, 2), dtype=mx.bfloat16),
                  scales=mx.ones((B,), dtype=mx.bfloat16))
    g_bf = mlx_extract(image_bf, vp_bf, GLIMPSE_PX)
    out_bf = mlx_m_bf16(g_bf, mlx_m_bf16.init_state(B, canvas_grid), vp_bf)
    mx.eval(out_bf.state.canvas, g_bf)

    c32 = np.array(out32.state.canvas)
    cbf = np.array(out_bf.state.canvas.astype(mx.float32))
    rel = np.linalg.norm(c32 - cbf) / (np.linalg.norm(c32) + 1e-8)
    log.info("PCA: canvas bf16 vs fp32 rel_err=%.4e", rel)

    def pca_rgb(canvas_np, pca_fit=None):
        spatial = canvas_np[0, n_regs:]
        g = int(math.sqrt(spatial.shape[0]))
        assert g * g == spatial.shape[0]
        if pca_fit is None:
            pca_fit = PCA(n_components=3, whiten=True).fit(spatial)
        proj = pca_fit.transform(spatial)[:, :3]
        rgb = 1.0 / (1.0 + np.exp(-2.0 * np.clip(proj, -10, 10)))
        return rgb.reshape(g, g, 3), pca_fit

    rgb32, pca_fit = pca_rgb(c32)
    rgb_bf, _ = pca_rgb(cbf, pca_fit)

    input_np = _denorm(np.array(image_mlx[0]))
    g32_np = _denorm(np.array(g32[0]))
    gbf_np = _denorm(np.array(g_bf[0].astype(mx.float32)))
    image_px = image_mlx.shape[1]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(input_np)
    axes[0, 0].set_title(f"Input ({image_px}×{image_px})")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(g32_np)
    axes[0, 1].set_title(f"Glimpse fp32 ({GLIMPSE_PX}px)")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(gbf_np)
    axes[0, 2].set_title(f"Glimpse bf16 ({GLIMPSE_PX}px)")
    axes[0, 2].axis("off")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(rgb32)
    axes[1, 1].set_title("Canvas PCA fp32")
    axes[1, 1].axis("off")
    axes[1, 2].imshow(rgb_bf)
    axes[1, 2].set_title(f"Canvas PCA bf16 (rel_err={rel:.4f})")
    axes[1, 2].axis("off")
    fig.suptitle(f"Single step from init canvas, grid={canvas_grid}", fontsize=13)
    plt.tight_layout()
    path = run_dir / "pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Saved %s", path)


# ── Output ──────────────────────────────────────────────────────────────────



def save_results(run_dir: Path, all_results: dict[int, list[Row]], cfg,
                  pt_device: str) -> Path:
    import json

    import polars as pl

    ts = datetime.now(timezone.utc)
    git = git_sha()

    meta = {
        "timestamp": ts.isoformat(),
        "git_sha": git,
        "torch_version": torch.__version__,
        "mlx_version": mx.__version__,
        "batch_size": B,
        "glimpse_px": GLIMPSE_PX,
        "warmup": WARMUP,
        "time_budget_secs": TIME_BUDGET_SECS,
        "min_iters": MIN_ITERS,
        "max_iters": MAX_ITERS,
        "seed": SEED,
        "embed_dim": cfg.embed_dim,
        "canvas_dim": cfg.canvas_dim,
        "n_blocks": cfg.n_blocks,
        "grids": sorted(all_results.keys()),
        "pt_device": pt_device,
    }

    records = []
    for grid, rows in sorted(all_results.items()):
        for r in rows:
            has_mlx = r.name not in TEACHER_COMPONENTS
            rec = {
                **meta,
                "grid": grid,
                "image_px": max(grid * cfg.patch_size, GLIMPSE_PX),
                "canvas_tokens": cfg.n_canvas_registers + grid ** 2,
                "component": r.name,
                "count": r.count,
                "pt_med_us": r.pt.med,
                "pt_min_us": r.pt.mn,
                "pt_max_us": r.pt.mx,
                "pt_mean_us": r.pt.mean,
                "pt_raw_us": r.pt.raw_us,
                "mlx_med_us": r.mlx.med if has_mlx else None,
                "mlx_min_us": r.mlx.mn if has_mlx else None,
                "mlx_max_us": r.mlx.mx if has_mlx else None,
                "mlx_mean_us": r.mlx.mean if has_mlx else None,
                "mlx_raw_us": r.mlx.raw_us if has_mlx else None,
                "ratio": r.ratio if has_mlx else None,
                "rel_err": r.rel_err,
            }
            if r.bf16:
                rec |= {
                    "bf16_med_us": r.bf16.med,
                    "bf16_min_us": r.bf16.mn,
                    "bf16_max_us": r.bf16.mx,
                    "bf16_mean_us": r.bf16.mean,
                    "bf16_raw_us": r.bf16.raw_us,
                }
            if r.bf16_errs:
                for k, v in r.bf16_errs.items():
                    rec[f"bf16_err_{k}"] = v
            records.append(rec)

    df = pl.DataFrame(records)
    pq_path = run_dir / "results.parquet"
    json_path = run_dir / "meta.json"
    df.write_parquet(pq_path)
    json_path.write_text(json.dumps(meta, indent=2) + "\n")
    log.info("Saved %s + %s (%d rows, git=%s)", pq_path.name, json_path.name, len(records), git)
    return pq_path


# ── Main ────────────────────────────────────────────────────────────────────

def main(grids: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15, 16,
                                    17, 18, 19, 20, 21, 22, 23, 24,
                                    28, 32, 40, 48, 56, 64, 128),
         pt_device: str = "mps",
         dinov3_bf16: bool = False,
         teacher_max_px: int = 384,
         out_dir: str | None = None) -> None:

    from PIL import Image
    from transformers import AutoModel

    from canvit import CanViTForPretrainingHFHub, sample_at_viewpoint as pt_sample
    from canvit.rope import compute_rope as pt_rope, make_rope_periods as pt_periods
    from canvit.viewpoint import Viewpoint as PTVP

    from canvit_mlx import Viewpoint as MLXVP, load_canvit
    from canvit_mlx.glimpse import extract_glimpse_at_viewpoint as mlx_extract
    from canvit_mlx.preprocess import load_and_preprocess
    from canvit_mlx.rope import compute_rope as mlx_rope, make_rope_periods as mlx_periods

    device = torch.device(pt_device)
    pt_sync = make_pt_sync(device)

    # Output directory
    ts = datetime.now(timezone.utc)
    run_dir = Path(out_dir) if out_dir else BENCH_DIR / ts.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir.mkdir(exist_ok=True)
    log.info("Output: %s", run_dir)

    log.info("Loading models...")
    pt_m = CanViTForPretrainingHFHub.from_pretrained(HF_REPO).to(device).eval()
    mlx_m = load_canvit(WEIGHTS)
    mlx_m_bf16 = load_canvit(WEIGHTS)
    mlx_m_bf16.apply(lambda p: p.astype(mx.bfloat16))
    cfg = mlx_m.cfg

    log.info("Loading teacher baselines...")
    d3_model = AutoModel.from_pretrained(DINOV3_REPO, dtype=torch.float32).to(device).eval()
    d3_model_bf = (AutoModel.from_pretrained(DINOV3_REPO, dtype=torch.bfloat16).to(device).eval()
                   if dinov3_bf16 else None)
    vs_model = AutoModel.from_pretrained(DINOV3S_REPO, dtype=torch.float32).to(device).eval()
    log.info("Teachers loaded: ViT-B (%s), ViT-S (%s)", DINOV3_REPO, DINOV3S_REPO)

    teachers: list[tuple[str, torch.nn.Module, torch.nn.Module | None, str]] = [
        ("dinov3", d3_model, d3_model_bf, DINOV3_REPO),
        ("dinov3s", vs_model, None, DINOV3S_REPO),
    ]
    pil_image = Image.open(IMAGE_PATH)

    n_patches = (GLIMPSE_PX // cfg.patch_size) ** 2
    n_prefix = (1 if cfg.enable_vpe else 0) + 2 + cfg.n_register_tokens
    n_local = n_prefix + n_patches

    log.info("B=%d, glimpse=%dpx, warmup=%d, budget=%.1fs, iters=%d-%d, seed=%d",
             B, GLIMPSE_PX, WARMUP, TIME_BUDGET_SECS, MIN_ITERS, MAX_ITERS, SEED)
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
    log.info("Torch %s, MLX %s, git %s", torch.__version__, mx.__version__, git_sha())
    log.info("Grids: %s", grids)

    for g in grids:
        px = max(g * cfg.patch_size, GLIMPSE_PX)
        assert px % cfg.patch_size == 0, f"grid={g} → image={px}px not divisible by patch_size={cfg.patch_size}"

    all_results: dict[int, list[Row]] = {}
    for canvas_grid in grids:
        image_px = max(canvas_grid * cfg.patch_size, GLIMPSE_PX)
        n_canvas = cfg.n_canvas_registers + canvas_grid ** 2
        log.info("=" * 60)
        canvas_mib = n_canvas * cfg.canvas_dim * mx.float32.size / (1024 * 1024)
        log.info("canvas_grid=%d → %d tokens (%.1f MiB fp32, %.1f MiB bf16), image=%dpx",
                 canvas_grid, n_canvas, canvas_mib, canvas_mib / 2, image_px)

        image_mlx = load_and_preprocess(str(IMAGE_PATH), target_size=image_px)
        mx.eval(image_mlx)

        with torch.inference_mode():
            rows = bench_at_grid(pt_m, mlx_m, mlx_m_bf16, cfg, device,
                                 canvas_grid, image_px,
                                 image_mlx, n_patches, n_local,
                                 pt_rope, pt_periods, mlx_rope, mlx_periods,
                                 pt_sample, mlx_extract, PTVP, MLXVP)

            # Teacher baselines (PT-only timing)
            for tname, tmodel, tmodel_bf, trepo in teachers:
                if image_px <= teacher_max_px:
                    rows.append(bench_teacher(
                        tname, tmodel, trepo, pil_image, image_px,
                        device, pt_sync, model_bf=tmodel_bf))
                else:
                    log.info("  %s skipped (image_px=%d > teacher_max_px=%d)",
                             tname, image_px, teacher_max_px)

        print_table(rows)
        all_results[canvas_grid] = rows

    save_results(run_dir, all_results, cfg, pt_device)

    last_grid = grids[-1]
    pca_px = max(last_grid * cfg.patch_size, GLIMPSE_PX)
    pca_image = load_and_preprocess(str(IMAGE_PATH), target_size=pca_px)
    mx.eval(pca_image)
    save_pca_and_artifacts(run_dir, mlx_m, mlx_m_bf16, pca_image, cfg,
                           last_grid, MLXVP, mlx_extract)

    print(f"\n→ {run_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    import tyro
    tyro.cli(main)
