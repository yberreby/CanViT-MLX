"""Plot benchmark results from bench/<timestamp>/results.parquet.

Usage:
    uv run python -m bench.plot                                    # latest results
    uv run python -m bench.plot bench/2026-xxx/results.parquet     # specific file
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

from .breakdown import plot_breakdown
from .forward import plot_forward
from .memory import plot_memory
from .precision import plot_precision

log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent.parent

# Breakdown shows only these scene resolutions (power-of-2 grid sizes).
BREAKDOWN_PXS = (128, 256, 512, 1024, 2048)


def find_latest_parquet() -> Path:
    parquets = sorted(BENCH_DIR.glob("*/results.parquet"), key=lambda p: p.stat().st_mtime)
    assert parquets, f"No results.parquet found in {BENCH_DIR}/*/"
    return parquets[-1]


def _suptitle(row0: dict) -> str:
    return (f"git {row0['git_sha']}  |  {row0['timestamp'][:19]}  |  "
            f"Torch {row0['torch_version']}  MLX {row0['mlx_version']}  "
            f"PT device: {row0['pt_device']}")


def _save_fig(fig, out_dir: Path, name: str) -> None:
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved %s", path)
    plt.close(fig)


def make_plot(df: pl.DataFrame, out_dir: Path, *,
              show_bf16: bool = True, clip_to_canvit: bool = True,
              log_scale: bool = False, show_precision: bool = False) -> None:
    row0 = df.row(0, named=True)
    title = _suptitle(row0)

    # Memory data lives in a separate memory.parquet from run_memory.py
    mem_path = out_dir / "memory.parquet"
    mem_df = pl.read_parquet(mem_path) if mem_path.exists() else None

    plots: list[tuple[str, object, dict]] = [
        ("breakdown", plot_breakdown, {"target_pxs": BREAKDOWN_PXS}),
        ("forward", plot_forward, {"show_bf16": show_bf16,
                                    "clip_to_canvit": clip_to_canvit,
                                    "log_scale": log_scale}),
    ]
    if show_precision:
        plots.append(("precision", plot_precision, {}))

    # ── Combined figure ──────────────────────────────────────────────────
    n = len(plots)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (_, plot_fn, kwargs) in zip(axes, plots):
        plot_fn(ax, df, **kwargs)
    fig.suptitle(title, fontsize=9, color='gray')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out_dir, "results")

    # ── Individual figures ───────────────────────────────────────────────
    for name, plot_fn, kwargs in plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        plot_fn(ax, df, **kwargs)
        fig.suptitle(title, fontsize=8, color='gray')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_fig(fig, out_dir, name)

    # ── Memory figure (from separate memory.parquet) ─────────────────
    if mem_df is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plot_memory(tuple(axes), mem_df)
        fig.suptitle(title, fontsize=8, color='gray')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        _save_fig(fig, out_dir, "memory")
