"""Plot benchmark results from bench/<timestamp>/results.parquet.

Reads the latest results and generates plots in the same directory.

Usage:
    uv run python bench/plot.py                                    # latest results
    uv run python bench/plot.py bench/2026-xxx/results.parquet     # specific file
"""

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from scipy.stats import bootstrap

log = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).parent
N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def find_latest_parquet() -> Path:
    parquets = sorted(BENCH_DIR.glob("*/results.parquet"), key=lambda p: p.stat().st_mtime)
    assert parquets, f"No results.parquet found in {BENCH_DIR}/*/"
    return parquets[-1]


def _bootstrap_ci(samples: list[float]) -> tuple[float, float, float]:
    res = bootstrap(
        (samples,), np.median,
        n_resamples=N_RESAMPLES, confidence_level=CONFIDENCE_LEVEL,
        rng=np.random.default_rng(SEED),
    )
    ci = res.confidence_interval
    return float(np.median(samples)), float(ci.low), float(ci.high)


def _ci_ms(df: pl.DataFrame, grid: int, component: str,
           raw_col: str, med_col: str) -> tuple[float, float, float]:
    """Median + CI in ms. Falls back to point estimate if no raw data."""
    row = df.filter((pl.col("grid") == grid) & (pl.col("component") == component))
    if raw_col in df.columns and row[raw_col][0] is not None:
        med, lo, hi = _bootstrap_ci(row[raw_col][0].to_list())
        return med / 1000, lo / 1000, hi / 1000
    v = row[med_col][0] / 1000
    return v, v, v


def make_plot(df: pl.DataFrame, out_dir: Path) -> None:
    grids = sorted(df["grid"].unique().to_list())
    image_pxs = [df.filter(pl.col("grid") == g)["image_px"][0] for g in grids]
    tick_labels = [f"{px}×{px}" for px in image_pxs]
    components = [c for c in df.filter(pl.col("grid") == grids[0])["component"].to_list()
                  if c not in ("full forward", "dinov3")]
    has_bf16 = "bf16_med_us" in df.columns
    row0 = df.row(0, named=True)

    log.info("Grids: %s, components: %s, bf16: %s", grids, components, has_bf16)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))
    x = np.arange(len(grids))

    # ── Left: stacked component breakdown (fp32 + bf16 grouped) ──
    # Two bars per grid: fp32 on left, bf16 on right, with sub-tick labels
    n_grids = len(grids)
    bar_w = 0.35 if has_bf16 else 0.6
    off_f32 = -bar_w / 2 if has_bf16 else 0.0
    off_bf16 = bar_w / 2

    def _weighted_ms(row: pl.DataFrame, med_col: str) -> float:
        return row[med_col][0] * row["count"][0] / 1000

    bottoms_f32 = np.zeros(n_grids)
    for i, cname in enumerate(components):
        vals = [_weighted_ms(df.filter((pl.col("grid") == g) & (pl.col("component") == cname)),
                             "mlx_med_us") for g in grids]
        ax1.bar(x + off_f32, vals, bar_w, bottom=bottoms_f32, color=colors[i])
        bottoms_f32 += vals

    if has_bf16:
        bottoms_bf = np.zeros(n_grids)
        for i, cname in enumerate(components):
            vals = [_weighted_ms(df.filter((pl.col("grid") == g) & (pl.col("component") == cname)),
                                 "bf16_med_us") for g in grids]
            ax1.bar(x + off_bf16, vals, bar_w, bottom=bottoms_bf,
                    color=colors[i], alpha=0.75)
            bottoms_bf += vals

    # Full-forward diamonds
    for label, raw_col, med_col, off, marker, clr in [
        ("full fwd fp32", "mlx_raw_us", "mlx_med_us", off_f32, "D", "black"),
        ("full fwd bf16", "bf16_raw_us", "bf16_med_us", off_bf16, "d", "tab:orange"),
    ]:
        if "bf16" in label and not has_bf16:
            continue
        meds, los, his = zip(*[_ci_ms(df, g, "full forward", raw_col, med_col) for g in grids])
        yerr = [[m - lo for m, lo in zip(meds, los)],
                [hi - m for m, hi in zip(meds, his)]]
        ax1.errorbar(x + off, meds, yerr=yerr, fmt=marker, ms=7,
                     color=clr, zorder=5, capsize=4)

    # Two-level x-axis: resolution on primary ticks, f32/bf16 as minor labels
    ax1.set_xticks(x)
    ax1.set_xticklabels(tick_labels, fontsize=7)
    if has_bf16:
        for xi in x:
            ax1.text(xi + off_f32, -0.03, "f32", transform=ax1.get_xaxis_transform(),
                     ha="center", va="top", fontsize=5, color="0.4")
            ax1.text(xi + off_bf16, -0.03, "bf16", transform=ax1.get_xaxis_transform(),
                     ha="center", va="top", fontsize=5, color="0.4")

    handles = list(Patch(facecolor=colors[i], label=c) for i, c in enumerate(components))
    ax1.legend(handles=handles, fontsize=7, ncol=2)
    ax1.set_xlabel("Scene resolution")
    ax1.set_ylabel("MLX time (ms)")
    ax1.set_title("MLX component breakdown (weighted)")

    # ── Right: full forward comparison ────────────────────────────────
    has_dinov3 = df.filter(pl.col("component") == "dinov3").shape[0] > 0
    pt_dev = row0["pt_device"].upper()
    series: list[tuple[str, str, str, str, str, str, str]] = [
        (f"CanViT PT fp32 ({pt_dev})",  "full forward", "pt_raw_us",   "pt_med_us",   "#d62728", "s", "-"),
        ("CanViT MLX fp32", "full forward", "mlx_raw_us",  "mlx_med_us",  "#1f77b4", "o", "-"),
    ]
    if has_bf16:
        series.append(
            ("CanViT MLX bf16", "full forward", "bf16_raw_us", "bf16_med_us", "#1f77b4", "o", "--"))
    if has_dinov3:
        series.append((f"DINOv3 fp32 ({pt_dev})", "dinov3", "pt_raw_us", "pt_med_us", "#2ca02c", "^", "-"))
        d3_has_bf16 = (has_bf16 and
                       df.filter(pl.col("component") == "dinov3")["bf16_med_us"][0] is not None)
        if d3_has_bf16:
            series.append((f"DINOv3 bf16 ({pt_dev})", "dinov3", "bf16_raw_us", "bf16_med_us", "#2ca02c", "^", "--"))

    for label, comp, raw_col, med_col, color, marker, ls in series:
        meds, los, his = zip(*[_ci_ms(df, g, comp, raw_col, med_col) for g in grids])
        ax2.plot(x, meds, marker=marker, label=label, color=color, linestyle=ls)
        if any(lo != hi for lo, hi in zip(los, his)):
            ax2.fill_between(x, los, his, alpha=0.15, color=color)

    ax2.set_yscale("log")
    ax2.set_xticks(x)
    ax2.set_xticklabels(tick_labels)
    ax2.set_xlabel("Scene resolution")
    ax2.set_ylabel("Full forward time (ms)")
    ax2.set_title("Full forward (95% CI)")
    ax2.legend(fontsize=8)

    # ── Panel 3: bf16 precision loss ─────────────────────────────────
    # Plot all bf16_err_* columns found in the data
    err_cols = [c for c in df.columns if c.startswith("bf16_err_")]
    if err_cols:
        err_series: list[tuple[str, str, str, str]] = []
        for col in err_cols:
            metric = col.removeprefix("bf16_err_")
            if metric == "last_hidden_state":
                err_series.append(("DINOv3", "dinov3", col, "#2ca02c"))
            else:
                err_series.append((f"CanViT {metric}", "full forward", col, "#1f77b4"))

        n_s = len(err_series)
        bar_w3 = 0.7 / max(n_s, 1)
        for i, (label, comp, col, color) in enumerate(err_series):
            errs = []
            for g in grids:
                row = df.filter((pl.col("grid") == g) & (pl.col("component") == comp))
                errs.append(row[col][0] * 100 if row[col][0] is not None else 0.0)
            off = (i - (n_s - 1) / 2) * bar_w3
            ax3.bar(x + off, errs, bar_w3 * 0.9, label=label, color=color,
                    alpha=0.6 + 0.4 * i / max(n_s - 1, 1))
            for j, v in enumerate(errs):
                ax3.text(x[j] + off, v + 0.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=6)

        ax3.set_xticks(x)
        ax3.set_xticklabels(tick_labels)
        ax3.set_xlabel("Scene resolution")
        ax3.set_ylabel("bf16 vs fp32 relative error (%)")
        ax3.set_title("bf16 precision loss")
        ax3.legend(fontsize=7)

    # Metadata subtitle
    fig.suptitle(f"git {row0['git_sha']}  |  {row0['timestamp'][:19]}  |  "
                 f"Torch {row0['torch_version']}  MLX {row0['mlx_version']}  "
                 f"PT device: {row0['pt_device']}",
                 fontsize=9, color='gray')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "svg"):
        path = out_dir / f"results.{ext}"
        fig.savefig(path, dpi=150)
        log.info("Saved %s", path)
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        parquet_path = Path(sys.argv[1])
    else:
        parquet_path = find_latest_parquet()

    log.info("Reading %s", parquet_path)
    df = pl.read_parquet(parquet_path)
    log.info("%d rows, grids=%s", df.shape[0], sorted(df["grid"].unique().to_list()))

    make_plot(df, parquet_path.parent)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
