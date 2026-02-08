"""Stacked component breakdown: weighted MLX time per component."""

import numpy as np
import polars as pl
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from .ci import ci_ms


def _weighted_ci(df: pl.DataFrame, grid: int, component: str,
                 raw_col: str, med_col: str) -> tuple[float, float, float]:
    """Weighted CI in ms: median, lo, hi multiplied by component count."""
    med, lo, hi = ci_ms(df, grid, component, raw_col, med_col)
    count = df.filter((pl.col("grid") == grid) & (pl.col("component") == component))["count"][0]
    return med * count, lo * count, hi * count


def _stacked_bars(ax: Axes, x: np.ndarray, grids: list[int], components: list[str],
                  df: pl.DataFrame, raw_col: str, med_col: str,
                  bar_w: float, offset: float, colors: np.ndarray,
                  alpha: float = 1.0) -> None:
    n_grids = len(grids)
    bottoms = np.zeros(n_grids)
    for i, cname in enumerate(components):
        meds_los_his = [_weighted_ci(df, g, cname, raw_col, med_col) for g in grids]
        meds = np.array([med for med, _, _ in meds_los_his])
        yerr_lo = np.array([med - lo for med, lo, _ in meds_los_his])
        yerr_hi = np.array([hi - med for med, _, hi in meds_los_his])
        ax.bar(x + offset, meds, bar_w, bottom=bottoms, color=colors[i],
               yerr=[yerr_lo, yerr_hi], capsize=2, error_kw={"lw": 0.8, "alpha": 0.6},
               alpha=alpha)
        bottoms += meds


def plot_breakdown(ax: Axes, df: pl.DataFrame, *,
                   target_pxs: tuple[int, ...] | None = None) -> None:
    all_grids = sorted(df["grid"].unique().to_list())
    all_pxs = {g: df.filter(pl.col("grid") == g)["image_px"][0] for g in all_grids}
    if target_pxs is not None:
        target = set(target_pxs)
        grids = [g for g in all_grids if all_pxs[g] in target]
    else:
        grids = all_grids
    assert grids, "no grids to plot"
    image_pxs = [all_pxs[g] for g in grids]
    tick_labels = [f"${px}^2$" for px in image_pxs]
    first_grid = df.filter((pl.col("grid") == grids[0]) & pl.col("mlx_med_us").is_not_null())
    components = [c for c in first_grid["component"].to_list() if c != "full forward"]
    has_bf16 = "bf16_med_us" in df.columns
    n_grids = len(grids)
    colors = __import__("matplotlib").pyplot.cm.Set2(np.linspace(0, 1, len(components)))
    x = np.arange(n_grids)

    bar_w = 0.35 if has_bf16 else 0.6
    off_f32 = -bar_w / 2 if has_bf16 else 0.0
    off_bf16 = bar_w / 2

    _stacked_bars(ax, x, grids, components, df,
                  "mlx_raw_us", "mlx_med_us", bar_w, off_f32, colors)

    if has_bf16:
        _stacked_bars(ax, x, grids, components, df,
                      "bf16_raw_us", "bf16_med_us", bar_w, off_bf16, colors, alpha=0.75)

    for label, raw_col, med_col, off, marker, clr in [
        ("CanViT-B fwd fp32", "mlx_raw_us", "mlx_med_us", off_f32, "D", "black"),
        ("CanViT-B fwd bf16", "bf16_raw_us", "bf16_med_us", off_bf16, "d", "tab:orange"),
    ]:
        if "bf16" in label and not has_bf16:
            continue
        meds, los, his = zip(*[ci_ms(df, g, "full forward", raw_col, med_col) for g in grids])
        yerr = [[m - lo for m, lo in zip(meds, los)],
                [hi - m for m, hi in zip(meds, his)]]
        ax.errorbar(x + off, meds, yerr=yerr, fmt=marker, ms=7,
                    color=clr, zorder=5, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=7)
    if has_bf16:
        ax.tick_params(axis="x", pad=20)
        for xi in x:
            ax.text(xi + off_f32, -0.03, "f32", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5, color="0.4")
            ax.text(xi + off_bf16, -0.03, "bf16", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5, color="0.4")

    handles = list(Patch(facecolor=colors[i], label=c) for i, c in enumerate(components))
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.set_xlabel("Scene resolution")
    ax.set_ylabel("MLX time (ms)")
    ax.set_title("CanViT-B MLX component breakdown (weighted)")
    ax.ticklabel_format(axis="y", style="plain")
