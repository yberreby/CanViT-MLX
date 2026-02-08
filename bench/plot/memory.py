"""Memory vs scene resolution — two panels: MLX (peak) and MPS (no inference_mode).

Reads memory.parquet from run_memory.py.
"""

import numpy as np
import polars as pl
from matplotlib.axes import Axes

MiB = 1024 * 1024

MLX_SERIES = [
    ("canvit_mlx_fp32_peak", "CanViT-B fp32", "#1f77b4", "o", "-"),
    ("canvit_mlx_bf16_peak", "CanViT-B bf16", "#ff7f0e", "s", "--"),
]

PT_SERIES = [
    ("canvit_pt_mps", "CanViT-B fp32", "#d62728", "D", "-"),
    ("dinov3_mps", "DINOv3 ViT-B fp32", "#2ca02c", "^", "-"),
    ("dinov3s_mps", "DINOv3 ViT-S fp32", "#9467bd", "v", "-"),
]


def _style_ax(ax: Axes, df: pl.DataFrame, title: str) -> None:
    all_pxs = sorted(df["image_px"].unique().to_list())
    major_pxs = [px for px in all_pxs if px & (px - 1) == 0]
    ax.set_xticks(major_pxs)
    ax.set_xticklabels([f"${px}^2$" for px in major_pxs])
    ax.set_xlabel("Scene resolution")
    ax.set_ylabel("Memory (MiB)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    ax.ticklabel_format(axis="y", style="plain")


def _plot_series(ax: Axes, df: pl.DataFrame,
                 series: list[tuple[str, str, str, str, str]]) -> None:
    for col, label, color, marker, ls in series:
        if col not in df.columns:
            continue
        sub = df.filter(pl.col(col).is_not_null()).sort("image_px")
        if sub.shape[0] == 0:
            continue
        pxs = np.array(sub["image_px"].to_list(), dtype=float)
        mem = np.array(sub[col].to_list(), dtype=float) / MiB
        ax.plot(pxs, mem, marker=marker, linestyle=ls, color=color, label=label)


def plot_memory(axes: tuple[Axes, Axes], df: pl.DataFrame) -> None:
    if df.shape[0] == 0:
        for ax in axes:
            ax.set_visible(False)
        return

    ax_mlx, ax_pt = axes
    _plot_series(ax_mlx, df, MLX_SERIES)
    _style_ax(ax_mlx, df, "MLX peak memory")

    _plot_series(ax_pt, df, PT_SERIES)
    _style_ax(ax_pt, df, "PyTorch MPS current_allocated\n(no inference_mode)")
