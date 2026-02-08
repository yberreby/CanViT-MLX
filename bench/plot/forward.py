"""Full forward timing: CanViT (PT/MLX) vs teacher baselines."""

import numpy as np
import polars as pl
from matplotlib.axes import Axes

from .ci import ci_ms

# (component, label_base, color, marker)
TEACHER_SERIES = [
    ("dinov3", "DINOv3 ViT-B", "#2ca02c", "^"),
    ("dinov3s", "DINOv3 ViT-S", "#9467bd", "v"),
]


def plot_forward(ax: Axes, df: pl.DataFrame, *,
                 show_bf16: bool = False,
                 clip_to_canvit: bool = True,
                 log_scale: bool = False) -> None:
    grids = sorted(df["grid"].unique().to_list())
    image_pxs = [df.filter(pl.col("grid") == g)["image_px"][0] for g in grids]
    has_bf16 = "bf16_med_us" in df.columns
    row0 = df.row(0, named=True)
    pt_dev = row0["pt_device"].upper()

    series: list[tuple[str, str, str, str, str, str, str]] = [
        (f"CanViT-B PT fp32 ({pt_dev})", "full forward", "pt_raw_us", "pt_med_us", "#d62728", "s", "-"),
        ("CanViT-B MLX fp32", "full forward", "mlx_raw_us", "mlx_med_us", "#1f77b4", "o", "-"),
    ]
    if has_bf16 and show_bf16:
        series.append(
            ("CanViT-B MLX bf16", "full forward", "bf16_raw_us", "bf16_med_us", "#1f77b4", "o", "--"))

    for comp, label_base, color, marker in TEACHER_SERIES:
        if df.filter(pl.col("component") == comp).shape[0] == 0:
            continue
        series.append((f"{label_base} fp32 ({pt_dev})", comp,
                        "pt_raw_us", "pt_med_us", color, marker, "-"))
        comp_has_bf16 = (has_bf16 and
                         df.filter(pl.col("component") == comp)["bf16_med_us"][0] is not None)
        if comp_has_bf16:
            series.append((f"{label_base} bf16 ({pt_dev})", comp,
                            "bf16_raw_us", "bf16_med_us", color, marker, "--"))

    for label, comp, raw_col, med_col, color, marker, ls in series:
        comp_grids = sorted(df.filter(pl.col("component") == comp)["grid"].unique().to_list())
        comp_pxs = np.array([
            df.filter((pl.col("grid") == g) & (pl.col("component") == comp))["image_px"][0]
            for g in comp_grids
        ], dtype=float)
        meds, los, his = zip(*[ci_ms(df, g, comp, raw_col, med_col) for g in comp_grids])
        ax.plot(comp_pxs, meds, marker=marker, label=label, color=color, linestyle=ls)
        if any(lo != hi for lo, hi in zip(los, his)):
            ax.fill_between(comp_pxs, los, his, alpha=0.15, color=color)

    if log_scale:
        import matplotlib.pyplot as plt
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax.ticklabel_format(axis="y", style="plain")
    major_pxs = [px for px in image_pxs if px & (px - 1) == 0]
    ax.set_xticks(major_pxs)
    ax.set_xticklabels([f"${px}^2$" for px in major_pxs])
    ax.set_xlabel("Scene resolution")
    ax.set_ylabel("Full forward time (ms)")
    ax.set_title("Full forward (95% CI)")
    ax.set_ylim(bottom=0)
    if clip_to_canvit:
        canvit_max = max(ci_ms(df, g, "full forward", "pt_raw_us", "pt_med_us")[0] for g in grids)
        ax.set_ylim(top=canvit_max * 1.3)
    ax.legend(fontsize=8, loc="upper right")
