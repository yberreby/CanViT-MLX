"""bf16 vs fp32 precision loss bar chart."""

import numpy as np
import polars as pl
from matplotlib.axes import Axes


def plot_precision(ax: Axes, df: pl.DataFrame) -> None:
    grids = sorted(df["grid"].unique().to_list())
    image_pxs = [df.filter(pl.col("grid") == g)["image_px"][0] for g in grids]
    tick_labels = [f"${px}^2$" for px in image_pxs]
    x = np.arange(len(grids))

    err_cols = [c for c in df.columns if c.startswith("bf16_err_")]
    if not err_cols:
        ax.set_visible(False)
        return

    err_series: list[tuple[str, str, str, str]] = []
    for col in err_cols:
        metric = col.removeprefix("bf16_err_")
        if metric == "last_hidden_state":
            err_series.append(("DINOv3", "dinov3", col, "#2ca02c"))
        else:
            err_series.append((f"CanViT-B {metric}", "full forward", col, "#1f77b4"))

    n_s = len(err_series)
    bar_w = 0.7 / max(n_s, 1)
    for i, (label, comp, col, color) in enumerate(err_series):
        errs = []
        for g in grids:
            row = df.filter((pl.col("grid") == g) & (pl.col("component") == comp))
            if row.shape[0] == 0 or row[col][0] is None:
                errs.append(0.0)
            else:
                errs.append(row[col][0] * 100)
        off = (i - (n_s - 1) / 2) * bar_w
        ax.bar(x + off, errs, bar_w * 0.9, label=label, color=color,
               alpha=0.6 + 0.4 * i / max(n_s - 1, 1))
        for j, v in enumerate(errs):
            ax.text(x[j] + off, v + 0.1, f"{v:.1f}%", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("Scene resolution")
    ax.set_ylabel("bf16 vs fp32 relative error (%)")
    ax.set_title("bf16 precision loss")
    ax.legend(fontsize=7)
