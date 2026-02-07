"""Plot benchmark results from bench/<timestamp>.parquet.

Reads the latest results parquet and generates:
  - bench/<stem>.png
  - bench/<stem>.svg

Usage:
    uv run python bench/plot.py                           # latest results
    uv run python bench/plot.py bench/2026-xxx.parquet    # specific file
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import bootstrap

log = logging.getLogger(__name__)

OUT_DIR = Path(__file__).parent
N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def find_latest_parquet() -> Path:
    parquets = sorted(OUT_DIR.glob("*.parquet"), key=lambda p: p.stat().st_mtime)
    assert parquets, f"No .parquet files in {OUT_DIR}"
    return parquets[-1]


def bootstrap_ci(samples: list[float]) -> tuple[float, float, float]:
    """Return (median, lo, hi) via BCa bootstrap (scipy)."""
    res = bootstrap(
        (samples,), np.median,
        n_resamples=N_RESAMPLES, confidence_level=CONFIDENCE_LEVEL,
        rng=np.random.default_rng(SEED),
    )
    ci = res.confidence_interval
    return float(np.median(samples)), float(ci.low), float(ci.high)


def make_plot(df: pl.DataFrame, stem: str) -> None:
    grids = sorted(df["grid"].unique().to_list())
    components = [c for c in df.filter(pl.col("grid") == grids[0])["component"].to_list()
                  if c != "full forward"]
    has_raw = "pt_raw_us" in df.columns

    log.info("Bootstrap: n_resamples=%d, confidence_level=%.2f, seed=%d", N_RESAMPLES, CONFIDENCE_LEVEL, SEED)
    log.info("Grids: %s", grids)
    log.info("Components: %s", components)
    log.info("Has raw times: %s", has_raw)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(components)))

    # ── Left: stacked bar of weighted MLX time ───────────────────
    x = np.arange(len(grids))
    bottoms = np.zeros(len(grids))

    log.info("--- Left panel: stacked bar (weighted MLX ms) ---")
    for i, cname in enumerate(components):
        vals = []
        for g in grids:
            row = df.filter((pl.col("grid") == g) & (pl.col("component") == cname))
            vals.append(row["mlx_med_us"][0] * row["count"][0] / 1000)
        log.info("  %-16s %s", cname, ["%.1f" % v for v in vals])
        ax1.bar(x, vals, bottom=bottoms, label=cname, color=colors[i])
        bottoms += vals

    log.info("  %-16s %s", "Σ components", ["%.1f" % v for v in bottoms])

    full_meds, full_los, full_his = [], [], []
    for g in grids:
        row = df.filter((pl.col("grid") == g) & (pl.col("component") == "full forward"))
        if has_raw:
            med, lo, hi = bootstrap_ci(row["mlx_raw_us"][0].to_list())
            full_meds.append(med / 1000); full_los.append(lo / 1000); full_his.append(hi / 1000)
        else:
            v = row["mlx_med_us"][0] / 1000
            full_meds.append(v); full_los.append(v); full_his.append(v)
    yerr_lo = [m - lo for m, lo in zip(full_meds, full_los)]
    yerr_hi = [hi - m for m, hi in zip(full_meds, full_his)]
    log.info("  full forward med: %s", ["%.1f" % v for v in full_meds])
    log.info("  full forward CI:  %s", ["[%.1f, %.1f]" % (lo, hi) for lo, hi in zip(full_los, full_his)])
    ax1.errorbar(x, full_meds, yerr=[yerr_lo, yerr_hi], fmt='D', ms=8,
                 color='black', zorder=5, capsize=4, label='full forward')

    ax1.set_xticks(x)
    ax1.set_xticklabels([str(g) for g in grids])
    ax1.set_xlabel("Canvas grid size")
    ax1.set_ylabel("MLX time (ms)")
    ax1.set_title("MLX component breakdown (weighted by count)")
    ax1.legend(fontsize=8)

    # ── Right: PT vs MLX absolute time with 95% bootstrap CI ────
    log.info("--- Right panel: PT vs MLX full forward ---")

    for label, raw_col, med_col, color, marker in [
        ("PyTorch", "pt_raw_us", "pt_med_us", "#d62728", "s"),
        ("MLX",     "mlx_raw_us", "mlx_med_us", "#1f77b4", "o"),
    ]:
        meds, los, his = [], [], []
        for g in grids:
            row = df.filter((pl.col("grid") == g) & (pl.col("component") == "full forward"))
            if has_raw:
                med, lo, hi = bootstrap_ci(row[raw_col][0].to_list())
                meds.append(med / 1000); los.append(lo / 1000); his.append(hi / 1000)
            else:
                v = row[med_col][0] / 1000
                meds.append(v); los.append(v); his.append(v)

        log.info("  %-8s med: %s", label, ["%.1f" % v for v in meds])
        log.info("  %-8s CI:  %s", label, ["[%.1f, %.1f]" % (lo, hi) for lo, hi in zip(los, his)])
        ci_pct = [(hi - lo) / med * 100 for med, lo, hi in zip(meds, los, his)]
        log.info("  %-8s CI%%: %s", label, ["%.1f%%" % p for p in ci_pct])

        ax2.plot(x, meds, marker=marker, label=label, color=color)
        if has_raw:
            ax2.fill_between(x, los, his, alpha=0.2, color=color)

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(g) for g in grids])
    ax2.set_xlabel("Canvas grid size")
    ax2.set_ylabel("Full forward time (ms)")
    ax2.set_title("PT vs MLX (full forward, 95% CI)")
    ax2.legend(fontsize=8)

    # Metadata subtitle
    row0 = df.row(0, named=True)
    fig.suptitle(f"git {row0['git_sha']}  |  {row0['timestamp'][:19]}  |  "
                 f"Torch {row0['torch_version']}  MLX {row0['mlx_version']}",
                 fontsize=9, color='gray')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    for ext in ("png", "svg"):
        path = OUT_DIR / f"{stem}.{ext}"
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

    stem = parquet_path.stem
    make_plot(df, stem)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
