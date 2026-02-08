"""Bootstrap confidence interval helpers."""

import numpy as np
import polars as pl
from scipy.stats import bootstrap

N_RESAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
SEED = 42


def bootstrap_ci(samples: list[float]) -> tuple[float, float, float]:
    res = bootstrap(
        (samples,), np.median,
        n_resamples=N_RESAMPLES, confidence_level=CONFIDENCE_LEVEL,
        rng=np.random.default_rng(SEED),
    )
    ci = res.confidence_interval
    return float(np.median(samples)), float(ci.low), float(ci.high)


def ci_ms(df: pl.DataFrame, grid: int, component: str,
          raw_col: str, med_col: str) -> tuple[float, float, float]:
    """Median + CI in ms. Falls back to point estimate if no raw data."""
    row = df.filter((pl.col("grid") == grid) & (pl.col("component") == component))
    if raw_col in df.columns and row[raw_col][0] is not None:
        med, lo, hi = bootstrap_ci(row[raw_col][0].to_list())
        return med / 1000, lo / 1000, hi / 1000
    v = row[med_col][0] / 1000
    return v, v, v
