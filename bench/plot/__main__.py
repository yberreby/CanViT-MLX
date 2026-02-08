"""Entry point: uv run python -m bench.plot [path]."""

import logging
import sys
from pathlib import Path

import polars as pl

from . import find_latest_parquet, make_plot

log = logging.getLogger(__name__)


def main() -> None:
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
