# Default: run all checks
default: check

# Full smoketest: readme, lint, typecheck, dependency enforcement, tests
check: readme lint typecheck tach test

readme:
    uv run python generate_readme.py

lint:
    uv run ruff check canvit_mlx/ tests/ convert.py

typecheck:
    uv run basedpyright canvit_mlx/

tach:
    uv run tach check

test:
    uv run pytest

convert:
    uv run python convert.py

bench *ARGS:
    #!/usr/bin/env bash
    set -euo pipefail
    DIR="bench/$(date -u +%Y-%m-%dT%H-%M-%S)"
    mkdir -p "$DIR"
    uv run python -m bench.run_latency --out-dir "$DIR" {{ARGS}}
    uv run python -m bench.run_memory --out-dir "$DIR"
    uv run python -m bench.plot "$DIR/results.parquet"

run-bench *ARGS:
    uv run python -m bench.run_latency {{ARGS}}

run-memory *ARGS:
    uv run python -m bench.run_memory {{ARGS}}

plot-bench *ARGS:
    uv run python -m bench.plot {{ARGS}}

tree:
    uv run pypatree canvit_mlx
