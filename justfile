# Default: run all checks
default: check

# Full smoketest: lint, typecheck, dependency enforcement, tests
check: lint typecheck tach test

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

tree:
    uv run pypatree canvit_mlx
