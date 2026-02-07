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

tree:
    uv run pypatree canvit_mlx
