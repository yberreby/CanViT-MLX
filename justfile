# Default: run all checks
default: check

# Full smoketest: lint, typecheck, dependency enforcement, tests
check: lint typecheck no-canvit-dep test

lint:
    uv run ruff check canvit_mlx/ tests/ convert.py

typecheck:
    uv run basedpyright canvit_mlx/

# Enforce: canvit_mlx/ must not import canvit (PyTorch reference)
no-canvit-dep:
    @! grep -rn 'import canvit\b\|from canvit[^_]' canvit_mlx/ --include='*.py' \
        && echo "OK: canvit_mlx has no canvit imports" \
        || (echo "FAIL: canvit_mlx must not import canvit" && exit 1)

test:
    uv run pytest

convert:
    uv run python convert.py

tree:
    uv run pypatree canvit_mlx
