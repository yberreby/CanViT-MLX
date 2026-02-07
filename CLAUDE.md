# CanViT-MLX

MLX inference port of [CanViT](https://github.com/yberreby/CanViT).

## Setup

```bash
uv run python convert.py   # HF Hub → MLX-native safetensors
uv run pytest -v            # verify against PyTorch reference (f32 CPU)
```

## Structure

```
canvit_mlx/           # MLX package (NO canvit/PyTorch dependency)
  coords/             # Viewpoint, grid_coords, sample_at_viewpoint
  model/              # CanViT, CanViTConfig, building blocks
  rope/               # 2D RoPE (two equivalent styles)
  weights/            # load_canvit from safetensors
tests/                # all tests (may import canvit for reference)
  conftest.py         # shared fixtures, assert_close, PT helpers
  test_*.py           # unit + integration tests
convert.py            # HF Hub → MLX-native safetensors (key remapping + permutation)
```

## Key constraints

- Reference = f32 CPU PyTorch. MPS has known kernel bugs.
- `canvit_mlx` must NOT import `canvit` — only tests may.
- MLX `meshgrid` defaults to `indexing='xy'`, PyTorch uses `'ij'`. Always explicit.
- Conv2d weight layout: PyTorch `[O,I,H,W]` → MLX `[O,H,W,I]`.
- Two RoPE styles (mathematically equivalent): DINOv3 rotate-half (backbone) vs explicit sin/cos split (canvas).
- f32 SDPA accumulation error grows with sequence length; canvas atol ~5.0 is justified.
