# CanViT-MLX

@README.md

## Development

```bash
uv run python convert.py   # HF Hub → MLX-native safetensors
uv run just                 # readme + lint + typecheck + tach + tests
```

## Key constraints

- Reference = f32 CPU PyTorch. MPS has known kernel bugs.
- `canvit_mlx` must NOT import `canvit` — only tests and convert.py may.
- MLX `meshgrid` defaults to `indexing='xy'`, PyTorch uses `'ij'`. Always explicit.
- Conv2d weight layout: PyTorch `[O,I,H,W]` → MLX `[O,H,W,I]`.
- Two RoPE styles (mathematically equivalent): rotate-half (backbone) vs explicit sin/cos split (canvas).
- f32 SDPA accumulation error grows with sequence length; canvas atol ~5.0 is justified.
- `convert.py` fuses ReparamLayerScale (init_scale + delta_scale → gamma) and strips residual wrappers.
