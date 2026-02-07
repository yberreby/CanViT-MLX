# CanViT-MLX

MLX inference port of [CanViT](https://github.com/yberreby/CanViT).

## Setup (from clean clone)

```bash
uv run python convert.py          # downloads from HF Hub, creates weights.safetensors
uv run pytest test_modules.py -v  # verify against PyTorch reference (f32 CPU)
```

## Files

- `canvit_mlx.py` — MLX model implementation + weight loading
- `convert.py` — HF Hub → safetensors conversion
- `test_modules.py` — module-by-module correctness tests against PyTorch
- `LOG.md` — chronological development log (bugs, decisions, lessons)

## Key constraints

- Reference = f32 CPU PyTorch. MPS has known kernel bugs.
- MLX `meshgrid` defaults to `indexing='xy'`, PyTorch uses `indexing='ij'`. Always explicit.
- Conv2d weight layout differs: PyTorch `[O,I,H,W]` → MLX `[O,H,W,I]`.
- Two RoPE styles coexist (mathematically equivalent, different perf characteristics):
  DINOv3 rotate-half (backbone) vs explicit sin/cos split (canvas attention).
