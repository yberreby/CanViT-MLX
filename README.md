# CanViT-MLX

MLX inference port of [CanViT](https://github.com/yberreby/CanViT-PyTorch) for Apple Silicon.

## Quickstart

```bash
# Basic demo: classify image + PCA visualization (downloads weights from HF Hub)
uv run python demos/basic.py

# Real-time webcam PCA (requires opencv-python)
uv run --with opencv-python python demos/realtime_pca.py
```

```python
from canvit_mlx import load_canvit, load_and_preprocess, Viewpoint

model = load_canvit()  # downloads from HF Hub, or pass a local .safetensors path
glimpse = load_and_preprocess("image.jpg", target_size=128)
state = model.init_state(batch_size=1, canvas_grid_size=32)
output = model(glimpse, state, Viewpoint.full_scene(batch_size=1))
# output.state.canvas: [1, 1040, 1024] (16 registers + 1024 spatial tokens)
```

## Weight conversion

Pre-converted MLX weights are hosted on HF Hub at
[`canvit/canvitb16-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx`](https://huggingface.co/canvit/canvitb16-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx).
`load_canvit()` downloads them automatically.

To convert from the PyTorch checkpoint yourself:

```bash
uv run python convert.py           # PT HF Hub → local MLX safetensors
uv run python push_to_hub.py       # upload to HF Hub (optional)
```

## Dependencies

Core (`canvit-mlx`): mlx, numpy, safetensors — no PyTorch, no HF Hub.

HF Hub loading requires `huggingface_hub` (optional extra):
```bash
uv add canvit-mlx[hub]
```

Demos use additional packages (matplotlib, tyro, scikit-learn, etc.) — listed in
the `[dependency-groups] dev` section of `pyproject.toml`.

## Development

```bash
uv run just   # lint + typecheck + tests
```

## License

[MIT](LICENSE)
