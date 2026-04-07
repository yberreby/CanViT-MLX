# CanViT-MLX

> **Experimental.** [MLX](https://github.com/ml-explore/mlx) implementation of [CanViT](https://arxiv.org/abs/2603.22570), the Canvas Vision Transformer, for Apple Silicon. Reference implementation: [CanViT-PyTorch](https://github.com/m2b3/CanViT-PyTorch). May break at any time.

_[CanViT: Toward Active-Vision Foundation Models](https://arxiv.org/abs/2603.22570) (arXiv:2603.22570)_

## Install

```bash
uv add "canvit-mlx[hub] @ git+https://github.com/yberreby/CanViT-MLX.git"
```

## Quickstart

```python
import mlx.core as mx
from canvit_mlx import load_from_hf_hub, load_and_preprocess, Viewpoint, extract_glimpse_at_viewpoint

model = load_from_hf_hub("canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-mlx")
image = load_and_preprocess("test_data/Cat03.jpg", target_size=512)

state = model.init_state(batch_size=1, canvas_grid_size=32)
vp = Viewpoint.full_scene(batch_size=1)
glimpse = extract_glimpse_at_viewpoint(image, vp, glimpse_size_px=128)
out = model(glimpse, state, vp)
mx.eval(out.state.canvas, out.state.recurrent_cls, out.local_patches)

# Canvas spatial features (linearly decodable into dense predictions)
canvas_spatial = model.get_spatial(out.state.canvas)  # [1, 1024, 1024]
print(canvas_spatial.shape)
```

## Classification

```python
from pathlib import Path
from canvit_mlx import CanViTForImageClassification, Viewpoint, extract_glimpse_at_viewpoint, load_and_preprocess

clf = CanViTForImageClassification.from_pretrained_with_probe(
    pretrained_weights=Path("weights/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02.safetensors"),
    pretrained_config=Path("weights/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02.json"),
    probe_weights=Path("path/to/probe.safetensors"),
)

image = load_and_preprocess("test_data/Cat03.jpg", target_size=512)
state = clf.init_state(batch_size=1, canvas_grid_size=32)
vp = Viewpoint.full_scene(batch_size=1)
glimpse = extract_glimpse_at_viewpoint(image, vp, glimpse_size_px=128)
logits, new_state = clf(glimpse, state, vp)
```

## Demos

```bash
uv run --group demos python demos/basic.py
uv run --group demos python demos/basic.py --image test_data/Cat03.jpg --canvas-grid 64
```

## Converting weights

Convert a PyTorch checkpoint from HuggingFace Hub to MLX format:

```bash
uv run python convert.py
uv run python convert.py --verify  # includes PT vs MLX numerical comparison
```

## License

[MIT](LICENSE)
