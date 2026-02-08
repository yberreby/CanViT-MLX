# CanViT-MLX

MLX inference port of [CanViT](https://github.com/yberreby/CanViT) for Apple Silicon.

## Quickstart

```bash
uv run python convert.py          # HF Hub → MLX-native safetensors (~5s)
uv run just                        # lint + typecheck + dep enforcement + tests
```

```python
from canvit_mlx import load_canvit, load_and_preprocess, Viewpoint, extract_glimpse_at_viewpoint

image = load_and_preprocess("test_data/Cat03.jpg", target_size=512)
model = load_canvit("weights/canvit-vitb16-pretrain-512px-in21k.safetensors")
state = model.init_state(batch_size=1, canvas_grid_size=32)
vp = Viewpoint.full_scene(batch_size=1)
glimpse = extract_glimpse_at_viewpoint(image, vp, glimpse_size_px=128)
output = model(glimpse, state, vp)
```

## Module tree

```
canvit_mlx  CanViT inference on Apple Silicon via MLX.
├── canvas_attention  Canvas cross-attention: local reads from / writes to the 
│   persistent canvas.
│   ├── CanvasReadAttention(local_dim: int, canvas_dim: int, num_heads: int)
│   └── CanvasWriteAttention(local_dim: int, canvas_dim: int, num_heads: int)
├── canvit  CanViT model: recurrent vision transformer with canvas memory.
│   ├── CanViT(cfg: canvit_mlx.config.CanViTConfig)
│   ├── CanViTOutput(state: canvit_mlx.canvit.RecurrentState, ephemeral_cls: 
│   │   mlx.core.array, local_patches: mlx.core.array) -> None
│   └── RecurrentState(canvas: mlx.core.array, recurrent_cls: mlx.core.array) ->
│       None
├── checkpoint  Load a CanViT model from safetensors checkpoint.
│   └── load_canvit(weights_path: str, config_path: str | None = None) -> 
│       canvit_mlx.canvit.CanViT
├── config  Model hyperparameters and architecture configuration.
│   └── CanViTConfig(embed_dim: int, num_heads: int, n_blocks: int, patch_size: 
│       int, ffn_ratio: float, n_register_tokens: int, rw_stride: int, 
│       n_canvas_registers: int, canvas_num_heads: int, canvas_head_dim: int, 
│       enable_vpe: bool, teacher_dim: int, std_grid_size: int) -> None
├── glimpse  Bilinear glimpse extraction from an image at a given viewpoint.
│   └── extract_glimpse_at_viewpoint(image: mlx.core.array, viewpoint: 
│       canvit_mlx.viewpoint.Viewpoint, glimpse_size_px: int) -> mlx.core.array
├── grid  Normalized coordinate grids in [-1, 1], (y, x) order.
│   ├── canvas_coords_for_glimpse(center: mlx.core.array, scale: mlx.core.array,
│   │   H: int, W: int) -> mlx.core.array
│   └── grid_coords(H: int, W: int) -> mlx.core.array
├── layer_scale  Per-channel learned scaling.
│   └── LayerScale(dim: int)
├── patch_embed  Image-to-patch-tokens via strided convolution.
│   └── PatchEmbed(patch_size: int, embed_dim: int)
├── preprocess  Image preprocessing: resize shortest edge, center crop, ImageNet
│   normalize.
│   └── load_and_preprocess(path: str, target_size: int) -> mlx.core.array
├── rope  2D Rotary Position Embeddings.
│   ├── apply_rope_with_prefix(x: mlx.core.array, sin: mlx.core.array, cos: 
│   │   mlx.core.array) -> mlx.core.array
│   ├── compute_rope(positions: mlx.core.array, periods: mlx.core.array) -> 
│   │   tuple[mlx.core.array, mlx.core.array]
│   └── make_rope_periods(head_dim: int, base: float = 100.0) -> mlx.core.array
├── viewpoint  Viewpoint: where the model is looking (center + scale).
│   └── Viewpoint(centers: mlx.core.array, scales: mlx.core.array) -> None
├── vit_block  ViT transformer block: self-attention + MLP with LayerScale.
│   ├── MLP(dim: int, hidden_dim: int)
│   ├── SelfAttention(dim: int, num_heads: int)
│   └── ViTBlock(dim: int, num_heads: int, ffn_ratio: float)
└── vpe  Viewpoint Position Encoding via random Fourier features.
    └── VPEEncoder(rff_dim: int)
```

## License

[MIT](LICENSE)
