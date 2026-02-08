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
bench
├── plot  Plot benchmark results from bench/<timestamp>/results.parquet.
│   ├── find_latest_parquet() -> pathlib.Path
│   ├── make_plot(df: polars.dataframe.frame.DataFrame, out_dir: pathlib.Path, *, show_bf16: bool = True, clip_to_canvit: bool = True, log_scale: bool = False, show_precision: bool = False) -> None
│   ├── __main__  Entry point: uv run python -m bench.plot .
│   │   └── main() -> None
│   ├── breakdown  Stacked component breakdown: weighted MLX time per component.
│   │   └── plot_breakdown(ax: matplotlib.axes._axes.Axes, df: polars.dataframe.frame.DataFrame, *, target_pxs: tuple[int, ...] | None = None) -> None
│   ├── ci  Bootstrap confidence interval helpers.
│   │   ├── bootstrap_ci(samples: list[float]) -> tuple[float, float, float]
│   │   └── ci_ms(df: polars.dataframe.frame.DataFrame, grid: int, component: str, raw_col: str, med_col: str) -> tuple[float, float, float]
│   ├── forward  Full forward timing: CanViT (PT/MLX) vs teacher baselines.
│   │   └── plot_forward(ax: matplotlib.axes._axes.Axes, df: polars.dataframe.frame.DataFrame, *, show_bf16: bool = False, clip_to_canvit: bool = True, log_scale: bool = False) -> None
│   ├── memory  Memory vs scene resolution — two panels: MLX (peak) and MPS (no inference_mode).
│   │   └── plot_memory(axes: tuple[matplotlib.axes._axes.Axes, matplotlib.axes._axes.Axes], df: polars.dataframe.frame.DataFrame) -> None
│   └── precision  bf16 vs fp32 precision loss bar chart.
│       └── plot_precision(ax: matplotlib.axes._axes.Axes, df: polars.dataframe.frame.DataFrame) -> None
├── run_latency  Component-level latency benchmark: PT vs MLX (fp32 + bf16).
│   ├── Row(name: str, count: int, pt: bench.run_latency.Stats, mlx: bench.run_latency.Stats, rel_err: float, bf16: bench.run_latency.Stats | None = None, bf16_errs: dict[str, float] | None = None) -> 
│   │   None
│   ├── Stats(med: float, mn: float, mx: float, mean: float, raw_us: list[float]) -> None
│   ├── bench_at_grid(pt_m, mlx_m, mlx_m_bf16, cfg, device, canvas_grid, image_px, image_mlx: mlx.core.array, n_patches, n_local, pt_rope, pt_periods, mlx_rope, mlx_periods, pt_sample, mlx_extract, 
│   │   PTVP, MLXVP) -> list[bench.run_latency.Row]
│   ├── bench_teacher(name: str, model: torch.nn.modules.module.Module, repo: str, pil_image, image_px: int, device: torch.device, pt_sync, *, model_bf: torch.nn.modules.module.Module | None = None) ->
│   │   bench.run_latency.Row
│   ├── check(name: str, ref: numpy.ndarray, got: numpy.ndarray) -> float
│   ├── check_and_bench(name: str, count: int, pt_fn, mlx_fn, pt_sync, mlx_sync) -> bench.run_latency.Row
│   ├── main(grids: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 28, 32, 40, 48, 56, 64, 128), pt_device: str = 'mps', dinov3_bf16: bool = False, teacher_max_px:
│   │   int = 384, out_dir: str | None = None) -> None
│   ├── print_table(rows: list[bench.run_latency.Row]) -> None
│   ├── save_pca_and_artifacts(run_dir: pathlib.Path, mlx_m, mlx_m_bf16, image_mlx, cfg, canvas_grid: int, MLXVP, mlx_extract)
│   ├── save_results(run_dir: pathlib.Path, all_results: dict[int, list[bench.run_latency.Row]], cfg, pt_device: str) -> pathlib.Path
│   └── time_fn(fn, sync) -> bench.run_latency.Stats
├── run_memory  Memory benchmark: MLX peak vs MPS current_allocated (absolute).
│   └── main(grids: tuple[int, ...] = (8, 16, 32, 64, 128), teacher_max_px: int = 1024, out_dir: str | None = None) -> None
└── shared  Shared constants and utilities for bench scripts.
    ├── git_sha() -> str
    └── make_pt_sync(device: torch.device)
canvit_mlx  CanViT inference on Apple Silicon via MLX.
├── canvas_attention  Canvas cross-attention: local reads from / writes to the persistent canvas.
│   ├── CanvasReadAttention(local_dim: int, canvas_dim: int, num_heads: int)
│   └── CanvasWriteAttention(local_dim: int, canvas_dim: int, num_heads: int)
├── canvit  CanViT model: recurrent vision transformer with canvas memory.
│   ├── CanViT(cfg: canvit_mlx.config.CanViTConfig)
│   ├── CanViTOutput(state: canvit_mlx.canvit.RecurrentState, ephemeral_cls: mlx.core.array, local_patches: mlx.core.array) -> None
│   └── RecurrentState(canvas: mlx.core.array, recurrent_cls: mlx.core.array) -> None
├── checkpoint  Load a CanViT model from safetensors checkpoint.
│   └── load_canvit(weights_path: str, config_path: str | None = None) -> canvit_mlx.canvit.CanViT
├── config  Model hyperparameters and architecture configuration.
│   └── CanViTConfig(embed_dim: int, num_heads: int, n_blocks: int, patch_size: int, ffn_ratio: float, n_register_tokens: int, rw_stride: int, n_canvas_registers: int, canvas_num_heads: int, 
│       canvas_head_dim: int, enable_vpe: bool, teacher_dim: int, std_grid_size: int) -> None
├── glimpse  Bilinear glimpse extraction from an image at a given viewpoint.
│   └── extract_glimpse_at_viewpoint(image: mlx.core.array, viewpoint: canvit_mlx.viewpoint.Viewpoint, glimpse_size_px: int) -> mlx.core.array
├── grid  Normalized coordinate grids in [-1, 1], (y, x) order.
│   ├── canvas_coords_for_glimpse(center: mlx.core.array, scale: mlx.core.array, H: int, W: int) -> mlx.core.array
│   └── grid_coords(H: int, W: int) -> mlx.core.array
├── layer_scale  Per-channel learned scaling.
│   └── LayerScale(dim: int)
├── patch_embed  Image-to-patch-tokens via strided convolution.
│   └── PatchEmbed(patch_size: int, embed_dim: int)
├── preprocess  Image preprocessing: resize shortest edge, center crop, ImageNet normalize.
│   └── load_and_preprocess(path: str, target_size: int) -> mlx.core.array
├── rope  2D Rotary Position Embeddings.
│   ├── apply_rope_with_prefix(x: mlx.core.array, sin: mlx.core.array, cos: mlx.core.array) -> mlx.core.array
│   ├── compute_rope(positions: mlx.core.array, periods: mlx.core.array) -> tuple[mlx.core.array, mlx.core.array]
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
