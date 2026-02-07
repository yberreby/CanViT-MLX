# CanViT-MLX Port Log

## 2026-02-07: Initial Port

### Approach
1. Read and understood full CanViT architecture from ~/code/CanViT/
2. Converted PyTorch checkpoint to safetensors (convert.py)
3. Built pure MLX reimplementation (canvit_mlx.py)
4. Systematic module-by-module verification against f32 CPU PyTorch reference

### Architecture Summary
- **Backbone**: DINOv3 ViT-B/16 (12 blocks, 768 dim, 12 heads, 4 registers)
- **Canvas**: 1024 dim (8 heads x 128 head_dim), 16 register tokens + G² spatial tokens
- **Canvas attention**: Asymmetric cross-attention with dense projections only on local (few tokens) side
  - Read: local queries canvas (ScaledResidualAttention with reparameterized LayerScale)
  - Write: canvas queries local (ungated residual)
- **RoPE**: Custom 2D RoPE in [-1,1]² coordinate space, different from DINOv3's rotate-half RoPE
- **Interleaving**: rw_stride=2 → read after blocks [1,5,9], write after blocks [3,7,11]
- **VPE**: Viewpoint Positional Encoding via Random Fourier Features
- **Total**: ~98M params

### Key Decisions & Gotchas

#### CRITICAL: Two different RoPE styles!
- **DINOv3 backbone self-attention**: Uses `rotate_half` convention: `x * cos + rotate_half(x) * sin`
  where rotate_half splits in half and negates second half: `[-x2, x1]`
- **CanViT canvas cross-attention**: Uses explicit sin/cos multiplication with tiling:
  splits into first/second half, applies `(x1*cos1 - x2*sin1, x2*cos2 + x1*sin2)`
- These are NOT the same operation. Getting them confused would produce subtly wrong results.

#### Weight conversion notes
- PyTorch Conv2d weight: `[out_ch, in_ch, kH, kW]` → MLX Conv2d: `[out_ch, kH, kW, in_ch]`
- MLX Linear weight layout matches PyTorch (no transpose needed)
- ytch's `LayerScale` is reparameterized: stores `init_scale` + `delta_scale` (actual = sum)
  - In checkpoint: `read_attn.0.scale.init_scale` and `read_attn.0.scale.delta_scale`
- DINOv3 attention has `qkv.bias_mask` parameter — not needed for inference, skipped
- MLX uses channels-last (NHWC) vs PyTorch channels-first (NCHW)

#### Verification Results (module-by-module, all f32)
- RoPE computation: PASS (max error ~5e-7)
- Patch embedding: PASS (max error ~8e-5)
- VPE encoding: PASS (max error ~1e-7)
- Initial state: PASS (exact match)
- Token packing: PASS (max error ~8e-5, from patch embed)
- Backbone blocks: PASS (max error ~2e-2 at block 11, values ~7000, relative ~2e-6)
- Canvas read attention: PASS (max error ~8e-3, relative ~3e-7)
- Canvas write attention: max abs error ~1.0, BUT relative error ~6e-4
  - Error concentrated in register tokens (highest magnitude values)
  - Proportional to output magnitude (~1300) → normal f32 SDPA accumulation error
  - NOT a bug — verified by checking residual magnitudes match

### BUG FOUND: MLX meshgrid default indexing
**MLX `mx.meshgrid` defaults to `indexing='xy'`** (NumPy convention), while
**PyTorch `torch.meshgrid` with `indexing='ij'`** uses matrix indexing.
This caused `grid_coords` to produce transposed coordinate grids — (x, y) order
instead of (y, x). The end-to-end test initially passed because it used reference
RoPE values from PyTorch, masking the bug. The isolated RoPE test caught it.
**Fix**: `mx.meshgrid(y, x, indexing="ij")`.
**Lesson**: ALWAYS verify coordinate conventions independently. MLX ≠ PyTorch defaults.

### Issues / Potential Pitfalls
- `sample_at_viewpoint`: hand-rolled bilinear interpolation matching F.grid_sample(align_corners=False).
  VERIFIED against PyTorch for full-scene and off-center viewpoints (atol=1e-4).
- Standardizers are grid-size-specific (keyed by "32"). Only grid_size=32 supported.

### RoPE equivalence (VERIFIED)
The two RoPE styles are **mathematically identical** (numerically verified, max diff = 0.0):
- `rotate_half`: `x * cos + [-x2, x1] * sin` → first half `x1*cos - x2*sin`, second half `x2*cos + x1*sin`
- `canvit explicit`: `cat(x1*cos - x2*sin, x2*cos + x1*sin)`

Same rotation, different expression. The CanViT version avoids the `rotate_half` intermediate
allocation, saving one full-tensor copy — matters for the canvas token stream (memory bandwidth bound).

## 2026-02-07: Restructuring

### Package structure
Restructured from monolithic `canvit_mlx.py` (789 lines) to proper hatchling package:
- `canvit_mlx/` — 5 submodules (coords, model, rope, weights, __init__), zero canvit dependency
- `tests/` — separate directory, only place that imports canvit (PyTorch reference)
- `convert.py` — outputs to `weights/canvit-vitb16.safetensors` (not cwd)

### Tooling
- `just` default rule: lint (ruff) → typecheck (basedpyright) → enforce no-canvit-dep → test (pytest)
- `pytest-cov` enabled by default — 100% coverage (339 stmts)
- Dependency enforcement: `just no-canvit-dep` greps canvit_mlx/ for canvit imports, fails if found
- `pypatree` for API review

### End-to-end verification
22 tests covering:
- Unit: RoPE, coordinates, sampling, patch embed, VPE, init state, R/W schedule, weight loading
- Integration: full-scene E2E, off-center viewpoint, 2-step recurrence, run_trajectory, teacher heads
- Infrastructure: convert.py produces loadable weights

### TODO: Clean weight loading
The current `load_canvit()` does PyTorch→MLX key remapping at load time. This is ugly:
conversion logic bleeds into the inference package. The right architecture:
- `convert.py` does ALL key remapping, outputs checkpoint with MLX-native key names
- `load_canvit()` becomes trivial: safe_open → model.load_weights(), zero key surgery
- Future: HF Hub integration for canvit_mlx (download pre-converted weights directly,
  possibly as optional dependency)

This means the .safetensors format will change (breaking), but that's fine — convert.py
regenerates them in ~5s. The key insight: canvit_mlx should stand on its own with a clean,
stable checkpoint format, not have idiosyncrasies that only make sense relative to the
PyTorch reference.

### Tolerance justification
f32 SDPA accumulation error grows with sequence length (~1040 tokens for canvas).
Canvas values reach ~2600 magnitude, giving occasional absolute errors ~3-4 (relative ~0.15%).
Canvas atol=5.0 is justified; other outputs atol=1.0-2.0 with rtol=1e-3 to 2e-3.
