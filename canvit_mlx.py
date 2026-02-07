"""CanViT inference in MLX.

Pure MLX reimplementation of CanViT (dual-stream ViT with canvas cross-attention).
Loads weights from safetensors exported by convert.py.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.core.fast
import mlx.nn as nn

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CanViTConfig:
    # Backbone
    embed_dim: int = 768
    num_heads: int = 12
    n_blocks: int = 12
    patch_size: int = 16
    ffn_ratio: float = 4.0
    n_register_tokens: int = 4

    # Canvas
    rw_stride: int = 2
    n_canvas_registers: int = 16
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    enable_vpe: bool = True

    # Pretraining heads
    teacher_dim: int = 768

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


# ---------------------------------------------------------------------------
# 2D RoPE (custom, matching CanViT's implementation)
# ---------------------------------------------------------------------------

def make_rope_periods(head_dim: int, base: float = 100.0) -> mx.array:
    n_freqs = head_dim // 4
    exponents = mx.arange(n_freqs) / n_freqs
    return base ** exponents


def compute_rope(positions: mx.array, periods: mx.array) -> tuple[mx.array, mx.array]:
    """Compute 2D RoPE sin/cos from positions [B, N, 2] and periods [n_freqs].

    Returns sin, cos each [B, 1, N, head_dim].
    """
    # angles[b, n, dim, freq] = 2π * pos / period
    angles = 2 * math.pi * mx.expand_dims(positions, -1) / periods
    # Flatten [y_freqs, x_freqs] -> [y0..yn, x0..xn]
    B, N, _two, _nf = angles.shape
    angles = angles.reshape(B, N, -1)
    # Tile for rotation pairs: [y, x, y, x]
    angles = mx.concatenate([angles, angles], axis=-1)

    sin = mx.sin(angles)
    cos = mx.cos(angles)
    # Add head dim: [B, 1, N, head_dim]
    return mx.expand_dims(sin, 1), mx.expand_dims(cos, 1)


def rope_apply_with_prefix(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Apply 2D RoPE to spatial tokens, leaving prefix tokens unchanged.

    x: [B, H, N_total, head_dim]
    sin, cos: [B, 1, N_spatial, head_dim]
    """
    n_prefix = x.shape[2] - sin.shape[2]
    half = x.shape[3] // 2

    if n_prefix > 0:
        prefix = x[:, :, :n_prefix]
    x_s = x[:, :, n_prefix:]

    x1 = x_s[..., :half]
    x2 = x_s[..., half:]
    cos1, cos2 = cos[..., :half], cos[..., half:]
    sin1, sin2 = sin[..., :half], sin[..., half:]

    rot1 = x1 * cos1 - x2 * sin1
    rot2 = x2 * cos2 + x1 * sin2
    spatial = mx.concatenate([rot1, rot2], axis=-1)

    if n_prefix > 0:
        return mx.concatenate([prefix, spatial], axis=2)
    return spatial


# ---------------------------------------------------------------------------
# DINOv3-style RoPE (rotate-half style, used inside backbone self-attention)
# ---------------------------------------------------------------------------

def rope_rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def dinov3_rope_apply(x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
    """Apply RoPE with rotate-half convention (DINOv3 style)."""
    return x * cos + rope_rotate_half(x) * sin


def dinov3_rope_apply_with_prefix(
    x: mx.array, sin: mx.array, cos: mx.array
) -> mx.array:
    """Apply DINOv3 RoPE to spatial tokens, skip prefix."""
    n_prefix = x.shape[2] - sin.shape[2]
    if n_prefix > 0:
        prefix = x[:, :, :n_prefix]
        spatial = dinov3_rope_apply(x[:, :, n_prefix:], sin, cos)
        return mx.concatenate([prefix, spatial], axis=2)
    return dinov3_rope_apply(x, sin, cos)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """2D image to patch embedding using Conv2d."""

    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, int, int]:
        """x: [B, H, W, C] (MLX uses channels-last). Returns (tokens [B, N, D], grid_h, grid_w)."""
        x = self.proj(x)  # [B, H', W', D]
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D), H, W


class LayerScale(nn.Module):
    """Elementwise scale: y = gamma * x. gamma is a 1D vector."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class ReparamLayerScale(nn.Module):
    """Reparameterized LayerScale: y = (init_scale + delta_scale) * x."""

    def __init__(self, dim: int):
        super().__init__()
        self.init_scale = mx.ones((dim,))
        self.delta_scale = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * (self.init_scale + self.delta_scale)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class SelfAttention(nn.Module):
    """Multi-head self-attention with fused QKV and RoPE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        B, N, D = x.shape
        qkv = self.qkv(x)  # [B, N, 3*D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # [B, N, H, hd] -> [B, H, N, hd]
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q = dinov3_rope_apply_with_prefix(q, sin, cos)
        k = dinov3_rope_apply_with_prefix(k, sin, cos)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, D)
        return self.proj(out)


class ViTBlock(nn.Module):
    """Standard ViT block: LN -> SelfAttn -> LS -> residual -> LN -> MLP -> LS -> residual."""

    def __init__(self, dim: int, num_heads: int, ffn_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        x = x + self.ls1(self.attn(self.norm1(x), sin, cos))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Canvas Attention
# ---------------------------------------------------------------------------

def to_multihead(x: mx.array, num_heads: int) -> mx.array:
    """[B, N, D] -> [B, H, N, hd]"""
    B, N, D = x.shape
    return x.reshape(B, N, num_heads, D // num_heads).transpose(0, 2, 1, 3)


def from_multihead(x: mx.array) -> mx.array:
    """[B, H, N, hd] -> [B, N, D]"""
    B, H, N, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * hd)


class CanvasReadAttention(nn.Module):
    """Local queries canvas. Dense projections on local side, identity on canvas."""

    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(local_dim)
        self.pre_kv_ln = nn.LayerNorm(canvas_dim)
        self.q_transform = nn.Linear(local_dim, canvas_dim)
        self.out_transform = nn.Linear(canvas_dim, local_dim)

    def __call__(
        self,
        query: mx.array,
        kv: mx.array,
        query_sin: mx.array, query_cos: mx.array,
        kv_sin: mx.array, kv_cos: mx.array,
    ) -> mx.array:
        q = to_multihead(self.q_transform(self.pre_q_ln(query)), self.num_heads)
        kv_normed = self.pre_kv_ln(kv)
        k = to_multihead(kv_normed, self.num_heads)
        v = to_multihead(kv_normed, self.num_heads)

        q = rope_apply_with_prefix(q, query_sin, query_cos)
        k = rope_apply_with_prefix(k, kv_sin, kv_cos)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_transform(from_multihead(out))


class CanvasWriteAttention(nn.Module):
    """Canvas queries local. Dense projections on local side, identity on canvas."""

    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(canvas_dim)
        self.pre_kv_ln = nn.LayerNorm(local_dim)
        self.k_transform = nn.Linear(local_dim, canvas_dim)
        self.v_transform = nn.Linear(local_dim, canvas_dim)

    def __call__(
        self,
        query: mx.array,
        kv: mx.array,
        query_sin: mx.array, query_cos: mx.array,
        kv_sin: mx.array, kv_cos: mx.array,
    ) -> mx.array:
        q = to_multihead(self.pre_q_ln(query), self.num_heads)
        kv_normed = self.pre_kv_ln(kv)
        k = to_multihead(self.k_transform(kv_normed), self.num_heads)
        v = to_multihead(self.v_transform(kv_normed), self.num_heads)

        q = rope_apply_with_prefix(q, query_sin, query_cos)
        k = rope_apply_with_prefix(k, kv_sin, kv_cos)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return from_multihead(out)


class ScaledResidualRead(nn.Module):
    """Read attention with reparameterized LayerScale gating."""

    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.attn = CanvasReadAttention(local_dim, canvas_dim, num_heads)
        self.scale = ReparamLayerScale(local_dim)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array,
                 kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        return query + self.scale(self.attn(query, kv, q_sin, q_cos, kv_sin, kv_cos))


class ResidualWrite(nn.Module):
    """Write attention with ungated residual."""

    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.attn = CanvasWriteAttention(local_dim, canvas_dim, num_heads)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array,
                 kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        return query + self.attn(query, kv, q_sin, q_cos, kv_sin, kv_cos)


# ---------------------------------------------------------------------------
# VPE Encoder
# ---------------------------------------------------------------------------

class VPEEncoder(nn.Module):
    """Viewpoint Positional Encoding via Random Fourier Features."""

    def __init__(self, rff_dim: int):
        super().__init__()
        self.rff_dim = rff_dim
        self.B_mat = mx.zeros((rff_dim // 2, 3))  # loaded from weights
        self.norm = nn.LayerNorm(rff_dim)

    def __call__(self, y: mx.array, x: mx.array, s: mx.array) -> mx.array:
        z = mx.stack([y / s, x / s, mx.log(s)], axis=-1)
        proj = z @ self.B_mat.T
        enc = mx.concatenate([mx.cos(proj), mx.sin(proj)], axis=-1)
        return self.norm(enc)


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def grid_coords(H: int, W: int) -> mx.array:
    """Normalized [-1, 1] grid coordinates. Returns [H, W, 2] with (y, x)."""
    y = (mx.arange(H) + 0.5) / H * 2 - 1
    x = (mx.arange(W) + 0.5) / W * 2 - 1
    yy, xx = mx.meshgrid(y, x, indexing="ij")
    return mx.stack([yy, xx], axis=-1)


def canvas_coords_for_glimpse(
    center: mx.array, scale: mx.array, H: int, W: int
) -> mx.array:
    """Canvas coordinates for glimpse grid. Returns [B, H*W, 2]."""
    B = center.shape[0]
    retino = grid_coords(H, W)  # [H, W, 2]
    retino = retino.reshape(1, H * W, 2)
    center = center.reshape(B, 1, 2)
    scale = scale.reshape(B, 1, 1)
    return center + scale * retino


# ---------------------------------------------------------------------------
# Viewpoint & sampling
# ---------------------------------------------------------------------------

@dataclass
class Viewpoint:
    centers: mx.array  # [B, 2]
    scales: mx.array   # [B]

    @staticmethod
    def full_scene(batch_size: int) -> "Viewpoint":
        return Viewpoint(
            centers=mx.zeros((batch_size, 2)),
            scales=mx.ones((batch_size,)),
        )


def sample_at_viewpoint(
    image: mx.array, viewpoint: Viewpoint, glimpse_size_px: int
) -> mx.array:
    """Bilinear sampling of glimpse from image at viewpoint.

    image: [B, H, W, C] (channels-last for MLX)
    Returns: [B, gH, gW, C]
    """
    B, H_img, W_img, C = image.shape
    gH = gW = glimpse_size_px

    # Compute sampling grid in pixel coordinates
    coords = grid_coords(gH, gW)  # [gH, gW, 2], values in [-1, 1]
    coords = coords.reshape(1, gH, gW, 2)

    centers = viewpoint.centers.reshape(B, 1, 1, 2)
    scales = viewpoint.scales.reshape(B, 1, 1, 1)
    grid = centers + scales * coords  # [B, gH, gW, 2] in [-1, 1]

    # Convert from [-1, 1] to pixel coordinates (align_corners=False convention)
    # pixel = (coord + 1) / 2 * size - 0.5
    grid_y = (grid[..., 0] + 1) / 2 * H_img - 0.5  # [B, gH, gW]
    grid_x = (grid[..., 1] + 1) / 2 * W_img - 0.5

    # Bilinear interpolation
    y0 = mx.floor(grid_y).astype(mx.int32)
    x0 = mx.floor(grid_x).astype(mx.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = grid_y - y0.astype(mx.float32)
    wx = grid_x - x0.astype(mx.float32)

    # Clamp to valid range
    y0 = mx.clip(y0, 0, H_img - 1)
    y1 = mx.clip(y1, 0, H_img - 1)
    x0 = mx.clip(x0, 0, W_img - 1)
    x1 = mx.clip(x1, 0, W_img - 1)

    # Gather pixels - need to index [B, H, W, C] with [B, gH, gW] indices
    # Flatten spatial dims for take_along_axis
    def gather(yy: mx.array, xx: mx.array) -> mx.array:
        idx = yy * W_img + xx  # [B, gH, gW]
        flat_img = image.reshape(B, H_img * W_img, C)  # [B, H*W, C]
        idx_flat = idx.reshape(B, gH * gW)  # [B, gH*gW]
        idx_exp = mx.expand_dims(idx_flat, -1)  # [B, gH*gW, 1]
        idx_exp = mx.broadcast_to(idx_exp, (B, gH * gW, C))
        gathered = mx.take_along_axis(flat_img, idx_exp, axis=1)  # [B, gH*gW, C]
        return gathered.reshape(B, gH, gW, C)

    p00 = gather(y0, x0)
    p01 = gather(y0, x1)
    p10 = gather(y1, x0)
    p11 = gather(y1, x1)

    wy = mx.expand_dims(wy, -1)
    wx = mx.expand_dims(wx, -1)

    result = (
        p00 * (1 - wy) * (1 - wx) +
        p01 * (1 - wy) * wx +
        p10 * wy * (1 - wx) +
        p11 * wy * wx
    )
    return result


# ---------------------------------------------------------------------------
# CanViT Model
# ---------------------------------------------------------------------------

def compute_rw_positions(n_blocks: int, rw_stride: int) -> tuple[list[int], list[int]]:
    rw_positions = list(range(rw_stride - 1, n_blocks, rw_stride))
    read_after = []
    write_after = []
    for i, pos in enumerate(rw_positions):
        if i % 2 == 0:
            read_after.append(pos)
        else:
            write_after.append(pos)
    last_block = n_blocks - 1
    if not write_after or write_after[-1] != last_block:
        write_after.append(last_block)
    return read_after, write_after


@dataclass
class RecurrentState:
    canvas: mx.array      # [B, n_regs + G², canvas_dim]
    recurrent_cls: mx.array  # [B, 1, local_dim]


@dataclass
class CanViTOutput:
    state: RecurrentState
    ephemeral_cls: mx.array
    local_patches: mx.array


class CanViT(nn.Module):
    def __init__(self, cfg: CanViTConfig):
        super().__init__()
        self.cfg = cfg

        # Backbone
        self.patch_embed = PatchEmbed(cfg.patch_size, 3, cfg.embed_dim)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        self.storage_tokens = mx.zeros((1, cfg.n_register_tokens, cfg.embed_dim))
        self.backbone_norm = nn.LayerNorm(cfg.embed_dim)
        self.rope_periods_backbone = mx.zeros((cfg.head_dim // 4,))

        self.blocks = [ViTBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_ratio) for _ in range(cfg.n_blocks)]

        # Canvas attention
        read_after, write_after = compute_rw_positions(cfg.n_blocks, cfg.rw_stride)
        self.read_after_blocks = read_after
        self.write_after_blocks = write_after

        self.read_attn = [
            ScaledResidualRead(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads)
            for _ in range(len(read_after))
        ]
        self.write_attn = [
            ResidualWrite(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads)
            for _ in range(len(write_after))
        ]

        # Canvas init params
        self.canvas_register_init = mx.zeros((1, cfg.n_canvas_registers, cfg.canvas_dim))
        self.canvas_spatial_init = mx.zeros((1, 1, cfg.canvas_dim))
        self.recurrent_cls_init = mx.zeros((1, 1, cfg.embed_dim))

        # VPE
        self.vpe_encoder = VPEEncoder(cfg.embed_dim) if cfg.enable_vpe else None

        # Pretraining heads
        self.scene_patches_ln = nn.LayerNorm(cfg.canvas_dim)
        self.scene_patches_proj = nn.Linear(cfg.canvas_dim, cfg.teacher_dim)
        self.scene_cls_ln = nn.LayerNorm(cfg.embed_dim)
        self.scene_cls_proj = nn.Linear(cfg.embed_dim, cfg.teacher_dim)

        # Standardizers
        self.cls_std_mean = mx.zeros((1, cfg.teacher_dim))
        self.cls_std_var = mx.ones((1, cfg.teacher_dim))
        self.scene_std_mean = mx.zeros((1024, cfg.teacher_dim))
        self.scene_std_var = mx.ones((1024, cfg.teacher_dim))

    def init_state(self, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        n_spatial = canvas_grid_size ** 2
        regs = mx.broadcast_to(self.canvas_register_init, (batch_size, self.cfg.n_canvas_registers, self.cfg.canvas_dim))
        spatial = mx.broadcast_to(self.canvas_spatial_init, (batch_size, n_spatial, self.cfg.canvas_dim))
        canvas = mx.concatenate([regs, spatial], axis=1)
        cls = mx.broadcast_to(self.recurrent_cls_init, (batch_size, 1, self.cfg.embed_dim))
        return RecurrentState(canvas=canvas, recurrent_cls=cls)

    def __call__(
        self,
        glimpse: mx.array,
        state: RecurrentState,
        viewpoint: Viewpoint,
    ) -> CanViTOutput:
        """Forward pass.

        glimpse: [B, gH, gW, C] channels-last image crop
        """
        B = glimpse.shape[0]
        cfg = self.cfg
        canvas = state.canvas
        recurrent_cls = state.recurrent_cls

        # Patch embed
        patches, H, W = self.patch_embed(glimpse)
        n_patches = H * W

        # Build token sequence: [vpe?, recurrent_cls, ephemeral_cls, registers, patches]
        ephemeral_cls = mx.broadcast_to(self.cls_token, (B, 1, cfg.embed_dim))
        registers = mx.broadcast_to(self.storage_tokens, (B, cfg.n_register_tokens, cfg.embed_dim))

        parts = []
        has_vpe = self.vpe_encoder is not None
        if has_vpe:
            assert self.vpe_encoder is not None
            vpe_tok = self.vpe_encoder(
                viewpoint.centers[:, 0],
                viewpoint.centers[:, 1],
                viewpoint.scales,
            )
            parts.append(mx.expand_dims(vpe_tok, 1))

        parts.extend([recurrent_cls, ephemeral_cls, registers, patches])
        local = mx.concatenate(parts, axis=1)

        n_prefix = (1 if has_vpe else 0) + 1 + 1 + cfg.n_register_tokens

        # Compute positions for glimpse patches in canvas coordinate space
        local_pos = canvas_coords_for_glimpse(
            viewpoint.centers, viewpoint.scales, H, W
        )  # [B, H*W, 2]

        # Compute RoPE
        backbone_periods = make_rope_periods(cfg.head_dim)
        canvas_periods = make_rope_periods(cfg.canvas_head_dim)

        local_sin_bb, local_cos_bb = compute_rope(local_pos, backbone_periods)
        local_sin_ca, local_cos_ca = compute_rope(local_pos, canvas_periods)

        # Canvas spatial RoPE
        n_canvas_spatial = canvas.shape[1] - cfg.n_canvas_registers
        canvas_grid_size = int(math.sqrt(n_canvas_spatial))
        spatial_pos = grid_coords(canvas_grid_size, canvas_grid_size).reshape(1, -1, 2)
        spatial_pos = mx.broadcast_to(spatial_pos, (B, n_canvas_spatial, 2))
        canvas_sin, canvas_cos = compute_rope(spatial_pos, canvas_periods)

        # Run backbone blocks with interleaved canvas attention
        read_idx = 0
        write_idx = 0

        for block_idx in range(cfg.n_blocks):
            local = self.blocks[block_idx](local, local_sin_bb, local_cos_bb)

            if read_idx < len(self.read_after_blocks) and block_idx == self.read_after_blocks[read_idx]:
                local = self.read_attn[read_idx](
                    local, canvas, local_sin_ca, local_cos_ca, canvas_sin, canvas_cos
                )
                read_idx += 1

            if write_idx < len(self.write_after_blocks) and block_idx == self.write_after_blocks[write_idx]:
                canvas = self.write_attn[write_idx](
                    canvas, local, canvas_sin, canvas_cos, local_sin_ca, local_cos_ca
                )
                write_idx += 1

        # Unpack local tokens
        idx = 0
        if has_vpe:
            idx += 1
        new_recurrent_cls = local[:, idx:idx+1]
        idx += 1
        new_ephemeral_cls = local[:, idx:idx+1]
        idx += 1
        idx += cfg.n_register_tokens  # skip registers
        out_patches = local[:, idx:idx+n_patches]

        new_state = RecurrentState(canvas=canvas, recurrent_cls=new_recurrent_cls)
        return CanViTOutput(
            state=new_state,
            ephemeral_cls=new_ephemeral_cls,
            local_patches=out_patches,
        )

    def predict_teacher_scene(self, canvas: mx.array) -> mx.array:
        spatial = canvas[:, self.cfg.n_canvas_registers:]
        return self.scene_patches_proj(self.scene_patches_ln(spatial))

    def predict_scene_teacher_cls(self, global_cls: mx.array) -> mx.array:
        return self.scene_cls_proj(self.scene_cls_ln(global_cls[:, 0]))

    def destandardize_cls(self, x: mx.array) -> mx.array:
        eps = 1e-6
        return x * mx.sqrt(self.cls_std_var + eps) + self.cls_std_mean


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_canvit(weights_path: str, config_path: str | None = None) -> CanViT:
    """Load CanViT from safetensors weights + JSON config."""
    weights_path = Path(weights_path)
    if config_path is None:
        config_path_p = weights_path.with_suffix(".json")
    else:
        config_path_p = Path(config_path)

    meta = json.loads(config_path_p.read_text())
    mc = meta["model_config"]

    # Determine backbone params from backbone_name
    backbone_name = meta["backbone_name"]
    assert backbone_name == "dinov3_vitb16", f"Only dinov3_vitb16 supported, got {backbone_name}"

    cfg = CanViTConfig(
        embed_dim=768,
        num_heads=12,
        n_blocks=12,
        patch_size=16,
        ffn_ratio=4.0,
        n_register_tokens=4,
        rw_stride=mc["rw_stride"],
        n_canvas_registers=mc["n_canvas_registers"],
        canvas_num_heads=mc["canvas_num_heads"],
        canvas_head_dim=mc["canvas_head_dim"],
        enable_vpe=mc["enable_vpe"],
        teacher_dim=mc["teacher_dim"],
    )

    model = CanViT(cfg)

    # Load safetensors
    from safetensors import safe_open

    tensors: dict[str, mx.array] = {}
    with safe_open(str(weights_path), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = mx.array(f.get_tensor(key))

    # Map PyTorch keys to MLX model structure
    weights: dict[str, mx.array] = {}

    for key, val in tensors.items():
        mlx_key = _map_key(key)
        if mlx_key is not None:
            weights[mlx_key] = val

    # Add rope periods
    weights["rope_periods_backbone"] = tensors["backbone.vit.rope_embed.periods"]

    model.load_weights(list(weights.items()))
    return model


def _map_key(pt_key: str) -> str | None:
    """Map a PyTorch state_dict key to MLX model attribute path."""

    # Skip buffers we handle separately
    skip_prefixes = [
        "backbone.vit.rope_embed.",
        "backbone.vit.mask_token",
    ]
    for sp in skip_prefixes:
        if pt_key.startswith(sp):
            return None

    # Skip standardizer initialized flags
    if pt_key.endswith("._initialized"):
        return None

    k = pt_key

    # Backbone patch embed
    k = k.replace("backbone.vit.patch_embed.proj.", "patch_embed.proj.")
    # Backbone norm
    k = k.replace("backbone.vit.norm.", "backbone_norm.")
    # Backbone cls/storage tokens
    if k == "backbone.vit.cls_token":
        return "cls_token"
    if k == "backbone.vit.storage_tokens":
        return "storage_tokens"

    # Backbone blocks
    k = k.replace("backbone.vit.blocks.", "blocks.")
    # Within blocks: attn.qkv -> attn.qkv, attn.proj -> attn.proj
    # ls1.gamma -> ls1.gamma, ls2.gamma -> ls2.gamma
    # norm1/norm2 -> norm1/norm2
    # mlp.fc1/fc2 -> mlp.fc1/fc2
    # Remove qkv.bias_mask (not needed for inference)
    if "qkv.bias_mask" in k:
        return None

    # Canvas init params
    if k == "canvas_register_init":
        return "canvas_register_init"
    if k == "canvas_spatial_init":
        return "canvas_spatial_init"
    if k == "recurrent_cls_init":
        return "recurrent_cls_init"

    # Read attention
    # read_attn.0.attn.pre_q_ln.weight -> read_attn.0.attn.pre_q_ln.weight (same)
    # read_attn.0.scale.init_scale -> read_attn.0.scale.init_scale
    # read_attn.0.scale.delta_scale -> read_attn.0.scale.delta_scale

    # Write attention (same structure)

    # VPE encoder
    k = k.replace("vpe_encoder.B", "vpe_encoder.B_mat")

    # Pretraining heads
    k = k.replace("scene_patches_head.0.", "scene_patches_ln.")
    k = k.replace("scene_patches_head.1.", "scene_patches_proj.")
    k = k.replace("scene_cls_head.0.", "scene_cls_ln.")
    k = k.replace("scene_cls_head.1.", "scene_cls_proj.")

    # Standardizers
    if k.startswith("cls_standardizers.32.mean"):
        return "cls_std_mean"
    if k.startswith("cls_standardizers.32.var"):
        return "cls_std_var"
    if k.startswith("scene_standardizers.32.mean"):
        return "scene_std_mean"
    if k.startswith("scene_standardizers.32.var"):
        return "scene_std_var"

    return k
