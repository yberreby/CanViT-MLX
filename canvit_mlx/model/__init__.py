"""CanViT: DINOv3 ViT-B/16 backbone with canvas cross-attention."""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..coords import Viewpoint, canvas_coords_for_glimpse, grid_coords, sample_at_viewpoint
from ..rope import apply_dinov3_with_prefix, apply_with_prefix, compute_rope, make_rope_periods


# ---------------------------------------------------------------------------
# Config & output types
# ---------------------------------------------------------------------------

@dataclass
class CanViTConfig:
    embed_dim: int = 768
    num_heads: int = 12
    n_blocks: int = 12
    patch_size: int = 16
    ffn_ratio: float = 4.0
    n_register_tokens: int = 4
    rw_stride: int = 2
    n_canvas_registers: int = 16
    canvas_num_heads: int = 8
    canvas_head_dim: int = 128
    enable_vpe: bool = True
    teacher_dim: int = 768

    @property
    def canvas_dim(self) -> int:
        return self.canvas_num_heads * self.canvas_head_dim

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads


@dataclass
class RecurrentState:
    canvas: mx.array      # [B, n_regs + G², canvas_dim]
    recurrent_cls: mx.array  # [B, 1, embed_dim]


@dataclass
class CanViTOutput:
    state: RecurrentState
    ephemeral_cls: mx.array   # [B, 1, embed_dim]
    local_patches: mx.array   # [B, H*W, embed_dim]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def __call__(self, x: mx.array) -> tuple[mx.array, int, int]:
        x = self.proj(x)
        B, H, W, D = x.shape
        return x.reshape(B, H * W, D), H, W


class LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class ReparamLayerScale(nn.Module):
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
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def __call__(self, x: mx.array, sin: mx.array, cos: mx.array) -> mx.array:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))
        q = apply_dinov3_with_prefix(q, sin, cos)
        k = apply_dinov3_with_prefix(k, sin, cos)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, D))


class ViTBlock(nn.Module):
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
        return x + self.ls2(self.mlp(self.norm2(x)))


# ---------------------------------------------------------------------------
# Canvas attention
# ---------------------------------------------------------------------------

def _to_mh(x: mx.array, h: int) -> mx.array:
    B, N, D = x.shape
    return x.reshape(B, N, h, D // h).transpose(0, 2, 1, 3)

def _from_mh(x: mx.array) -> mx.array:
    B, H, N, hd = x.shape
    return x.transpose(0, 2, 1, 3).reshape(B, N, H * hd)


class CanvasReadAttention(nn.Module):
    """Local queries canvas. Dense projections on local side only."""
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(local_dim)
        self.pre_kv_ln = nn.LayerNorm(canvas_dim)
        self.q_transform = nn.Linear(local_dim, canvas_dim)
        self.out_transform = nn.Linear(canvas_dim, local_dim)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array, kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        q = apply_with_prefix(_to_mh(self.q_transform(self.pre_q_ln(query)), self.num_heads), q_sin, q_cos)
        kv_n = self.pre_kv_ln(kv)
        k = apply_with_prefix(_to_mh(kv_n, self.num_heads), kv_sin, kv_cos)
        v = _to_mh(kv_n, self.num_heads)
        return self.out_transform(_from_mh(mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)))


class CanvasWriteAttention(nn.Module):
    """Canvas queries local. Dense projections on local side only."""
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (canvas_dim // num_heads) ** -0.5
        self.pre_q_ln = nn.LayerNorm(canvas_dim)
        self.pre_kv_ln = nn.LayerNorm(local_dim)
        self.k_transform = nn.Linear(local_dim, canvas_dim)
        self.v_transform = nn.Linear(local_dim, canvas_dim)

    def __call__(self, query: mx.array, kv: mx.array,
                 q_sin: mx.array, q_cos: mx.array, kv_sin: mx.array, kv_cos: mx.array) -> mx.array:
        q = apply_with_prefix(_to_mh(self.pre_q_ln(query), self.num_heads), q_sin, q_cos)
        kv_n = self.pre_kv_ln(kv)
        k = apply_with_prefix(_to_mh(self.k_transform(kv_n), self.num_heads), kv_sin, kv_cos)
        v = _to_mh(self.v_transform(kv_n), self.num_heads)
        return _from_mh(mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale))


class ScaledResidualRead(nn.Module):
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.attn = CanvasReadAttention(local_dim, canvas_dim, num_heads)
        self.scale = ReparamLayerScale(local_dim)

    def __call__(self, *a: mx.array) -> mx.array:
        return a[0] + self.scale(self.attn(*a))


class ResidualWrite(nn.Module):
    def __init__(self, local_dim: int, canvas_dim: int, num_heads: int):
        super().__init__()
        self.attn = CanvasWriteAttention(local_dim, canvas_dim, num_heads)

    def __call__(self, *a: mx.array) -> mx.array:
        return a[0] + self.attn(*a)


# ---------------------------------------------------------------------------
# VPE
# ---------------------------------------------------------------------------

class VPEEncoder(nn.Module):
    def __init__(self, rff_dim: int):
        super().__init__()
        self.B_mat = mx.zeros((rff_dim // 2, 3))
        self.norm = nn.LayerNorm(rff_dim)

    def __call__(self, y: mx.array, x: mx.array, s: mx.array) -> mx.array:
        z = mx.stack([y / s, x / s, mx.log(s)], axis=-1)
        proj = z @ self.B_mat.T
        return self.norm(mx.concatenate([mx.cos(proj), mx.sin(proj)], axis=-1))


# ---------------------------------------------------------------------------
# R/W scheduling
# ---------------------------------------------------------------------------

def compute_rw_positions(n_blocks: int, rw_stride: int) -> tuple[list[int], list[int]]:
    read_after, write_after = [], []
    for i, pos in enumerate(range(rw_stride - 1, n_blocks, rw_stride)):
        (read_after if i % 2 == 0 else write_after).append(pos)
    if not write_after or write_after[-1] != n_blocks - 1:
        write_after.append(n_blocks - 1)
    return read_after, write_after


# ---------------------------------------------------------------------------
# CanViT
# ---------------------------------------------------------------------------

class CanViT(nn.Module):
    def __init__(self, cfg: CanViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.embed_dim)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        self.storage_tokens = mx.zeros((1, cfg.n_register_tokens, cfg.embed_dim))
        self.backbone_norm = nn.LayerNorm(cfg.embed_dim)  # loaded but unused by CanViT
        self.blocks = [ViTBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_ratio) for _ in range(cfg.n_blocks)]

        read_after, write_after = compute_rw_positions(cfg.n_blocks, cfg.rw_stride)
        self.read_after_blocks = read_after
        self.write_after_blocks = write_after
        self.read_attn = [ScaledResidualRead(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads) for _ in read_after]
        self.write_attn = [ResidualWrite(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads) for _ in write_after]

        self.canvas_register_init = mx.zeros((1, cfg.n_canvas_registers, cfg.canvas_dim))
        self.canvas_spatial_init = mx.zeros((1, 1, cfg.canvas_dim))
        self.recurrent_cls_init = mx.zeros((1, 1, cfg.embed_dim))

        self.vpe_encoder = VPEEncoder(cfg.embed_dim) if cfg.enable_vpe else None

        self.scene_patches_ln = nn.LayerNorm(cfg.canvas_dim)
        self.scene_patches_proj = nn.Linear(cfg.canvas_dim, cfg.teacher_dim)
        self.scene_cls_ln = nn.LayerNorm(cfg.embed_dim)
        self.scene_cls_proj = nn.Linear(cfg.embed_dim, cfg.teacher_dim)

        self.cls_std_mean = mx.zeros((1, cfg.teacher_dim))
        self.cls_std_var = mx.ones((1, cfg.teacher_dim))
        self.scene_std_mean = mx.zeros((1024, cfg.teacher_dim))
        self.scene_std_var = mx.ones((1024, cfg.teacher_dim))
        self.rope_periods_backbone = mx.zeros((cfg.head_dim // 4,))  # loaded but unused

    def init_state(self, batch_size: int, canvas_grid_size: int) -> RecurrentState:
        cfg = self.cfg
        n_sp = canvas_grid_size ** 2
        regs = mx.broadcast_to(self.canvas_register_init, (batch_size, cfg.n_canvas_registers, cfg.canvas_dim))
        spatial = mx.broadcast_to(self.canvas_spatial_init, (batch_size, n_sp, cfg.canvas_dim))
        return RecurrentState(
            canvas=mx.concatenate([regs, spatial], axis=1),
            recurrent_cls=mx.broadcast_to(self.recurrent_cls_init, (batch_size, 1, cfg.embed_dim)),
        )

    def __call__(self, glimpse: mx.array, state: RecurrentState, viewpoint: Viewpoint) -> CanViTOutput:
        """Single-step forward. glimpse: [B, gH, gW, C] channels-last."""
        B, cfg = glimpse.shape[0], self.cfg
        canvas = state.canvas
        patches, H, W = self.patch_embed(glimpse)

        parts: list[mx.array] = []
        if self.vpe_encoder is not None:
            vpe = self.vpe_encoder(viewpoint.centers[:, 0], viewpoint.centers[:, 1], viewpoint.scales)
            parts.append(mx.expand_dims(vpe, 1))
        parts.extend([state.recurrent_cls,
                       mx.broadcast_to(self.cls_token, (B, 1, cfg.embed_dim)),
                       mx.broadcast_to(self.storage_tokens, (B, cfg.n_register_tokens, cfg.embed_dim)),
                       patches])
        local = mx.concatenate(parts, axis=1)

        local_pos = canvas_coords_for_glimpse(viewpoint.centers, viewpoint.scales, H, W)
        bb_sin, bb_cos = compute_rope(local_pos, make_rope_periods(cfg.head_dim))
        ca_sin, ca_cos = compute_rope(local_pos, make_rope_periods(cfg.canvas_head_dim))
        n_cs = canvas.shape[1] - cfg.n_canvas_registers
        cg = int(math.sqrt(n_cs))
        sp = mx.broadcast_to(grid_coords(cg, cg).reshape(1, -1, 2), (B, n_cs, 2))
        c_sin, c_cos = compute_rope(sp, make_rope_periods(cfg.canvas_head_dim))

        ri, wi = 0, 0
        for bi in range(cfg.n_blocks):
            local = self.blocks[bi](local, bb_sin, bb_cos)
            if ri < len(self.read_after_blocks) and bi == self.read_after_blocks[ri]:
                local = self.read_attn[ri](local, canvas, ca_sin, ca_cos, c_sin, c_cos)
                ri += 1
            if wi < len(self.write_after_blocks) and bi == self.write_after_blocks[wi]:
                canvas = self.write_attn[wi](canvas, local, c_sin, c_cos, ca_sin, ca_cos)
                wi += 1

        idx = (1 if self.vpe_encoder is not None else 0)
        new_cls = local[:, idx:idx + 1]
        idx += 1
        new_eph = local[:, idx:idx + 1]
        idx += 1 + cfg.n_register_tokens
        return CanViTOutput(
            state=RecurrentState(canvas=canvas, recurrent_cls=new_cls),
            ephemeral_cls=new_eph,
            local_patches=local[:, idx:idx + H * W],
        )

    def run_trajectory(self, image: mx.array, viewpoints: list[Viewpoint],
                       glimpse_px: int, canvas_grid_size: int) -> list[CanViTOutput]:
        """Multi-step inference. image [B, H, W, C] → one output per viewpoint."""
        state = self.init_state(image.shape[0], canvas_grid_size)
        outputs: list[CanViTOutput] = []
        for vp in viewpoints:
            out = self(sample_at_viewpoint(image, vp, glimpse_px), state, vp)
            mx.eval(out.state.canvas, out.state.recurrent_cls, out.ephemeral_cls, out.local_patches)
            state = out.state
            outputs.append(out)
        return outputs

    def predict_teacher_scene(self, canvas: mx.array) -> mx.array:
        return self.scene_patches_proj(self.scene_patches_ln(canvas[:, self.cfg.n_canvas_registers:]))

    def predict_scene_teacher_cls(self, global_cls: mx.array) -> mx.array:
        return self.scene_cls_proj(self.scene_cls_ln(global_cls[:, 0]))
