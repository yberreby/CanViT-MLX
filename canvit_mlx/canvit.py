__all__ = ["RecurrentState", "CanViTOutput", "compute_rw_positions", "CanViT"]

import logging
import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .vit_block import ViTBlock
from .patch_embed import PatchEmbed
from .canvas_attention import CanvasReadAttention, CanvasWriteAttention
from .config import CanViTConfig
from .grid import canvas_coords_for_glimpse, grid_coords
from .layer_scale import LayerScale
from .rope import compute_rope, make_rope_periods
from .glimpse import extract_glimpse_at_viewpoint
from .viewpoint import Viewpoint
from .vpe import VPEEncoder

log = logging.getLogger(__name__)


@dataclass
class RecurrentState:
    canvas: mx.array      # [B, n_regs + G², canvas_dim]
    recurrent_cls: mx.array  # [B, 1, embed_dim]


@dataclass
class CanViTOutput:
    state: RecurrentState
    ephemeral_cls: mx.array   # [B, 1, embed_dim]
    local_patches: mx.array   # [B, H*W, embed_dim]


def compute_rw_positions(n_blocks: int, rw_stride: int) -> tuple[list[int], list[int]]:
    read_after, write_after = [], []
    for i, pos in enumerate(range(rw_stride - 1, n_blocks, rw_stride)):
        (read_after if i % 2 == 0 else write_after).append(pos)
    if not write_after or write_after[-1] != n_blocks - 1:
        write_after.append(n_blocks - 1)
    return read_after, write_after


class CanViT(nn.Module):
    def __init__(self, cfg: CanViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg.patch_size, cfg.embed_dim)
        self.cls_token = mx.zeros((1, 1, cfg.embed_dim))
        self.storage_tokens = mx.zeros((1, cfg.n_register_tokens, cfg.embed_dim))
        self.blocks = [ViTBlock(cfg.embed_dim, cfg.num_heads, cfg.ffn_ratio) for _ in range(cfg.n_blocks)]

        read_after, write_after = compute_rw_positions(cfg.n_blocks, cfg.rw_stride)
        self.read_after_blocks = read_after
        self.write_after_blocks = write_after
        self.read_attn = [CanvasReadAttention(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads) for _ in read_after]
        self.read_scales = [LayerScale(cfg.embed_dim) for _ in read_after]
        self.write_attn = [CanvasWriteAttention(cfg.embed_dim, cfg.canvas_dim, cfg.canvas_num_heads) for _ in write_after]

        self.canvas_register_init = mx.zeros((1, cfg.n_canvas_registers, cfg.canvas_dim))
        self.canvas_spatial_init = mx.zeros((1, 1, cfg.canvas_dim))
        self.recurrent_cls_init = mx.zeros((1, 1, cfg.embed_dim))

        self.vpe_encoder = VPEEncoder(cfg.embed_dim) if cfg.enable_vpe else None

        self.scene_patches_ln = nn.LayerNorm(cfg.canvas_dim)
        self.scene_patches_proj = nn.Linear(cfg.canvas_dim, cfg.teacher_dim)
        self.scene_cls_ln = nn.LayerNorm(cfg.embed_dim)
        self.scene_cls_proj = nn.Linear(cfg.embed_dim, cfg.teacher_dim)

        n_spatial = cfg.std_grid_size ** 2
        self.cls_std_mean = mx.zeros((1, cfg.teacher_dim))
        self.cls_std_var = mx.ones((1, cfg.teacher_dim))
        self.scene_std_mean = mx.zeros((n_spatial, cfg.teacher_dim))
        self.scene_std_var = mx.ones((n_spatial, cfg.teacher_dim))

        log.info("CanViT: %d blocks, read_after=%s, write_after=%s, vpe=%s",
                 cfg.n_blocks, read_after, write_after, cfg.enable_vpe)

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

        n_prefix = (1 if self.vpe_encoder is not None else 0) + 2 + cfg.n_register_tokens
        expected_local = n_prefix + H * W

        parts: list[mx.array] = []
        if self.vpe_encoder is not None:
            vpe = self.vpe_encoder(viewpoint.centers[:, 0], viewpoint.centers[:, 1], viewpoint.scales)
            parts.append(mx.expand_dims(vpe, 1))
        parts.extend([state.recurrent_cls,
                       mx.broadcast_to(self.cls_token, (B, 1, cfg.embed_dim)),
                       mx.broadcast_to(self.storage_tokens, (B, cfg.n_register_tokens, cfg.embed_dim)),
                       patches])
        local = mx.concatenate(parts, axis=1)
        assert local.shape[1] == expected_local, (
            f"token packing: expected {expected_local}, got {local.shape[1]}")

        local_pos = canvas_coords_for_glimpse(viewpoint.centers, viewpoint.scales, H, W)
        bb_sin, bb_cos = compute_rope(local_pos, make_rope_periods(cfg.head_dim))
        ca_sin, ca_cos = compute_rope(local_pos, make_rope_periods(cfg.canvas_head_dim))
        n_cs = canvas.shape[1] - cfg.n_canvas_registers
        cg = int(math.sqrt(n_cs))
        assert cg * cg == n_cs, f"canvas spatial tokens ({n_cs}) must be a perfect square"
        sp = mx.broadcast_to(grid_coords(cg, cg).reshape(1, -1, 2), (B, n_cs, 2))
        c_sin, c_cos = compute_rope(sp, make_rope_periods(cfg.canvas_head_dim))

        ri, wi = 0, 0
        for bi in range(cfg.n_blocks):
            local = self.blocks[bi](local, bb_sin, bb_cos)
            if ri < len(self.read_after_blocks) and bi == self.read_after_blocks[ri]:
                local = local + self.read_scales[ri](self.read_attn[ri](local, canvas, ca_sin, ca_cos, c_sin, c_cos))
                ri += 1
            if wi < len(self.write_after_blocks) and bi == self.write_after_blocks[wi]:
                canvas = canvas + self.write_attn[wi](canvas, local, c_sin, c_cos, ca_sin, ca_cos)
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
        """Multi-step inference. image [B, H, W, C] -> one output per viewpoint."""
        state = self.init_state(image.shape[0], canvas_grid_size)
        outputs: list[CanViTOutput] = []
        for vp in viewpoints:
            out = self(extract_glimpse_at_viewpoint(image, vp, glimpse_px), state, vp)
            mx.eval(out.state.canvas, out.state.recurrent_cls, out.ephemeral_cls, out.local_patches)
            state = out.state
            outputs.append(out)
        return outputs

    def predict_teacher_scene(self, canvas: mx.array) -> mx.array:
        return self.scene_patches_proj(self.scene_patches_ln(canvas[:, self.cfg.n_canvas_registers:]))

    def predict_scene_teacher_cls(self, global_cls: mx.array) -> mx.array:
        return self.scene_cls_proj(self.scene_cls_ln(global_cls[:, 0]))
