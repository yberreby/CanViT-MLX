"""Model building block tests: patch embed, VPE, init state, block loop."""

import mlx.core as mx
import numpy as np
import torch

from conftest import assert_close, B, CANVAS_GRID, GLIMPSE_PX, SEED


class TestRWPositions:
    def test_default_config(self):
        from canvit_mlx.model import compute_rw_positions
        r, w = compute_rw_positions(12, 2)
        assert r == [1, 5, 9]
        assert w == [3, 7, 11]

    def test_forced_final_write(self):
        from canvit_mlx.model import compute_rw_positions
        r, w = compute_rw_positions(10, 2)
        assert w[-1] == 9  # forced: 7 != 9


class TestPatchEmbed:
    def test_output(self, mlx_model, glimpse_pair, pt_model):
        from canvit.viewpoint import Viewpoint
        glimpse_pt, glimpse_mlx = glimpse_pair
        backbone_tokens, H_pt, W_pt = pt_model.backbone.prepare_tokens(glimpse_pt)
        ref = backbone_tokens[:, 1 + pt_model.backbone.n_register_tokens:].detach().numpy()

        tokens, H, W = mlx_model.patch_embed(glimpse_mlx)
        mx.eval(tokens)
        assert H == H_pt and W == W_pt
        assert_close("patches", ref, tokens, atol=1e-4)


class TestVPE:
    def test_full_scene(self, mlx_model, pt_model):
        from canvit.viewpoint import Viewpoint
        vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
        ref = pt_model.vpe_encoder(y=vp.centers[:, 0], x=vp.centers[:, 1], s=vp.scales).detach().numpy()

        assert mlx_model.vpe_encoder is not None
        got = mlx_model.vpe_encoder(mx.zeros((1,)), mx.zeros((1,)), mx.ones((1,)))
        mx.eval(got)
        assert_close("vpe", ref, got, atol=1e-4)


class TestInitState:
    def test_matches_pytorch(self, mlx_model, pt_model):
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        mx.eval(state_mlx.canvas, state_mlx.recurrent_cls)
        assert_close("init_canvas", state_pt.canvas.detach().numpy(), state_mlx.canvas, atol=1e-6)
        assert_close("init_cls", state_pt.recurrent_cls.detach().numpy(), state_mlx.recurrent_cls, atol=1e-6)


class TestBlockLoop:
    """Full block loop with R/W interleaving, fed from PyTorch-computed inputs."""

    def test_with_ref_inputs(self, mlx_model, pt_model, glimpse_pair):
        from canvit.coords import canvas_coords_for_glimpse, grid_coords
        from canvit.rope import compute_rope, make_rope_periods
        from canvit.viewpoint import Viewpoint
        from canvit.model.base.impl import LocalTokens

        glimpse_pt, _ = glimpse_pair
        vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)

        backbone_tokens, H, W = pt_model.backbone.prepare_tokens(glimpse_pt)
        patches = backbone_tokens[:, 1 + pt_model.backbone.n_register_tokens:]
        cls_token = backbone_tokens[:, 0:1]
        registers = backbone_tokens[:, 1:1 + pt_model.backbone.n_register_tokens]
        vpe = pt_model.vpe_encoder(y=vp.centers[:, 0], x=vp.centers[:, 1], s=vp.scales)
        tokens = LocalTokens(vpe=vpe.unsqueeze(1), recurrent_cls=state_pt.recurrent_cls,
                             ephemeral_cls=cls_token, registers=registers, patches=patches)
        packed = tokens.pack()

        local_pos = canvas_coords_for_glimpse(center=vp.centers, scale=vp.scales, H=H, W=W).flatten(1, 2)
        ca_periods = make_rope_periods(head_dim=pt_model.cfg.canvas_head_dim, device=torch.device("cpu"))
        spatial_pos = grid_coords(H=CANVAS_GRID, W=CANVAS_GRID, device=torch.device("cpu")).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
        c_rope = compute_rope(positions=spatial_pos, periods=ca_periods, dtype=torch.float32)

        # Run PyTorch block loop
        x_pt, canvas_pt = packed, state_pt.canvas
        ri, wi = 0, 0
        with torch.inference_mode():
            for bi in range(pt_model.backbone.n_blocks):
                bb_rope = compute_rope(positions=local_pos, periods=make_rope_periods(head_dim=pt_model.backbone.head_dim, device=torch.device("cpu")), dtype=torch.float32)
                x_pt = pt_model.backbone.forward_block(bi, x_pt, bb_rope)
                if ri < len(pt_model.read_after_blocks) and bi == pt_model.read_after_blocks[ri]:
                    ca_rope = compute_rope(positions=local_pos, periods=ca_periods, dtype=torch.float32)
                    x_pt = pt_model.read_attn[ri](query=x_pt, kv=canvas_pt, query_rope=ca_rope, kv_rope=c_rope)
                    ri += 1
                if wi < len(pt_model.write_after_blocks) and bi == pt_model.write_after_blocks[wi]:
                    ca_rope = compute_rope(positions=local_pos, periods=ca_periods, dtype=torch.float32)
                    canvas_pt = pt_model.write_attn[wi](query=canvas_pt, kv=x_pt, query_rope=c_rope, kv_rope=ca_rope)
                    wi += 1

        # Run MLX block loop from same packed input
        cfg = mlx_model.cfg
        x = mx.array(packed.detach().numpy())
        canvas = mx.array(state_pt.canvas.detach().numpy())
        from canvit_mlx.rope import compute_rope as mlx_compute_rope, make_rope_periods as mlx_make_rope_periods
        from canvit_mlx.coords import grid_coords as mlx_grid_coords
        import math

        mlx_local_pos = mx.array(local_pos.numpy())
        bb_sin, bb_cos = mlx_compute_rope(mlx_local_pos, mlx_make_rope_periods(cfg.head_dim))
        ca_sin, ca_cos = mlx_compute_rope(mlx_local_pos, mlx_make_rope_periods(cfg.canvas_head_dim))
        sp = mx.broadcast_to(mlx_grid_coords(CANVAS_GRID, CANVAS_GRID).reshape(1, -1, 2), (B, CANVAS_GRID**2, 2))
        c_sin, c_cos = mlx_compute_rope(sp, mlx_make_rope_periods(cfg.canvas_head_dim))

        ri, wi = 0, 0
        for bi in range(cfg.n_blocks):
            x = mlx_model.blocks[bi](x, bb_sin, bb_cos)
            mx.eval(x)
            if ri < len(mlx_model.read_after_blocks) and bi == mlx_model.read_after_blocks[ri]:
                x = mlx_model.read_attn[ri](x, canvas, ca_sin, ca_cos, c_sin, c_cos)
                mx.eval(x)
                ri += 1
            if wi < len(mlx_model.write_after_blocks) and bi == mlx_model.write_after_blocks[wi]:
                canvas = mlx_model.write_attn[wi](canvas, x, c_sin, c_cos, ca_sin, ca_cos)
                mx.eval(canvas)
                wi += 1

        assert_close("block_loop_canvas", canvas_pt.numpy(), canvas, atol=2.0, rtol=1e-3)
        idx = 1  # skip VPE
        out_pt = LocalTokens.unpack(x_pt, has_vpe=True, n_registers=pt_model.backbone.n_register_tokens, n_patches=H * W)
        assert_close("block_loop_cls", out_pt.recurrent_cls.numpy(), x[:, idx:idx + 1], atol=1.0, rtol=1e-3)
        idx += 2 + cfg.n_register_tokens
        assert_close("block_loop_patches", out_pt.patches.numpy(), x[:, idx:idx + H * W], atol=1.0, rtol=1e-3)
