"""Model building block tests: patch embed, VPE, init state, R/W schedule."""

import mlx.core as mx
import torch

from conftest import assert_close, B, CANVAS_GRID


class TestRWPositions:
    def test_default_config(self):
        from canvit_mlx.canvit import compute_rw_positions
        r, w = compute_rw_positions(12, 2)
        assert r == [1, 5, 9]
        assert w == [3, 7, 11]

    def test_forced_final_write(self):
        from canvit_mlx.canvit import compute_rw_positions
        r, w = compute_rw_positions(10, 2)
        assert w[-1] == 9  # forced: 7 != 9


class TestPatchEmbed:
    def test_output(self, mlx_model, glimpse_pair, pt_model):
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
