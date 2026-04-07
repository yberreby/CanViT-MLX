"""Model building block tests: patch embed, VPE, init state, R/W schedule."""

import mlx.core as mx
import pytest
import torch

from canvit_mlx.canvit import _compute_rw_positions
from conftest import assert_close, B, CANVAS_GRID


@pytest.mark.parametrize("n_blocks,rw_stride,expect_reads,expect_writes", [
    (12, 2, [1, 5, 9], [3, 7, 11]),
    (10, 2, [1, 5, 9], [3, 7, 9]),
])
def test_rw_positions(n_blocks, rw_stride, expect_reads, expect_writes):
    reads, writes = _compute_rw_positions(n_blocks, rw_stride, enable_reads=True)
    assert reads == expect_reads
    assert writes == expect_writes


def test_rw_positions_reads_disabled():
    reads, writes = _compute_rw_positions(12, 2, enable_reads=False)
    assert reads == []
    assert writes == [3, 7, 11]


class TestPatchEmbed:
    def test_output(self, mlx_model, glimpse_pair, pt_model):
        glimpse_pt, glimpse_mlx = glimpse_pair
        ref_tokens, H_pt, W_pt = pt_model.backbone.patch_embed(glimpse_pt)
        ref = ref_tokens.detach().numpy()

        tokens, H, W = mlx_model.patch_embed(glimpse_mlx)
        mx.eval(tokens)
        assert H == H_pt and W == W_pt
        assert_close("patches", ref, tokens, atol=1e-4)


class TestVPE:
    def test_full_scene(self, mlx_model, pt_model):
        from canvit_pytorch.viewpoint import Viewpoint
        vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
        ref = pt_model.vpe(y=vp.centers[:, 0], x=vp.centers[:, 1], s=vp.scales).detach().numpy()

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
