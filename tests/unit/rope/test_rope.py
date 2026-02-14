"""RoPE unit tests: periods, sin/cos computation, equivalence of two styles."""

import mlx.core as mx
import numpy as np

from canvit_mlx.grid import grid_coords
from canvit_mlx.rope import compute_rope, make_rope_periods
from conftest import assert_close


class TestPeriods:
    def test_values(self):
        for hd in [64, 128]:
            nf = hd // 4
            ref = 100.0 ** (np.arange(nf) / nf)
            got = make_rope_periods(hd)
            mx.eval(got)
            assert_close(f"periods_hd{hd}", ref, got, atol=1e-5)


class TestSinCos:
    def test_shape(self):
        pos = mx.zeros((2, 10, 2))
        sin, cos = compute_rope(pos, make_rope_periods(64))
        mx.eval(sin, cos)
        assert sin.shape == (2, 1, 10, 64)
        assert cos.shape == (2, 1, 10, 64)

    def test_vs_pytorch(self, pt_model, mlx_model):
        """Compare sin/cos against PyTorch reference for full-scene 8x8 grid."""
        import torch
        from canvit.coords import canvas_coords_for_glimpse
        from canvit.rope import compute_rope as pt_compute_rope, make_rope_periods as pt_make_rope_periods
        from canvit.viewpoint import Viewpoint

        vp = Viewpoint.full_scene(batch_size=1, device=torch.device("cpu"))
        local_pos = canvas_coords_for_glimpse(center=vp.centers, scale=vp.scales, H=8, W=8).flatten(1, 2)

        for suffix, hd in [("bb", 64), ("ca", 128)]:
            periods = pt_make_rope_periods(head_dim=hd, base=100.0, device=torch.device("cpu"))
            rope = pt_compute_rope(positions=local_pos, periods=periods)

            pos_mlx = mx.array(grid_coords(8, 8).reshape(1, 64, 2))
            sin, cos = compute_rope(pos_mlx, make_rope_periods(hd))
            mx.eval(sin, cos)

            assert_close(f"rope_{suffix}_sin", rope.sin.numpy(), sin, atol=1e-5)
            assert_close(f"rope_{suffix}_cos", rope.cos.numpy(), cos, atol=1e-5)
