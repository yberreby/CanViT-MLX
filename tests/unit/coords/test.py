"""Coordinate and sampling unit tests."""

import mlx.core as mx
import numpy as np
import torch

from canvit_mlx.coords import grid_coords, sample_at_viewpoint, Viewpoint
from conftest import assert_close, B, SEED


class TestGridCoords:
    def test_shape_and_range(self):
        g = grid_coords(4, 6)
        mx.eval(g)
        assert g.shape == (4, 6, 2)
        g = np.array(g)
        assert g.min() > -1 and g.max() < 1  # strictly inside [-1, 1]

    def test_center(self):
        g = grid_coords(2, 2)
        mx.eval(g)
        assert np.allclose(np.array(g).mean(axis=(0, 1)), [0, 0], atol=1e-6)


class TestSampleAtViewpoint:
    def _compare(self, img_pt, vp_centers, vp_scales, glimpse_px):
        from conftest import pt_sample, pt_viewpoint

        vp_pt = pt_viewpoint(vp_centers, vp_scales)
        ref = pt_sample(img_pt, vp_pt, glimpse_px)

        img_mlx = mx.array(img_pt.numpy().transpose(0, 2, 3, 1))
        vp_mlx = Viewpoint(centers=mx.array(vp_centers), scales=mx.array(vp_scales))
        got = sample_at_viewpoint(img_mlx, vp_mlx, glimpse_px)
        mx.eval(got)
        return ref, got

    def test_full_scene(self):
        torch.manual_seed(SEED + 1)
        img_pt = torch.randn(B, 3, 64, 64)
        ref, got = self._compare(img_pt, [[0.0, 0.0]], [1.0], 32)
        assert_close("sample_full", ref, got, atol=1e-4)

    def test_off_center(self):
        torch.manual_seed(SEED + 2)
        img_pt = torch.randn(B, 3, 64, 64)
        ref, got = self._compare(img_pt, [[0.3, -0.2]], [0.5], 32)
        assert_close("sample_offcenter", ref, got, atol=1e-4)
