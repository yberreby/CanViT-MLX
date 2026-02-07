"""Integration tests: true end-to-end MLX vs PyTorch comparison.

Tests run the full forward path (raw glimpse → all outputs) and multi-step recurrence.
"""

import mlx.core as mx
import torch

from canvit_mlx import RecurrentState, Viewpoint
from conftest import (
    assert_close, pt_forward, pt_recurrent_state, pt_viewpoint, pt_viewpoint_full,
    B, CANVAS_GRID, GLIMPSE_PX, SEED,
)


def _mlx_forward(model, glimpse_mlx, vp, state):
    out = model(glimpse_mlx, state, vp)
    mx.eval(out.state.canvas, out.state.recurrent_cls, out.ephemeral_cls, out.local_patches)
    return out


class TestEndToEnd:
    def test_full_scene(self, pt_model, mlx_model, glimpse_pair):
        glimpse_pt, glimpse_mlx = glimpse_pair

        vp_pt = pt_viewpoint_full()
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref = pt_forward(pt_model, glimpse_pt, vp_pt, state_pt)

        vp_mlx = Viewpoint.full_scene(batch_size=B)
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        out = _mlx_forward(mlx_model, glimpse_mlx, vp_mlx, state_mlx)

        assert_close("e2e_canvas", ref["canvas"], out.state.canvas, atol=2.0, rtol=1e-3)
        assert_close("e2e_cls", ref["recurrent_cls"], out.state.recurrent_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_ephemeral", ref["ephemeral_cls"], out.ephemeral_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_patches", ref["local_patches"], out.local_patches, atol=1.0, rtol=1e-3)

    def test_off_center(self, pt_model, mlx_model):
        torch.manual_seed(SEED + 10)
        g_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        g_mlx = mx.array(g_pt.numpy().transpose(0, 2, 3, 1))

        vp_pt = pt_viewpoint([[-0.4, 0.3]], [0.6])
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref = pt_forward(pt_model, g_pt, vp_pt, state_pt)

        vp_mlx = Viewpoint(centers=mx.array([[-0.4, 0.3]]), scales=mx.array([0.6]))
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        out = _mlx_forward(mlx_model, g_mlx, vp_mlx, state_mlx)

        # Canvas values reach ~2600; f32 SDPA over 1040 tokens → occasional abs err ~3-4
        assert_close("e2e_offcenter_canvas", ref["canvas"], out.state.canvas, atol=5.0, rtol=2e-3)
        assert_close("e2e_offcenter_cls", ref["recurrent_cls"], out.state.recurrent_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_offcenter_patches", ref["local_patches"], out.local_patches, atol=1.0, rtol=1e-3)


class TestMultiStep:
    def test_two_steps(self, pt_model, mlx_model):
        torch.manual_seed(SEED + 20)
        g1_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        g2_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        g1_mlx = mx.array(g1_pt.numpy().transpose(0, 2, 3, 1))
        g2_mlx = mx.array(g2_pt.numpy().transpose(0, 2, 3, 1))

        # Step 1
        vp1_pt = pt_viewpoint_full()
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref1 = pt_forward(pt_model, g1_pt, vp1_pt, state_pt)

        vp1_mlx = Viewpoint.full_scene(batch_size=B)
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        out1 = _mlx_forward(mlx_model, g1_mlx, vp1_mlx, state_mlx)
        assert_close("step1_canvas", ref1["canvas"], out1.state.canvas, atol=2.0, rtol=1e-3)

        # Step 2: feed output state back in
        vp2_pt = pt_viewpoint([[0.2, -0.3]], [0.7])
        state_pt2 = pt_recurrent_state(ref1["canvas"], ref1["recurrent_cls"])
        ref2 = pt_forward(pt_model, g2_pt, vp2_pt, state_pt2)

        vp2_mlx = Viewpoint(centers=mx.array([[0.2, -0.3]]), scales=mx.array([0.7]))
        state_mlx2 = RecurrentState(canvas=out1.state.canvas, recurrent_cls=out1.state.recurrent_cls)
        out2 = _mlx_forward(mlx_model, g2_mlx, vp2_mlx, state_mlx2)

        assert_close("step2_canvas", ref2["canvas"], out2.state.canvas, atol=3.0, rtol=2e-3)
        assert_close("step2_cls", ref2["recurrent_cls"], out2.state.recurrent_cls, atol=2.0, rtol=2e-3)
        assert_close("step2_patches", ref2["local_patches"], out2.local_patches, atol=2.0, rtol=2e-3)


class TestRunTrajectory:
    def test_matches_manual_loop(self, mlx_model, glimpse_pair):
        """run_trajectory should produce same results as manual __call__ loop."""
        _, glimpse_mlx = glimpse_pair
        # Build a tiny image and 2 viewpoints
        img = mx.broadcast_to(glimpse_mlx, (B, GLIMPSE_PX, GLIMPSE_PX, 3))
        vps = [Viewpoint.full_scene(batch_size=B),
               Viewpoint(centers=mx.array([[0.2, -0.3]]), scales=mx.array([0.7]))]

        outputs = mlx_model.run_trajectory(img, vps, GLIMPSE_PX, CANVAS_GRID)
        assert len(outputs) == 2
        # Verify shapes
        for out in outputs:
            assert out.state.canvas.shape[1] == mlx_model.cfg.n_canvas_registers + CANVAS_GRID ** 2
            assert out.ephemeral_cls.shape == (B, 1, mlx_model.cfg.embed_dim)


class TestTeacherHeads:
    def test_predict_teacher_scene(self, mlx_model):
        canvas = mx.zeros((B, mlx_model.cfg.n_canvas_registers + CANVAS_GRID ** 2, mlx_model.cfg.canvas_dim))
        out = mlx_model.predict_teacher_scene(canvas)
        mx.eval(out)
        assert out.shape == (B, CANVAS_GRID ** 2, mlx_model.cfg.teacher_dim)

    def test_predict_scene_teacher_cls(self, mlx_model):
        cls = mx.zeros((B, 1, mlx_model.cfg.embed_dim))
        out = mlx_model.predict_scene_teacher_cls(cls)
        mx.eval(out)
        assert out.shape == (B, mlx_model.cfg.teacher_dim)
