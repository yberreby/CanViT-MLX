"""Tests for CanViTForImageClassification and fuse_probe."""

import mlx.core as mx
import numpy as np

from canvit_mlx import CanViTForImageClassification, fuse_probe
from conftest import assert_close, B, CANVAS_GRID


class TestFuseProbe:
    def test_fuse_matches_sequential(self):
        """Fused transform must produce same result as proj -> destd -> probe."""
        D, teacher_dim, n_classes = 768, 768, 1000
        mx.random.seed(42)

        W_proj = mx.random.normal((teacher_dim, D)) * 0.01
        b_proj = mx.random.normal((teacher_dim,)) * 0.01
        mu = mx.random.normal((teacher_dim,))
        sigma = mx.abs(mx.random.normal((teacher_dim,))) + 0.1
        W_probe = mx.random.normal((n_classes, teacher_dim)) * 0.01
        b_probe = mx.random.normal((n_classes,)) * 0.01

        W_fused, b_fused = fuse_probe(
            W_proj=W_proj, b_proj=b_proj,
            mu=mu, sigma=sigma,
            W_probe=W_probe, b_probe=b_probe,
        )
        mx.eval(W_fused, b_fused)

        # Sequential computation
        z = mx.random.normal((4, D))
        s = z @ W_proj.T + b_proj
        d = sigma * s + mu
        ref = d @ W_probe.T + b_probe

        fused = z @ W_fused.T + b_fused
        mx.eval(ref, fused)

        assert_close("fuse_probe", np.array(ref), fused, atol=1e-2, rtol=1e-4)


class TestClassificationModel:
    def test_from_pretrained_with_probe(self, mlx_model):
        """Verify from_pretrained_with_probe produces a working classifier."""
        from pathlib import Path
        from conftest import WEIGHTS

        weights_path = Path(WEIGHTS)
        config_path = weights_path.with_suffix(".json")

        # We need a probe. Create a synthetic one for testing.
        teacher_dim = mlx_model.cfg.teacher_dim
        assert teacher_dim is not None
        n_classes = 10
        mx.random.seed(123)
        probe_W = mx.random.normal((n_classes, teacher_dim)) * 0.01
        probe_b = mx.zeros((n_classes,))

        # Save synthetic probe to temp file
        import tempfile
        from safetensors.numpy import save_file
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            probe_path = Path(f.name)
            save_file({"weight": np.array(probe_W), "bias": np.array(probe_b)}, str(probe_path))

        try:
            clf = CanViTForImageClassification.from_pretrained_with_probe(
                pretrained_weights=weights_path,
                pretrained_config=config_path,
                probe_weights=probe_path,
            )
            assert clf.n_classes == n_classes

            # Run a forward pass
            from canvit_mlx import Viewpoint, extract_glimpse_at_viewpoint
            state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
            glimpse = mx.random.normal((B, 128, 128, 3))
            vp = Viewpoint.full_scene(batch_size=B)
            logits, new_state = clf(glimpse, state, vp)
            mx.eval(logits, new_state.canvas, new_state.recurrent_cls)

            assert logits.shape == (B, n_classes)
            assert new_state.canvas.shape[1] == clf.cfg.n_canvas_registers + CANVAS_GRID ** 2
        finally:
            probe_path.unlink()

    def test_head_forward_matches_call(self, mlx_model):
        """head_forward on extracted CLS must match full forward logits."""
        from pathlib import Path
        from conftest import WEIGHTS
        import tempfile
        from safetensors.numpy import save_file

        weights_path = Path(WEIGHTS)
        config_path = weights_path.with_suffix(".json")

        teacher_dim = mlx_model.cfg.teacher_dim
        assert teacher_dim is not None
        n_classes = 5
        mx.random.seed(456)

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            probe_path = Path(f.name)
            save_file({
                "weight": np.array(mx.random.normal((n_classes, teacher_dim)) * 0.01),
                "bias": np.array(mx.zeros((n_classes,))),
            }, str(probe_path))

        try:
            clf = CanViTForImageClassification.from_pretrained_with_probe(
                pretrained_weights=weights_path,
                pretrained_config=config_path,
                probe_weights=probe_path,
            )

            from canvit_mlx import Viewpoint
            state = clf.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
            glimpse = mx.random.normal((B, 128, 128, 3))
            vp = Viewpoint.full_scene(batch_size=B)

            # Full forward
            logits_full, new_state = clf(glimpse, state, vp)

            # Backbone + head_forward
            out = clf.canvit_forward(glimpse, state, vp)
            cls = out.state.recurrent_cls[:, 0]
            logits_split = clf.head_forward(cls)

            mx.eval(logits_full, logits_split)
            assert_close("head_vs_full", np.array(logits_full), logits_split, atol=1e-5)
        finally:
            probe_path.unlink()
