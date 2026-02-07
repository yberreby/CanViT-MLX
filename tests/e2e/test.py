"""Crown jewel: real image, 3-step trajectory, PT vs MLX comparison + plot."""

import mlx.core as mx

from conftest import assert_close
from . import OUTPUTS, RTOL, load_image
from . import pt, mlx, plot


def compare(pt_results, mlx_results):
    for step, (ref, got) in enumerate(zip(pt_results, mlx_results)):
        for name in OUTPUTS:
            assert_close(f"step{step}_{name}", ref[name], mx.array(got[name]),
                         atol=float("inf"), rtol=RTOL)


class TestE2E:
    def test_real_image_trajectory(self, pt_model, mlx_model):
        image_mlx, image_pt = load_image()
        pt_results = pt.run(pt_model, image_pt)
        mlx_results = mlx.run(mlx_model, image_mlx)
        compare(pt_results, mlx_results)
        plot.save(pt_results, mlx_results, image_mlx, mlx_model.cfg.n_canvas_registers)
