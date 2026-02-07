"""Module-by-module verification of MLX CanViT against f32 CPU PyTorch reference.

Each test generates the PyTorch reference in-process (via canvit from HF Hub),
then runs the MLX equivalent and asserts numerical agreement.

Run: uv run pytest test_modules.py -v

Requires: `uv run python convert.py` first to generate weights.safetensors.
"""

import logging

import mlx.core as mx
import numpy as np
import pytest
import torch

from canvit_mlx import (
    CanViTOutput,
    Viewpoint as MlxViewpoint,
    compute_rope as mlx_compute_rope,
    grid_coords as mlx_grid_coords,
    load_canvit,
    make_rope_periods as mlx_make_rope_periods,
    sample_at_viewpoint as mlx_sample_at_viewpoint,
)

log = logging.getLogger(__name__)

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
WEIGHTS = "weights.safetensors"
SEED = 42
B, CANVAS_GRID, GLIMPSE_PX = 1, 32, 128


def assert_close(
    name: str, ref: np.ndarray, got: mx.array, *, atol: float, rtol: float = 1e-4
) -> None:
    got_np = np.array(got)
    assert ref.shape == got_np.shape, f"{name}: shape {ref.shape} vs {got_np.shape}"
    diff = np.abs(ref - got_np)
    scale = np.abs(ref).max() + 1e-8
    max_abs = float(diff.max())
    max_rel = max_abs / scale
    if not (max_abs < atol or max_rel < rtol):
        idx = np.unravel_index(diff.argmax(), diff.shape)
        raise AssertionError(
            f"{name}: max_abs={max_abs:.2e} (atol={atol:.0e}), "
            f"max_rel={max_rel:.2e} (rtol={rtol:.0e}), "
            f"worst at {idx}: ref={ref[idx]:.6f}, got={got_np[idx]:.6f}"
        )


# ---- Fixtures ----

@pytest.fixture(scope="module")
def pt_model():
    from canvit import CanViTForPretrainingHFHub
    log.info("Loading PyTorch model from HF: %s", HF_REPO)
    return CanViTForPretrainingHFHub.from_pretrained(HF_REPO).eval()


@pytest.fixture(scope="module")
def mlx_model():
    log.info("Loading MLX model from %s", WEIGHTS)
    return load_canvit(WEIGHTS)


@pytest.fixture(scope="module")
def glimpse_pt():
    torch.manual_seed(SEED)
    return torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)


@pytest.fixture(scope="module")
def glimpse_mlx(glimpse_pt):
    return mx.array(glimpse_pt.numpy().transpose(0, 2, 3, 1))


@pytest.fixture(scope="module")
def pt_intermediates(pt_model, glimpse_pt):
    """Run full PyTorch forward collecting all intermediates."""
    from canvit.coords import canvas_coords_for_glimpse, grid_coords
    from canvit.model.base.impl import LocalTokens
    from canvit.rope import RoPE, compute_rope, make_rope_periods
    from canvit.viewpoint import Viewpoint

    model = pt_model
    vp = Viewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
    state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)

    backbone_tokens, H, W = model.backbone.prepare_tokens(glimpse_pt)
    patches = backbone_tokens[:, 1 + model.backbone.n_register_tokens:]
    cls_token = backbone_tokens[:, 0:1]
    registers = backbone_tokens[:, 1:1 + model.backbone.n_register_tokens]

    vpe_enc = model.vpe_encoder(y=vp.centers[:, 0], x=vp.centers[:, 1], s=vp.scales)

    tokens = LocalTokens(vpe=vpe_enc.unsqueeze(1), recurrent_cls=state.recurrent_cls,
                         ephemeral_cls=cls_token, registers=registers, patches=patches)
    local = tokens.pack()

    local_pos = canvas_coords_for_glimpse(center=vp.centers, scale=vp.scales, H=H, W=W).flatten(1, 2)
    bb_periods = make_rope_periods(head_dim=model.backbone.head_dim, device=torch.device("cpu"))
    ca_periods = make_rope_periods(head_dim=model.cfg.canvas_head_dim, device=torch.device("cpu"))
    rope_bb = RoPE(*compute_rope(positions=local_pos, periods=bb_periods, dtype=torch.float32))
    rope_ca = RoPE(*compute_rope(positions=local_pos, periods=ca_periods, dtype=torch.float32))
    spatial_pos = grid_coords(H=CANVAS_GRID, W=CANVAS_GRID, device=torch.device("cpu")).flatten(0, 1).unsqueeze(0).expand(B, -1, -1)
    c_rope = RoPE(*compute_rope(positions=spatial_pos, periods=ca_periods, dtype=torch.float32))

    d: dict[str, np.ndarray] = {
        "patches": patches.detach().numpy(),
        "vpe": vpe_enc.detach().numpy(),
        "init_canvas": state.canvas.detach().numpy(),
        "init_cls": state.recurrent_cls.detach().numpy(),
        "packed": local.detach().numpy(),
        "local_rope_bb_sin": rope_bb.sin.numpy(), "local_rope_bb_cos": rope_bb.cos.numpy(),
        "local_rope_ca_sin": rope_ca.sin.numpy(), "local_rope_ca_cos": rope_ca.cos.numpy(),
        "canvas_rope_sin": c_rope.sin.numpy(), "canvas_rope_cos": c_rope.cos.numpy(),
    }

    x, canvas = local, state.canvas
    ri, wi = 0, 0
    with torch.inference_mode():
        for bi in range(model.backbone.n_blocks):
            x = model.backbone.forward_block(bi, x, rope_bb)
            d[f"after_block{bi}"] = x.numpy()
            if ri < len(model.read_after_blocks) and bi == model.read_after_blocks[ri]:
                x = model.read_attn[ri](query=x, kv=canvas, query_rope=rope_ca, kv_rope=c_rope)
                d[f"after_read{ri}"] = x.numpy()
                ri += 1
            if wi < len(model.write_after_blocks) and bi == model.write_after_blocks[wi]:
                canvas = model.write_attn[wi](query=canvas, kv=x, query_rope=c_rope, kv_rope=rope_ca)
                d[f"after_write{wi}"] = canvas.numpy()
                wi += 1

    out = LocalTokens.unpack(x, has_vpe=True, n_registers=model.backbone.n_register_tokens, n_patches=H * W)
    d["final_cls"] = out.recurrent_cls.numpy()
    d["final_patches"] = out.patches.numpy()
    d["final_canvas"] = canvas.numpy()

    log.info("PyTorch intermediates computed: %d entries", len(d))
    return d


# ---- Tests ----

class TestRoPE:
    def test_periods(self, mlx_model):
        cfg = mlx_model.cfg
        for hd in [cfg.head_dim, cfg.canvas_head_dim]:
            nf = hd // 4
            ref = 100.0 ** (np.arange(nf) / nf)
            got = mlx_make_rope_periods(hd)
            mx.eval(got)
            assert_close(f"periods_hd{hd}", ref, got, atol=1e-5)

    def test_sincos_vs_reference(self, pt_intermediates):
        """Compute RoPE from the same positions as PyTorch and compare sin/cos."""
        for suffix, hd in [("bb", 64), ("ca", 128)]:
            ref_sin = pt_intermediates[f"local_rope_{suffix}_sin"]
            ref_cos = pt_intermediates[f"local_rope_{suffix}_cos"]
            # Reconstruct positions: full-scene 8x8 grid
            pos = mx.array(mlx_grid_coords(8, 8).reshape(1, 64, 2))
            sin, cos = mlx_compute_rope(pos, mlx_make_rope_periods(hd))
            mx.eval(sin, cos)
            assert_close(f"rope_{suffix}_sin", ref_sin, sin, atol=1e-5)
            assert_close(f"rope_{suffix}_cos", ref_cos, cos, atol=1e-5)


class TestPatchEmbed:
    def test_output(self, mlx_model, glimpse_mlx, pt_intermediates):
        tokens, H, W = mlx_model.patch_embed(glimpse_mlx)
        mx.eval(tokens)
        assert H == 8 and W == 8
        assert_close("patches", pt_intermediates["patches"], tokens, atol=1e-4)


class TestVPE:
    def test_full_scene(self, mlx_model, pt_intermediates):
        assert mlx_model.vpe_encoder is not None
        out = mlx_model.vpe_encoder(mx.zeros((1,)), mx.zeros((1,)), mx.ones((1,)))
        mx.eval(out)
        assert_close("vpe", pt_intermediates["vpe"], out, atol=1e-4)


class TestInitState:
    def test_state(self, mlx_model, pt_intermediates):
        state = mlx_model.init_state(batch_size=1, canvas_grid_size=32)
        mx.eval(state.canvas, state.recurrent_cls)
        assert_close("init_canvas", pt_intermediates["init_canvas"], state.canvas, atol=1e-6)
        assert_close("init_cls", pt_intermediates["init_cls"], state.recurrent_cls, atol=1e-6)


class TestBlock0:
    def test_from_ref_input(self, mlx_model, pt_intermediates):
        x = mx.array(pt_intermediates["packed"])
        sin = mx.array(pt_intermediates["local_rope_bb_sin"])
        cos = mx.array(pt_intermediates["local_rope_bb_cos"])
        out = mlx_model.blocks[0](x, sin, cos)
        mx.eval(out)
        assert_close("block0", pt_intermediates["after_block0"], out, atol=2e-3, rtol=1e-5)


class TestBlockLoop:
    """Verify the block loop with pre-computed inputs (isolates block logic from packing/RoPE)."""

    def test_with_ref_inputs(self, mlx_model, pt_intermediates):
        cfg = mlx_model.cfg
        x = mx.array(pt_intermediates["packed"])
        canvas = mx.array(pt_intermediates["init_canvas"])
        bb_sin = mx.array(pt_intermediates["local_rope_bb_sin"])
        bb_cos = mx.array(pt_intermediates["local_rope_bb_cos"])
        ca_sin = mx.array(pt_intermediates["local_rope_ca_sin"])
        ca_cos = mx.array(pt_intermediates["local_rope_ca_cos"])
        c_sin = mx.array(pt_intermediates["canvas_rope_sin"])
        c_cos = mx.array(pt_intermediates["canvas_rope_cos"])

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

        assert_close("final_canvas", pt_intermediates["final_canvas"], canvas, atol=2.0, rtol=1e-3)
        idx = 1  # skip VPE
        assert_close("final_cls", pt_intermediates["final_cls"], x[:, idx:idx+1], atol=1.0, rtol=1e-3)
        idx += 2 + cfg.n_register_tokens
        assert_close("final_patches", pt_intermediates["final_patches"], x[:, idx:idx+64], atol=1.0, rtol=1e-3)


class TestSampleAtViewpoint:
    """Verify hand-rolled bilinear sampling against PyTorch F.grid_sample."""

    def test_full_scene(self, pt_model):
        from canvit.viewpoint import Viewpoint as PtViewpoint, sample_at_viewpoint as pt_sample

        torch.manual_seed(SEED + 1)
        img_pt = torch.randn(B, 3, 64, 64)
        vp_pt = PtViewpoint.full_scene(batch_size=B, device=torch.device("cpu"))

        ref = pt_sample(spatial=img_pt, viewpoint=vp_pt, glimpse_size_px=32)
        ref = ref.numpy().transpose(0, 2, 3, 1)  # NCHW -> NHWC

        img_mlx = mx.array(img_pt.numpy().transpose(0, 2, 3, 1))
        vp_mlx = MlxViewpoint.full_scene(batch_size=B)
        got = mlx_sample_at_viewpoint(img_mlx, vp_mlx, 32)
        mx.eval(got)

        assert_close("sample_full_scene", ref, got, atol=1e-4)

    def test_off_center(self, pt_model):
        from canvit.viewpoint import Viewpoint as PtViewpoint, sample_at_viewpoint as pt_sample

        torch.manual_seed(SEED + 2)
        img_pt = torch.randn(B, 3, 64, 64)
        vp_pt = PtViewpoint(
            centers=torch.tensor([[0.3, -0.2]]),
            scales=torch.tensor([0.5]),
        )

        ref = pt_sample(spatial=img_pt, viewpoint=vp_pt, glimpse_size_px=32)
        ref = ref.numpy().transpose(0, 2, 3, 1)

        img_mlx = mx.array(img_pt.numpy().transpose(0, 2, 3, 1))
        vp_mlx = MlxViewpoint(
            centers=mx.array([[0.3, -0.2]]),
            scales=mx.array([0.5]),
        )
        got = mlx_sample_at_viewpoint(img_mlx, vp_mlx, 32)
        mx.eval(got)

        assert_close("sample_off_center", ref, got, atol=1e-4)


def _run_pt_forward(pt_model, glimpse_pt, viewpoint_pt, state_pt):
    """Run PyTorch forward and return outputs as numpy."""
    with torch.inference_mode():
        out = pt_model(glimpse=glimpse_pt, state=state_pt, viewpoint=viewpoint_pt)
    return {
        "canvas": out.state.canvas.numpy(),
        "recurrent_cls": out.state.recurrent_cls.numpy(),
        "ephemeral_cls": out.ephemeral_cls.numpy(),
        "local_patches": out.local_patches.numpy(),
    }


def _run_mlx_forward(mlx_model, glimpse_mlx, viewpoint_mlx, state_mlx):
    """Run MLX forward and return output."""
    out = mlx_model(glimpse_mlx, state_mlx, viewpoint_mlx)
    mx.eval(out.state.canvas, out.state.recurrent_cls, out.ephemeral_cls, out.local_patches)
    return out


class TestTrueEndToEnd:
    """True end-to-end: raw glimpse in, compare all outputs against PyTorch."""

    def test_full_scene(self, pt_model, mlx_model, glimpse_pt, glimpse_mlx):
        from canvit.viewpoint import Viewpoint as PtViewpoint

        vp_pt = PtViewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref = _run_pt_forward(pt_model, glimpse_pt, vp_pt, state_pt)

        vp_mlx = MlxViewpoint.full_scene(batch_size=B)
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        out = _run_mlx_forward(mlx_model, glimpse_mlx, vp_mlx, state_mlx)

        assert_close("e2e_canvas", ref["canvas"], out.state.canvas, atol=2.0, rtol=1e-3)
        assert_close("e2e_cls", ref["recurrent_cls"], out.state.recurrent_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_ephemeral", ref["ephemeral_cls"], out.ephemeral_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_patches", ref["local_patches"], out.local_patches, atol=1.0, rtol=1e-3)

    def test_off_center_viewpoint(self, pt_model, mlx_model):
        from canvit.viewpoint import Viewpoint as PtViewpoint

        torch.manual_seed(SEED + 10)
        glimpse_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        glimpse_mlx = mx.array(glimpse_pt.numpy().transpose(0, 2, 3, 1))

        vp_pt = PtViewpoint(
            centers=torch.tensor([[-0.4, 0.3]]),
            scales=torch.tensor([0.6]),
        )
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref = _run_pt_forward(pt_model, glimpse_pt, vp_pt, state_pt)

        vp_mlx = MlxViewpoint(
            centers=mx.array([[-0.4, 0.3]]),
            scales=mx.array([0.6]),
        )
        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        out = _run_mlx_forward(mlx_model, glimpse_mlx, vp_mlx, state_mlx)

        # Canvas values reach ~2600 magnitude; f32 SDPA accumulation over 1040 tokens
        # gives occasional absolute errors ~3-4 at extreme values (relative ~0.15%)
        assert_close("e2e_offcenter_canvas", ref["canvas"], out.state.canvas, atol=5.0, rtol=2e-3)
        assert_close("e2e_offcenter_cls", ref["recurrent_cls"], out.state.recurrent_cls, atol=1.0, rtol=1e-3)
        assert_close("e2e_offcenter_patches", ref["local_patches"], out.local_patches, atol=1.0, rtol=1e-3)


class TestMultiStep:
    """Two consecutive forward passes — verifies state recurrence."""

    def test_two_steps(self, pt_model, mlx_model):
        from canvit.viewpoint import Viewpoint as PtViewpoint

        torch.manual_seed(SEED + 20)
        g1_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        g2_pt = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX)
        g1_mlx = mx.array(g1_pt.numpy().transpose(0, 2, 3, 1))
        g2_mlx = mx.array(g2_pt.numpy().transpose(0, 2, 3, 1))

        vp1_pt = PtViewpoint.full_scene(batch_size=B, device=torch.device("cpu"))
        vp2_pt = PtViewpoint(centers=torch.tensor([[0.2, -0.3]]), scales=torch.tensor([0.7]))

        # Step 1
        state_pt = pt_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        ref1 = _run_pt_forward(pt_model, g1_pt, vp1_pt, state_pt)

        state_mlx = mlx_model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        vp1_mlx = MlxViewpoint.full_scene(batch_size=B)
        out1 = _run_mlx_forward(mlx_model, g1_mlx, vp1_mlx, state_mlx)

        assert_close("step1_canvas", ref1["canvas"], out1.state.canvas, atol=2.0, rtol=1e-3)

        # Step 2: feed output state back in
        from canvit.model.base.impl import RecurrentState as PtRecurrentState
        state_pt2 = PtRecurrentState(
            canvas=torch.tensor(ref1["canvas"]),
            recurrent_cls=torch.tensor(ref1["recurrent_cls"]),
        )
        ref2 = _run_pt_forward(pt_model, g2_pt, vp2_pt, state_pt2)

        from canvit_mlx import RecurrentState as MlxRecurrentState
        state_mlx2 = MlxRecurrentState(
            canvas=out1.state.canvas,
            recurrent_cls=out1.state.recurrent_cls,
        )
        vp2_mlx = MlxViewpoint(centers=mx.array([[0.2, -0.3]]), scales=mx.array([0.7]))
        out2 = _run_mlx_forward(mlx_model, g2_mlx, vp2_mlx, state_mlx2)

        assert_close("step2_canvas", ref2["canvas"], out2.state.canvas, atol=3.0, rtol=2e-3)
        assert_close("step2_cls", ref2["recurrent_cls"], out2.state.recurrent_cls, atol=2.0, rtol=2e-3)
        assert_close("step2_patches", ref2["local_patches"], out2.local_patches, atol=2.0, rtol=2e-3)
