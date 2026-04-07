"""Test that convert.py produces loadable weights."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from convert import Args, convert


def test_convert_produces_loadable_weights(tmp_path):
    out = tmp_path / "test.safetensors"
    convert(Args(out=out, verify=False))

    assert out.exists()
    assert out.with_suffix(".json").exists()

    from canvit_mlx import load_from_local
    model = load_from_local(out, out.with_suffix(".json"))
    assert model.cfg.embed_dim == 768
    assert len(model.blocks) == 12
