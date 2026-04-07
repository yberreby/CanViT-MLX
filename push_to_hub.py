"""Push pre-converted MLX weights + model card to HuggingFace Hub.

Usage:
    uv run python push_to_hub.py
    uv run python push_to_hub.py --repo canvit/custom-name-mlx
"""

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import HfApi

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
DEFAULT_WEIGHTS = WEIGHTS_DIR / "canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02.safetensors"
DEFAULT_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02-mlx"

PT_PRETRAINED = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-2026-02-02"

PRETRAINED_CARD = """\
---
library_name: mlx
pipeline_tag: image-feature-extraction
tags:
  - vision
  - mlx
  - apple-silicon
  - canvit
  - active-vision
license: mit
---

# {repo_name} (MLX)

MLX-native checkpoint for [CanViT](https://arxiv.org/abs/2603.22570), the Canvas Vision Transformer, converted from the [PyTorch checkpoint](https://huggingface.co/{pt_repo}).

Pretrained on ImageNet-21k via dense latent distillation from DINOv3 ViT-B.

## Usage

```bash
uv add "canvit-mlx[hub]"
```

```python
import mlx.core as mx
from canvit_mlx import load_from_hf_hub, load_and_preprocess, Viewpoint, extract_glimpse_at_viewpoint

model = load_from_hf_hub("{mlx_repo}")
image = load_and_preprocess("path/to/image.jpg", target_size=512)

state = model.init_state(batch_size=1, canvas_grid_size=32)
vp = Viewpoint.full_scene(batch_size=1)
glimpse = extract_glimpse_at_viewpoint(image, vp, glimpse_size_px=128)
out = model(glimpse, state, vp)
mx.eval(out.state.canvas, out.state.recurrent_cls, out.local_patches)

canvas_spatial = model.get_spatial(out.state.canvas)  # [1, G*G, canvas_dim]
```

Source: [CanViT-MLX](https://github.com/yberreby/CanViT-MLX)

## Citation

```bibtex
@article{{berreby2026canvit,
  title={{CanViT: Toward Active-Vision Foundation Models}},
  author={{Berreby, Yoha{{\\"i}}-Eliel and Du, Sabrina and Durand, Audrey and Krishna, B. Suresh}},
  year={{2026}},
  eprint={{2603.22570}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CV}}
}}
```
"""


@dataclass
class Args:
    weights: Path = DEFAULT_WEIGHTS
    repo: str = DEFAULT_REPO


def main(args: Args) -> None:
    config_path = args.weights.with_suffix(".json")
    assert args.weights.exists(), f"Weights not found: {args.weights}"
    assert config_path.exists(), f"Config not found: {config_path}"

    config = json.loads(config_path.read_text())
    config["source"] = PT_PRETRAINED

    repo_name = args.repo.split("/")[-1]
    card = PRETRAINED_CARD.format(
        repo_name=repo_name,
        pt_repo=PT_PRETRAINED,
        mlx_repo=args.repo,
    )

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        # Copy weights as-is
        import shutil
        shutil.copy2(args.weights, root / "model.safetensors")
        (root / "config.json").write_text(json.dumps(config, indent=2))
        (root / "README.md").write_text(card)

        log.info("Uploading to %s ...", args.repo)
        api.upload_folder(folder_path=tmpdir, repo_id=args.repo)

    log.info("Done: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(tyro.cli(Args))
