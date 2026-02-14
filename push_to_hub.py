"""Push pre-converted MLX weights to HuggingFace Hub.

Usage:
    uv run python push_to_hub.py
    uv run python push_to_hub.py --repo canvit/custom-name-mlx
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import HfApi

log = logging.getLogger(__name__)

WEIGHTS_DIR = Path("weights")
DEFAULT_WEIGHTS = WEIGHTS_DIR / "canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16.safetensors"
DEFAULT_REPO = "canvit/canvitb16-add-vpe-pretrain-g128px-s512px-in21k-dv3b16-mlx"


@dataclass
class Args:
    weights: Path = DEFAULT_WEIGHTS
    repo: str = DEFAULT_REPO


def main(args: Args) -> None:
    config_path = args.weights.with_suffix(".json")
    assert args.weights.exists(), f"Weights not found: {args.weights}"
    assert config_path.exists(), f"Config not found: {config_path}"

    api = HfApi()
    api.create_repo(args.repo, exist_ok=True)

    log.info("Uploading %s → %s", args.weights.name, args.repo)
    api.upload_file(path_or_fileobj=str(args.weights), path_in_repo="model.safetensors", repo_id=args.repo)

    log.info("Uploading %s → %s", config_path.name, args.repo)
    api.upload_file(path_or_fileobj=str(config_path), path_in_repo="config.json", repo_id=args.repo)

    log.info("Done: https://huggingface.co/%s", args.repo)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main(tyro.cli(Args))
