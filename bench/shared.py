"""Shared constants and utilities for bench scripts."""

import subprocess
from pathlib import Path

import numpy as np
import torch

B = 1
GLIMPSE_PX = 128
IMAGE_PATH = Path("test_data/Cat03.jpg")
WEIGHTS = "weights/canvit-vitb16-pretrain-512px-in21k.safetensors"
HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
DINOV3_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DINOV3S_REPO = "facebook/dinov3-vits16-pretrain-lvd1689m"
TEACHER_COMPONENTS = frozenset({"dinov3", "dinov3s"})

BENCH_DIR = Path(__file__).parent

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def make_pt_sync(device: torch.device):
    if device.type == "mps":
        return torch.mps.synchronize
    if device.type == "cuda":
        return torch.cuda.synchronize
    return lambda: None
