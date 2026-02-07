"""Inference latency benchmark: PyTorch (CPU / MPS) vs MLX (CPU / GPU).

Usage: uv run python throwaway/bench.py
"""

import time
import statistics
import torch
import mlx.core as mx

# ── Config ──────────────────────────────────────────────────────────────────
B = 1
GLIMPSE_PX = 128
CANVAS_GRID = 32
WARMUP = 3
ITERS = 10
SEED = 42

# ── Helpers ─────────────────────────────────────────────────────────────────

def fmt(times: list[float]) -> str:
    med = statistics.median(times) * 1000
    mn = min(times) * 1000
    mx_ = max(times) * 1000
    return f"median {med:7.1f} ms  (min {mn:.1f}, max {mx_:.1f})"


# ── PyTorch benchmark ──────────────────────────────────────────────────────

def bench_pt(device_name: str) -> list[float]:
    from canvit import CanViTForPretrainingHFHub
    from canvit.viewpoint import Viewpoint

    device = torch.device(device_name)
    print(f"\n[PT {device_name}] loading model…")
    model = CanViTForPretrainingHFHub.from_pretrained(HF_REPO).to(device).eval()

    torch.manual_seed(SEED)
    glimpse = torch.randn(B, 3, GLIMPSE_PX, GLIMPSE_PX, device=device)
    vp = Viewpoint.full_scene(batch_size=B, device=device)

    times: list[float] = []
    for i in range(WARMUP + ITERS):
        state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        if device_name == "mps":
            torch.mps.synchronize()

        t0 = time.perf_counter()
        with torch.inference_mode():
            _ = model(glimpse=glimpse, state=state, viewpoint=vp)
        if device_name == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t0

        if i >= WARMUP:
            times.append(dt)
            print(f"  iter {i - WARMUP}: {dt*1000:.1f} ms")

    print(f"  → {fmt(times)}")
    return times


# ── MLX benchmark ───────────────────────────────────────────────────────────

HF_REPO = "canvit/canvit-vitb16-pretrain-512px-in21k"
_MODEL_NAME = HF_REPO.split("/")[-1]  # org/model → model
WEIGHTS = f"weights/{_MODEL_NAME}.safetensors"


def bench_mlx(gpu: bool) -> list[float]:
    from canvit_mlx import load_canvit, Viewpoint

    label = "GPU" if gpu else "CPU"
    print(f"\n[MLX {label}] loading model…")
    model = load_canvit(WEIGHTS)

    mx.random.seed(SEED)
    glimpse = mx.random.normal((B, GLIMPSE_PX, GLIMPSE_PX, 3))
    vp = Viewpoint.full_scene(batch_size=B)
    mx.eval(glimpse)

    times: list[float] = []
    for i in range(WARMUP + ITERS):
        state = model.init_state(batch_size=B, canvas_grid_size=CANVAS_GRID)
        mx.eval(state.canvas, state.recurrent_cls)

        if not gpu:
            device = mx.cpu
        else:
            device = mx.gpu

        t0 = time.perf_counter()
        with mx.stream(device):
            out = model(glimpse, state, vp)
            mx.eval(out.state.canvas, out.state.recurrent_cls,
                    out.ephemeral_cls, out.local_patches)
        dt = time.perf_counter() - t0

        if i >= WARMUP:
            times.append(dt)
            print(f"  iter {i - WARMUP}: {dt*1000:.1f} ms")

    print(f"  → {fmt(times)}")
    return times


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"BS={B}, glimpse={GLIMPSE_PX}px, canvas_grid={CANVAS_GRID}, "
          f"warmup={WARMUP}, iters={ITERS}")

    results: dict[str, list[float]] = {}

    results["PT CPU"] = bench_pt("cpu")

    if torch.backends.mps.is_available():
        results["PT MPS"] = bench_pt("mps")
    else:
        print("\n[PT MPS] not available, skipping")

    results["MLX CPU"] = bench_mlx(gpu=False)
    results["MLX GPU"] = bench_mlx(gpu=True)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for name, times in results.items():
        print(f"  {name:10s}  {fmt(times)}")
