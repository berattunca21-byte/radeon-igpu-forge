"""Radeon 680M / TheRock PyTorch smoke test.

Run (from the repo root): GPU_PYTHON.cmd GPU_SMOKE_TEST.py
(or with any ROCm venv python — venv-rocm10, venv-rocm10-stable, venv-therock).
It does not read or modify model checkpoints; it only does temporary FP32
computation in RAM/GPU memory. A small CNN + GroupNorm (BatchNorm could hang on Windows/gfx1035).
"""

from __future__ import annotations

import math
import sys
import time

import torch
from torch import nn


def fail(message: str, code: int = 1) -> None:
    print(f"\nFAILED: {message}")
    raise SystemExit(code)


def main() -> None:
    print("=== RADEON 680M / THEROCK GPU SMOKE TEST ===")
    print("python:", sys.version.replace("\n", " "))
    print("torch :", torch.__version__)
    print("HIP   :", torch.version.hip)
    print("GPU   :", torch.cuda.is_available())
    if not torch.cuda.is_available():
        fail("torch cannot see the Radeon GPU; did not fall back to a CPU test", 2)

    props = torch.cuda.get_device_properties(0)
    arch = getattr(props, "gcnArchName", "?")
    print("device:", props.name)
    print("arch  :", arch)
    print("memory:", round(props.total_memory / 2**30, 2), "GB reported/shared")
    if not str(arch).startswith("gfx1035"):
        fail(f"expected gfx1035 but saw {arch}")

    torch.manual_seed(20260722)
    device = torch.device("cuda")
    # GroupNorm is a deliberate choice: the Windows/gfx1035 BatchNorm path could
    # hang in this project in the past.
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
        nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
        nn.Conv2d(32, 8, 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    target = torch.randint(0, 8, (8, 64, 64), device=device)
    x = torch.rand(8, 1, 128, 128, device=device)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    started = time.perf_counter()
    for _step in range(20):
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = criterion(output, target)
        if not torch.isfinite(loss):
            fail("NaN/Inf loss")
        loss.backward()
        for parameter in model.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                fail("NaN/Inf gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not all(math.isfinite(value) for value in losses):
        fail("loss history is not finite")
    if losses[-1] >= losses[0]:
        fail(f"loss did not decrease: {losses[0]:.5f} -> {losses[-1]:.5f}")

    print(f"loss  : {losses[0]:.5f} -> {losses[-1]:.5f}")
    print(f"time  : {elapsed:.2f} s / 20 FP32 steps")
    print(f"alloc : {torch.cuda.memory_allocated() / 2**20:.1f} MB")
    print("\nSUCCESS: gfx1035 FP32 forward/backward/optimizer test passed")


if __name__ == "__main__":
    main()

