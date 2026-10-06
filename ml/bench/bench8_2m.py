"""bench8_2m: ~2M-param model training-speed test on the fused HIP pipeline.

Arch: 1200 -> 1024 -> 768 -> 1  (~2.02M params)
L1 = FusedGatherLinear (HIP gather+bias+GEMM+Xg writeout, rocBLAS dW)
L2/L3 = Fn.linear. fp16 + AdamW fused + TunableOp.
Reports samples/sec and sec/epoch over N=500k rows.

Calistirma (KB kokunden): GPU_PYTHON.cmd ml/bench/bench8_2m.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fxgpu.env import setup_env  # noqa: E402
setup_env()

import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from fxgpu import FusedGatherLinear  # noqa: E402

N, F, W = 500_000, 40, 30
H1, H2, BS = 1024, 768, 32768
K = W * F
EPOCHS = 5


def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    print("tunable:", torch.cuda.tunable.is_enabled())

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((N, F)).astype(np.float32)).to(device)
    y = torch.from_numpy((rng.random(N) > 0.5).astype(np.float32)).to(device)
    X16 = X.half()
    lo, ntr = W - 1, N - (W - 1)

    torch.manual_seed(0)
    W1 = nn.Parameter(torch.randn(H1, K, device=device, dtype=torch.float16) * 0.02)
    b1 = nn.Parameter(torch.zeros(H1, device=device, dtype=torch.float16))
    W2 = nn.Parameter(torch.randn(H2, H1, device=device, dtype=torch.float16) * 0.03)
    b2 = nn.Parameter(torch.zeros(H2, device=device, dtype=torch.float16))
    W3 = nn.Parameter(torch.randn(1, H2, device=device, dtype=torch.float16) * 0.05)
    b3 = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float16))
    params = [W1, b1, W2, b2, W3, b3]
    nparam = sum(p.numel() for p in params)
    print(f"params: {nparam:,}  ({nparam/1e6:.2f}M)")
    opt = torch.optim.AdamW(params, lr=3e-4, fused=True)
    lossf = nn.BCEWithLogitsLoss()

    steps_per_epoch = max(1, ntr // BS)

    def step(n):
        for _ in range(n):
            r = torch.randint(0, ntr, (BS,), device=device)
            h = Fn.silu(FusedGatherLinear.apply(X16, r.to(torch.int32), W1, b1))
            h = Fn.silu(Fn.linear(h, W2, b2))
            out = Fn.linear(h, W3, b3)
            loss = lossf(out.ravel().float(), y[r + lo])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

    step(10)  # warmup + tunableop tuning
    torch.cuda.synchronize()

    for ep in range(EPOCHS):
        t0 = time.perf_counter()
        step(steps_per_epoch)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sps = steps_per_epoch * BS / dt
        print(f"epoch {ep}: {dt:.2f}s  ({sps:,.0f} samples/sec, "
              f"{steps_per_epoch} steps x {BS})")

    # throughput-only measure, more iters
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step(60)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"steady-state: {60 * BS / dt:,.0f} samples/sec")


if __name__ == "__main__":
    main()
