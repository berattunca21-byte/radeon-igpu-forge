"""bench7_fused_e2e: end-to-end training loop with custom HIP fused layer-1.

FusedGatherLinear: fwd = gather_gemm HIP kernel (gather+bias+GEMM+Xg writeout),
bwd = rocBLAS mm for dW (dx not needed — input layer), db = sum.
Compare vs baseline loop (index_select gather + nn.Linear path).

Run (from the repo root):
  GPU_PYTHON.cmd ml/bench/bench7_fused_e2e.py
  BNCH_FUSED=0 -> baseline (index_select + Linear) comparison
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

N, F, W, HID, BS = 500_000, 40, 30, 256, 32768
K = W * F
ITERS, WARM = 60, 15


def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    use_fused = os.environ.get("BNCH_FUSED", "1") == "1"
    print("fused L1:", use_fused, "| tunable:", torch.cuda.tunable.is_enabled())

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((N, F)).astype(np.float32)).to(device)
    y = torch.from_numpy((rng.random(N) > 0.5).astype(np.float32)).to(device)
    X16 = X.half()
    Xu16 = X16.unfold(0, W, 1)
    lo, ntr = W - 1, N - (W - 1)

    torch.manual_seed(0)
    W1 = nn.Parameter(torch.randn(HID, K, device=device, dtype=torch.float16) * 0.03)
    b1 = nn.Parameter(torch.zeros(HID, device=device, dtype=torch.float16))
    W2 = nn.Parameter(torch.randn(HID // 2, HID, device=device, dtype=torch.float16) * 0.06)
    b2 = nn.Parameter(torch.zeros(HID // 2, device=device, dtype=torch.float16))
    W3 = nn.Parameter(torch.randn(1, HID // 2, device=device, dtype=torch.float16) * 0.1)
    b3 = nn.Parameter(torch.zeros(1, device=device, dtype=torch.float16))
    params = [W1, b1, W2, b2, W3, b3]
    opt = torch.optim.AdamW(params, lr=1e-3, fused=True)
    lossf = nn.BCEWithLogitsLoss()

    def step(n):
        for _ in range(n):
            r = torch.randint(0, ntr, (BS,), device=device)
            if use_fused:
                r32 = r.to(torch.int32)
                h = Fn.silu(FusedGatherLinear.apply(X16, r32, W1, b1))
            else:
                xb = Xu16[r].transpose(1, 2).reshape(BS, K)
                h = Fn.silu(Fn.linear(xb, W1, b1))
            h = Fn.silu(Fn.linear(h, W2, b2))
            out = Fn.linear(h, W3, b3)
            loss = lossf(out.ravel().float(), y[r + lo])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()

    step(WARM)
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step(ITERS)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        print(f"t{trial}: {ITERS * BS / dt:,.0f} samples/sec ({dt:.2f}s)")

    # correctness spot-check: fresh params, one step, compare fused vs baseline grads
    torch.manual_seed(1)
    w1a = (torch.randn(HID, K, device=device, dtype=torch.float16) * 0.03).requires_grad_()
    b1a = (torch.randn(HID, device=device, dtype=torch.float16) * 0.01).requires_grad_()
    w1b = w1a.detach().clone().requires_grad_()
    b1b = b1a.detach().clone().requires_grad_()
    r = torch.randint(0, ntr, (4096,), device=device)
    g_seed = torch.randn(4096, HID, device=device, dtype=torch.float16)

    out_f = FusedGatherLinear.apply(X16, r.to(torch.int32), w1a, b1a)
    out_f.backward(g_seed)
    xb = Xu16[r].transpose(1, 2).reshape(4096, K)
    out_b = Fn.linear(xb, w1b, b1b)
    out_b.backward(g_seed)
    print(f"fwd out err={(out_f.double()-out_b.double()).abs().max().item():.4f} | "
          f"dW err={(w1a.grad.double()-w1b.grad.double()).abs().max().item():.4f} | "
          f"db err={(b1a.grad.double()-b1b.grad.double()).abs().max().item():.4f}")


if __name__ == "__main__":
    main()
