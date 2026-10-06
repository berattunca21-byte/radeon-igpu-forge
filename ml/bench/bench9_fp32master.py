"""bench9_fp32master: pure fp16 vs fp32-master-weights (fp16 compute) speed + stability.

Same model/data/batch as bench7 (500k x 40, W=30, HID=256, bs=32768), L1 =
fxgpu.FusedGatherLinear (HIP gather_gemm). Modes (BNCH_MODE env, comma separated):
  fp16      : same as bench7 — weights are fp16, AdamW runs on fp16
  master    : weights are fp32 (AdamW fp32), fp16 compute via .half() each step
  master_ls : master + static loss scale (BNCH_LOSS_SCALE, default 256)

Two measurements:
  SPEED     : WARM warm-up + 3 x ITERS steps, samples/s (TunableOp tunes during warm-up)
  STABILITY : STAB_STEPS steps with a learnable label; is the loss trace finite, is it decreasing

Run (from the repo root):
  GPU_PYTHON.cmd ml/bench/bench9_fp32master.py
  (or env: PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
             PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=200)
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fxgpu.env import setup_env
setup_env()

import torch  # noqa: E402
from torch import nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from fxgpu import FusedGatherLinear, require_gpu  # noqa: E402

N, F, W, HID, BS = 500_000, 40, 30, 256, 32768
K = W * F
ITERS, WARM = 60, 15
STAB_STEPS, STAB_LOG = 300, 50
LR = 1e-3


def build_params(device, dtype, seed=0):
    torch.manual_seed(seed)
    W1 = nn.Parameter(torch.randn(HID, K, device=device, dtype=dtype) * 0.03)
    b1 = nn.Parameter(torch.zeros(HID, device=device, dtype=dtype))
    W2 = nn.Parameter(torch.randn(HID // 2, HID, device=device, dtype=dtype) * 0.06)
    b2 = nn.Parameter(torch.zeros(HID // 2, device=device, dtype=dtype))
    W3 = nn.Parameter(torch.randn(1, HID // 2, device=device, dtype=dtype) * 0.1)
    b3 = nn.Parameter(torch.zeros(1, device=device, dtype=dtype))
    return [W1, b1, W2, b2, W3, b3]


def forward(params, X16, r32, master):
    W1, b1, W2, b2, W3, b3 = [p.half() for p in params] if master else params
    h = Fn.silu(FusedGatherLinear.apply(X16, r32, W1, b1))
    h = Fn.silu(Fn.linear(h, W2, b2))
    return Fn.linear(h, W3, b3)


def make_step(params, X16, y, ntr, lo, master, loss_scale, device):
    opt = torch.optim.AdamW(params, lr=LR, fused=True)
    lossf = nn.BCEWithLogitsLoss()
    grads_view = None

    def step():
        nonlocal grads_view
        r = torch.randint(0, ntr, (BS,), device=device)
        out = forward(params, X16, r.to(torch.int32), master)
        loss = lossf(out.ravel().float(), y[r + lo])
        opt.zero_grad(set_to_none=True)
        (loss * loss_scale).backward() if loss_scale != 1.0 else loss.backward()
        if loss_scale != 1.0:
            torch._foreach_div_([p.grad for p in params], loss_scale)
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        return loss.detach()

    return step


def run_mode(mode, X16, y, ntr, lo, device):
    master = mode.startswith("master")
    loss_scale = float(os.environ.get("BNCH_LOSS_SCALE", "256")) if mode == "master_ls" else 1.0
    dtype = torch.float32 if master else torch.float16
    print(f"\n=== MOD: {mode}  (param dtype={dtype}, loss_scale={loss_scale}) ===")

    # --- SPEED ---
    params = build_params(device, dtype)
    step = make_step(params, X16, y, ntr, lo, master, loss_scale, device)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARM):
        step()
    rates = []
    for trial in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        rates.append(ITERS * BS / dt)
        print(f"  t{trial}: {rates[-1]:,.0f} samples/s ({dt:.2f}s)")
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"  SPEED median: {sorted(rates)[1]:,.0f} samples/s | peak alloc {peak:.0f} MB")

    # --- STABILITY --- (fresh params, learnable label, same lr)
    params = build_params(device, dtype, seed=1)
    step = make_step(params, X16, y, ntr, lo, master, loss_scale, device)
    trace = torch.zeros(STAB_STEPS // STAB_LOG, device=device)
    for i in range(STAB_STEPS):
        l = step()
        if i % STAB_LOG == STAB_LOG - 1:
            trace[i // STAB_LOG] = l
    torch.cuda.synchronize()
    tr = trace.tolist()
    finite = all(np.isfinite(tr))
    wmax = max(p.detach().abs().max().item() for p in params)
    print(f"  loss @{STAB_LOG}..{STAB_STEPS}: " + " ".join(f"{v:.4f}" for v in tr))
    print(f"  finite={finite} decreasing={finite and tr[-1] < tr[0]} |w|max={wmax:.3f}")
    return sorted(rates)[1], finite, tr


def main():
    require_gpu()
    device = torch.device("cuda")
    print("torch", torch.__version__, "| tunable:", torch.cuda.tunable.is_enabled(),
          "| hipblaslt env:", os.environ.get("PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED", "(default)"))

    rng = np.random.default_rng(0)
    X = torch.from_numpy(rng.standard_normal((N, F)).astype(np.float32)).to(device)
    # learnable label: f0 of the window's last row + f1 of the previous row
    y = ((X[:, 0] + 0.5 * torch.roll(X[:, 1], 1)) > 0).float()
    X16 = X.half()
    lo, ntr = W - 1, N - (W - 1)

    modes = os.environ.get("BNCH_MODE", "fp16,master,master_ls").split(",")
    results = {m: run_mode(m.strip(), X16, y, ntr, lo, device) for m in modes}

    print("\n=== SUMMARY ===")
    base = results.get("fp16", (None,))[0]
    for m, (rate, finite, tr) in results.items():
        rel = f"{rate / base * 100:5.1f}%" if base else "  -  "
        print(f"  {m:10s} {rate:>12,.0f} samples/s  ({rel} of fp16)  "
              f"finite={finite}  last_loss={tr[-1]:.4f}")


if __name__ == "__main__":
    main()
