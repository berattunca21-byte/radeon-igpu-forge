"""fxgpu train_template: tabular/window training script for gfx1035 (Radeon 680M).

Run (from the repo root):
    GPU_PYTHON.cmd ml\\fxgpu\\train_template.py --npz DATA.npz

Project rules (all measured, applied in this script):
  - NO DataLoader: batch = randperm + indexing on the GPU.
  - All tensors on the GPU from the start; CPU<->GPU copy only while loading the npz.
  - X.half() pure fp16 (not AMP/autocast); out.float() cast for the loss.
  - L1 = FusedGatherLinear (HIP gather_gemm): window gather + GEMM + bias in a
    single kernel; backward dW via rocBLAS mm using the Xg write-out.
  - Big batch (default 32768), loss_sum accumulates as a GPU tensor;
    loss.item()/sync only once at the end of the epoch.
  - isfinite check once every --check-every batches (a sync — keep it rare).
  - Time-ordered split: the last --test-frac is test; normalize ONLY with train
    statistics (mu/sd from X[:cut], no leakage).
  - Window semantics same as bench7: sample r = rows X[r .. r+W-1],
    label y[r+W-1]; train r in [0, cut-W+1), test r in [cut-W+1, n-W+1).
  - If F % 8 != 0 zero columns are appended to X (the kernel needs 16B-aligned float4).
  - Eval with inference_mode + rank-AUC + top-10% confidence acc.
  - AdamW(fused=True) + clip 5.0 + lr warmup + optional fp32 master
    weights (on by default) and a static loss scale.

Measured trade-off (this machine, ~400k train rows): bs=32768 ~1.3M samples/s
but only ~13 updates per epoch; bs=8192 ~0.85M/s but 4x the updates
-> AUC 0.96 in 3 epochs. If the data is < ~2M rows choose bs 8192-16384 or more
epochs. --no-master diverged on synthetic data (pure fp16 AdamW is unstable) — master
is the default, do not turn it off.
"""

import argparse
import os
import sys
import time

# add the ml\ root to the path for the fxgpu package (runs as a plain script)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxgpu.env import setup_env  # noqa: E402 — BEFORE the torch import
setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from fxgpu import WindowMLP, require_gpu  # noqa: E402


@torch.inference_mode()
def evaluate(model, X16, y, W, r_lo, r_hi, device, bs=32768):
    """rank-AUC + acc + top-10% confidence acc (port of train_gpu_optimized)."""
    model.eval()
    ps, ys = [], []
    for i in range(r_lo, r_hi, bs):
        r = torch.arange(i, min(i + bs, r_hi), device=device,
                         dtype=torch.int32)
        ps.append(torch.sigmoid(model(X16, r)).ravel().float())
        ys.append(y[r.long() + (W - 1)])
    p = torch.cat(ps)
    yv = torch.cat(ys)
    pred = p > 0.5
    acc = float((pred == (yv > 0.5)).float().mean())
    ranks = torch.empty_like(p)
    ranks[torch.argsort(p)] = torch.arange(
        1, len(p) + 1, device=device, dtype=p.dtype)
    n_pos = float((yv > 0.5).sum())
    n_neg = len(yv) - n_pos
    auc = float((ranks[yv > 0.5].sum() - n_pos * (n_pos + 1) / 2)
                / max(n_pos * n_neg, 1))
    conf = (p - 0.5).abs()
    thr = torch.quantile(conf, 0.9)
    top = conf >= thr
    acc_top = float((pred[top] == (yv[top] > 0.5)).float().mean())
    return {"acc": acc, "auc": auc, "acc_top": acc_top,
            "n_top": int(top.sum()), "n": len(yv)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True,
                    help="X (n,f), y (n,) 0/1, optional ts, cols")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=-1,
                    help="-1 -> auto: min(200, max(10, total_steps//10))")
    ap.add_argument("--no-master", action="store_true",
                    help="turn off fp32 master weights (pure fp16)")
    ap.add_argument("--loss-scale", type=float, default=1.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()

    device = require_gpu()
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    # --- data: npz -> pad -> train-stats normalize -> GPU fp16 ---
    d = np.load(args.npz)
    X = np.asarray(d["X"], dtype=np.float32)
    y_np = np.asarray(d["y"], dtype=np.float32)
    cols = [str(c) for c in d["cols"]] if "cols" in d else \
        [f"f{i}" for i in range(X.shape[1])]
    n, F = X.shape
    n_pad = (8 - F % 8) % 8
    if n_pad:
        X = np.concatenate(
            [X, np.zeros((n, n_pad), dtype=np.float32)], axis=1)
        cols += [f"_pad{i}" for i in range(n_pad)]
        print(f"F={F} -> {F + n_pad} ({n_pad} zero columns, 16B alignment)")
    Fp = F + n_pad

    cut = int(n * (1.0 - args.test_frac))
    X = torch.from_numpy(X).to(device)
    mu = X[:cut].mean(0)
    sd = X[:cut].std(0) + 1e-8
    X16 = ((X - mu) / sd).half().contiguous()
    del X
    y = torch.from_numpy(y_np).to(device)
    print(f"On GPU: {n:,} x {Fp} fp16 "
          f"({X16.numel() * 2 / 2**20:.0f} MB) | cut={cut:,}")

    W = args.window
    tr_hi = cut - W + 1                      # train r in [0, tr_hi)
    te_lo, te_hi = tr_hi, n - W + 1          # test  r in [te_lo, te_hi)
    if tr_hi <= 0 or te_hi <= te_lo:
        raise SystemExit(f"invalid window/split: n={n} cut={cut} W={W}")
    ntr, nte = tr_hi, te_hi - te_lo

    master = not args.no_master
    model = WindowMLP(Fp, W, hidden=args.hidden, hidden2=args.hidden2,
                      master=master, dropout=args.dropout).to(device)
    params = list(model.parameters())
    print(f"train {ntr:,} test {nte:,} | master={master} "
          f"params={sum(p.numel() for p in params):,}")

    opt = torch.optim.AdamW(params, lr=args.lr,
                            weight_decay=args.weight_decay, fused=True)
    lossf = nn.BCEWithLogitsLoss()
    bs = args.batch
    steps_per_epoch = (ntr + bs - 1) // bs
    total_steps = args.epochs * steps_per_epoch
    if args.warmup_steps < 0:
        warmup = min(200, max(10, total_steps // 10))
        wtag = " (auto)"
    else:
        warmup = args.warmup_steps
        wtag = ""
    print(f"warmup={warmup}{wtag} | {steps_per_epoch} steps/epoch, "
          f"{total_steps} total steps")
    metrics = {}
    gstep = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)   # accumulates on the GPU
        nb = 0
        for i in range(0, ntr, bs):
            r = perm[i:i + bs]
            out = model(X16, r.to(torch.int32))
            loss = lossf(out.ravel().float(), y[r + (W - 1)])
            opt.zero_grad(set_to_none=True)
            if args.loss_scale != 1.0:
                (loss * args.loss_scale).backward()
                torch._foreach_div_(
                    [p.grad for p in params if p.grad is not None],
                    args.loss_scale)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            if warmup > 0:
                warm = min(1.0, (gstep + 1) / warmup)
                for gparam in opt.param_groups:
                    gparam["lr"] = args.lr * warm
            opt.step()
            gstep += 1
            loss_sum += loss.detach()                 # NO float()
            nb += 1
            if nb % args.check_every == 0 \
                    and not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"NaN/Inf loss (ep{ep + 1} batch{nb}) — stopped")
        torch.cuda.synchronize()                      # single sync: end of epoch
        dt = time.perf_counter() - t0
        metrics = evaluate(model, X16, y, W, te_lo, te_hi, device)
        metrics["train_loss"] = float(loss_sum) / nb
        print(f"ep{ep + 1}: loss={metrics['train_loss']:.4f} "
              f"acc={metrics['acc']:.4f} auc={metrics['auc']:.4f} "
              f"top10={metrics['acc_top']:.4f} "
              f"[{dt:.1f}s {ntr / dt:,.0f}/s]", flush=True)

    torch.save({"state": model.state_dict(),
                "mu": mu.cpu().numpy(), "sd": sd.cpu().numpy(),
                "window": W, "n_pad": n_pad, "cols": cols,
                "hidden": args.hidden, "hidden2": args.hidden2,
                "master": master, "metrics": metrics,
                "args": vars(args)}, args.out)
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
