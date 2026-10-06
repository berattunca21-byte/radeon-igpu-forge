"""fxgpu train_template: gfx1035 (Radeon 680M) icin tablo/pencere egitim scripti.

Calistirma (KB kokunden):
    GPU_PYTHON.cmd ml\\fxgpu\\train_template.py --npz VERI.npz

KB kurallari (hepsi olculmus, bu scriptte uygulanir):
  - DataLoader YOK: batch = GPU'da randperm + indexleme.
  - Tum tensorlar bastan GPU'da; CPU<->GPU kopya sadece npz yuklenirken.
  - X.half() saf fp16 (AMP/autocast degil); loss'a out.float() cast.
  - L1 = FusedGatherLinear (HIP gather_gemm): pencere gather + GEMM + bias
    tek kernel; Xg write-out ile backward dW rocBLAS mm'de.
  - Buyuk batch (varsayilan 32768), loss_sum GPU tensoru olarak birikir;
    loss.item()/sync sadece epoch sonunda bir kez.
  - isfinite kontrolu --check-every batch'te bir (sync — nadir tutulur).
  - Zaman-sirali split: son --test-frac test; normalize SADECE train
    istatistikleriyle (mu/sd X[:cut]'tan, sizma yok).
  - Pencere semantigi bench7 ile ayni: ornek r = X[r .. r+W-1] satirlari,
    etiket y[r+W-1]; train r in [0, cut-W+1), test r in [cut-W+1, n-W+1).
  - F % 8 != 0 ise X'e sifir sutun eklenir (kernel 16B-hizali float4 ister).
  - Eval inference_mode + rank-AUC + top-10% guven acc.
  - AdamW(fused=True) + clip 5.0 + lr warmup + opsiyonel fp32 master
    agirliklar (varsayilan acik) ve statik loss-scale.

Olculmus denge (bu makine, ~400k satir train): bs=32768 ~1.3M ornek/sn
ama epok basina sadece ~13 guncelleme; bs=8192 ~0.85M/sn ama 4x guncelleme
-> 3 epokta AUC 0.96. Veri < ~2M satirsa bs 8192-16384 veya daha cok epok
sec. --no-master synth veride iraksadi (saf fp16 AdamW kararsiz) — master
varsayilan, kapatma.
"""

import argparse
import os
import sys
import time

# fxgpu paketi icin ml\ kokunu path'e ekle (duz script olarak calisir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxgpu.env import setup_env  # noqa: E402 — torch import'undan ONCE
setup_env()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from fxgpu import WindowMLP, require_gpu  # noqa: E402


@torch.inference_mode()
def evaluate(model, X16, y, W, r_lo, r_hi, device, bs=32768):
    """rank-AUC + acc + top-10% guven acc (train_gpu_optimized portu)."""
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
                    help="X (n,f), y (n,) 0/1, opsiyonel ts, cols")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=-1,
                    help="-1 -> auto: min(200, max(10, toplam_adim//10))")
    ap.add_argument("--no-master", action="store_true",
                    help="fp32 master agirliklari kapat (saf fp16)")
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

    # --- veri: npz -> pad -> train-stats normalize -> GPU fp16 ---
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
        print(f"F={F} -> {F + n_pad} ({n_pad} sifir sutun, 16B hiza)")
    Fp = F + n_pad

    cut = int(n * (1.0 - args.test_frac))
    X = torch.from_numpy(X).to(device)
    mu = X[:cut].mean(0)
    sd = X[:cut].std(0) + 1e-8
    X16 = ((X - mu) / sd).half().contiguous()
    del X
    y = torch.from_numpy(y_np).to(device)
    print(f"GPU'da: {n:,} x {Fp} fp16 "
          f"({X16.numel() * 2 / 2**20:.0f} MB) | cut={cut:,}")

    W = args.window
    tr_hi = cut - W + 1                      # train r in [0, tr_hi)
    te_lo, te_hi = tr_hi, n - W + 1          # test  r in [te_lo, te_hi)
    if tr_hi <= 0 or te_hi <= te_lo:
        raise SystemExit(f"pencere/split gecersiz: n={n} cut={cut} W={W}")
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
    print(f"warmup={warmup}{wtag} | {steps_per_epoch} adim/epok, "
          f"{total_steps} toplam adim")
    metrics = {}
    gstep = 0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(ntr, device=device)
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)   # GPU'da birikim
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
            loss_sum += loss.detach()                 # float() YOK
            nb += 1
            if nb % args.check_every == 0 \
                    and not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"NaN/Inf loss (ep{ep + 1} batch{nb}) — durduruldu")
        torch.cuda.synchronize()                      # tek sync: epoch sonu
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
    print(f"kaydedildi: {args.out}")


if __name__ == "__main__":
    main()
