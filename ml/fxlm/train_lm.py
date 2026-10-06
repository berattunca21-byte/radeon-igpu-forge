"""fxlm.train_lm: small GPT training on gfx1035 (the main entry point).

Run (from the repo root):
    venv-rocm10\\Scripts\\python.exe ml\\fxlm\\prepare_bytes.py --out data\\demo <inputs...>
    venv-rocm10\\Scripts\\python.exe ml\\fxlm\\train_lm.py --data data\\demo_train.bin

Rules: the token sequence lives on the GPU; batch = device randint + index arithmetic
(no DataLoader); FlatAdamW fp32 master + fp16 param + dynamic scale;
one sync per log-every; atomic checkpoint (tmp + os.replace).
--tunable: the first tuning run takes minutes and the screen may hang —
the CSV is stored under ml\\fxlm\\tunable_cache\\ keyed by a shape hash.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True,
                        help="uint16 token .bin (output of prepare_bytes.py)")
    ap.add_argument("--val", default=None, help="val .bin (optional)")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--init-scale", type=float, default=2.0 ** 16)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", default=None, help="checkpoint directory (required)")
    ap.add_argument("--resume", default=None, help="checkpoint file")
    ap.add_argument("--tunable", action="store_true")
    ap.add_argument("--attn", choices=["flash", "sdpa"], default="flash")
    ap.add_argument("--norm", choices=["fused", "torch"], default="torch")
    ap.add_argument("--ce", choices=["fused", "torch"], default="fused")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = _parse()
    if args.batch * args.seq_len > 16384:
        raise SystemExit(
            f"batch*seq_len = {args.batch * args.seq_len} > 16384 — "
            "this many tokens in a single micro-step is risky for gfx1035; "
            "lower --batch or use --accum.")
    if args.out is None:
        raise SystemExit("--out <checkpoint directory> is required")

    tune_file = None
    if args.tunable:
        sig = f"v{0}-d{args.dim}-l{args.layers}-h{args.heads}-" \
              f"t{args.seq_len}-b{args.batch}"
        h = hashlib.sha1(sig.encode()).hexdigest()[:12]
        tdir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "tunable_cache")
        os.makedirs(tdir, exist_ok=True)
        tune_file = os.path.join(tdir, f"tune_{h}.csv")
        print("WARNING: --tunable is on — the first tuning run can take several "
              "minutes and the screen may hang. CSV: " + tune_file,
              flush=True)

    from fxlm.env import setup_lm_env
    setup_lm_env(tunable=args.tunable, tune_file=tune_file)

    import numpy as np
    import torch
    from fxlm.env import require_gpu
    from fxlm.model import GPT, GPTConfig
    from fxlm.optim import FlatAdamW

    device = require_gpu()
    print(f"TunableOp enabled: {torch.cuda.tunable.is_enabled()} "
          f"(default: off — a long tuning run froze this PC once)",
          flush=True)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)

    # --- meta + data ---------------------------------------------------------
    prefix = args.data[:-len("_train.bin")] \
        if args.data.endswith("_train.bin") \
        else os.path.splitext(args.data)[0]
    meta_path = prefix + ".json"
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    vocab = int(meta.get("vocab_size", 65536))
    print(f"meta: {meta_path} -> vocab_size={vocab} "
          f"tokenizer={meta.get('tokenizer', '?')}")

    tr = np.fromfile(args.data, dtype=np.uint16)
    tokens = torch.from_numpy(tr.astype(np.int32)).to(device)
    ntr = tokens.numel()
    vt = None
    if args.val:
        vv = np.fromfile(args.val, dtype=np.uint16)
        vt = torch.from_numpy(vv.astype(np.int32)).to(device)
    elif meta.get("n_val"):
        vp = prefix + "_val.bin"
        if os.path.exists(vp):
            vv = np.fromfile(vp, dtype=np.uint16)
            vt = torch.from_numpy(vv.astype(np.int32)).to(device)
    print(f"train {ntr:,} tokens" +
          (f" | val {vt.numel():,}" if vt is not None else ""))

    cfg = GPTConfig(vocab_size=vocab, seq_len=args.seq_len,
                    n_layer=args.layers, n_head=args.heads,
                    d_model=args.dim, attn_impl=args.attn,
                    norm_impl=args.norm, ce_impl=args.ce)
    model = GPT(cfg).to(device)
    print(f"GPT params: {model.num_params():,} ({model.num_params()/1e6:.1f}M)")

    opt = FlatAdamW(model, lr=args.lr, weight_decay=args.wd,
                    max_grad_norm=args.clip, init_scale=args.init_scale)

    ckpt_path = os.path.join(args.out, "ckpt_last.pt")
    start_step = 0
    if args.resume:
        sd = torch.load(args.resume, map_location=device,
                        weights_only=False)
        opt.load_state_dict(sd["opt"])   # master/m/v/scale + refresh flat16
        start_step = sd["step"]
        print(f"resume: {args.resume} @ step {start_step}")
    elif os.path.exists(ckpt_path):
        raise SystemExit(f"{ckpt_path} already exists — protect old runs: "
                         "choose another --out or use --resume.")
    os.makedirs(args.out, exist_ok=True)

    T, B, ACC = args.seq_len, args.batch, args.accum
    ar = torch.arange(T + 1, device=device)

    def get_batch(src):
        offs = torch.randint(0, src.numel() - T - 1, (B,),
                             device=device)
        seq = src[(offs[:, None] + ar[None, :])].long()
        return seq[:, :-1], seq[:, 1:]

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        c = 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))
        return args.lr * (args.min_lr_frac
                          + (1.0 - args.min_lr_frac) * c)

    def save_ckpt(step, tag="last"):
        sd = {
            "model": {k: v.detach().to("cpu", torch.float32)
                      for k, v in opt.master_state_dict().items()},
            "opt": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                    for k, v in opt.state_dict().items()},
            "config": cfg.__dict__, "step": step, "meta": meta,
            "args": vars(args),
        }
        tmp = os.path.join(args.out, f"ckpt_{tag}.tmp")
        final = os.path.join(args.out, f"ckpt_{tag}.pt")
        torch.save(sd, tmp)
        os.replace(tmp, final)

    @torch.inference_mode()
    def evaluate():
        model.eval()
        tot = torch.zeros((), device=device)
        for _ in range(args.eval_batches):
            x, y = get_batch(vt)
            tot += model(x, y)
        model.train()
        return float(tot) / args.eval_batches

    # --- training loop -------------------------------------------------------
    print(f"steps: {args.steps} | micro-batch {B}x{T} x{ACC} accum | "
          f"lr {args.lr} warmup {args.warmup}", flush=True)
    loss_acc = torch.zeros((), device=device)
    skipped = 0
    nb = 0
    t0 = None
    gstep = start_step
    while gstep < args.steps:
        if t0 is None:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        lr = lr_at(gstep)
        for _ in range(ACC):
            x, y = get_batch(tokens)
            loss = model(x, y, grad_scale=opt.scale)
            opt.backward(loss / ACC)
            loss_acc += loss.detach() / ACC
            nb += 1
        stats = opt.step(lr=lr)
        opt.zero_grad()
        gstep += 1
        if gstep % args.log_every == 0 or gstep == 1:
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            toks = nb * B * T
            gn = float(stats["grad_norm"])
            sc = float(stats["scale"])
            fi = bool(stats["found_inf"])
            skipped += int(fi)
            print(f"step {gstep:5d} | loss {float(loss_acc) / nb:.4f} | "
                  f"lr {lr:.2e} | {toks / dt:,.0f} tok/s | "
                  f"gnorm {gn:.3f} | scale {sc:.0f} | skip {skipped}",
                  flush=True)
            loss_acc = torch.zeros((), device=device)
            nb = 0
            t0 = None
        if vt is not None and (gstep % args.eval_every == 0
                               or gstep == args.steps):
            vl = evaluate()
            print(f"  eval @ {gstep}: val_loss {vl:.4f}", flush=True)
            save_ckpt(gstep)
    save_ckpt(gstep)
    print(f"done: ckpt -> {os.path.join(args.out, 'ckpt_last.pt')}",
          flush=True)


if __name__ == "__main__":
    main()
