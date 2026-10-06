"""bench10_lm: GPT training speed — naive PyTorch baseline vs fxlm (fused HIP kernels).

Each variant runs in a SEPARATE subprocess (the single-GPU-process rule), TunableOp is OFF by default.
Speed: 8 warm-up steps (including hiprtc compile) + 20 measured steps, with synchronize.

Run (from the repo root, NOT with GPU_PYTHON.cmd — that turns TunableOp on):
  venv-rocm10\\Scripts\\python.exe ml\\bench\\bench10_lm.py              speed (3 configs x variants)
  venv-rocm10\\Scripts\\python.exe ml\\bench\\bench10_lm.py --converge   learning quality (byte-level text)
  venv-rocm10\\Scripts\\python.exe ml\\bench\\bench10_lm.py --tunable    + fxlm TunableOp (first run takes minutes)

Variants:
  baseline        autocast fp16 + SDPA(MATH) + LayerNorm + nn.Linear + F.cross_entropy(fp32)
                  + GradScaler + AdamW(fused) + clip — a typical PyTorch GPT
  fxlm            flash attention HIP + fused RMSNorm + fused linear-CE + FlatAdamW (fp32 master,
                  pure fp16 weights) — same size as baseline (tie_embeddings=False)
  fxlm_sdpa / fxlm_torchnorm / fxlm_torchce   fxlm with a single component reverted to PyTorch (ablation)
  fxlm_tied       the fxlm default (shared embedding weight, ~4M fewer parameters)
"""
import json
import math
import os
import subprocess
import sys
import time

ML = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.dirname(ML)
sys.path.insert(0, ML)

CONFIGS = {
    "T256": dict(V=8192, D=512, L=8, H=8, T=256, B=16),
    "T512": dict(V=8192, D=512, L=8, H=8, T=512, B=8),
    "T1024": dict(V=8192, D=512, L=8, H=8, T=1024, B=4),
    "T256B32": dict(V=8192, D=512, L=8, H=8, T=256, B=32),
    "T512B16": dict(V=8192, D=512, L=8, H=8, T=512, B=16),
}
VARIANTS = ["baseline", "fxlm", "fxlm_sdpa", "fxlm_torchnorm", "fxlm_torchce", "fxlm_tied"]
WARM, ITERS = 8, 20
CONV = dict(V=256, D=384, L=6, H=6, T=256, B=16, steps=600, lr=1e-3, warmup=60, eval_every=150,
            eval_batches=16, corpus_mb=24)


# ---------------------------------------------------------------- baseline model
def build_baseline(c, gpt2_init=False):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    V, D, L, H, T = c["V"], c["D"], c["L"], c["H"], c["T"]

    class Block(nn.Module):
        def __init__(s):
            super().__init__()
            s.ln1 = nn.LayerNorm(D); s.ln2 = nn.LayerNorm(D)
            s.qkv = nn.Linear(D, 3 * D, bias=False); s.proj = nn.Linear(D, D, bias=False)
            s.fc = nn.Linear(D, 4 * D, bias=False); s.fc2 = nn.Linear(4 * D, D, bias=False)

        def forward(s, x):
            Bb, Tt, _ = x.shape
            q, k, v = s.qkv(s.ln1(x)).view(Bb, Tt, 3, H, D // H).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + s.proj(y.transpose(1, 2).reshape(Bb, Tt, D))
            return x + s.fc2(F.gelu(s.fc(s.ln2(x)), approximate="tanh"))

    class GPT(nn.Module):
        def __init__(s):
            super().__init__()
            s.emb = nn.Embedding(V, D); s.pos = nn.Parameter(torch.zeros(T, D))
            s.blocks = nn.ModuleList(Block() for _ in range(L)); s.lnf = nn.LayerNorm(D)
            s.head = nn.Linear(D, V, bias=False)

        def forward(s, idx):
            x = s.emb(idx) + s.pos[:idx.shape[1]]
            for b in s.blocks:
                x = b(x)
            return s.head(s.lnf(x))

    m = GPT()
    if gpt2_init:
        for n, p in m.named_parameters():
            if p.dim() >= 2:
                std = 0.02 / math.sqrt(2 * L) if n.endswith(("proj.weight", "fc2.weight")) else 0.02
                nn.init.normal_(p, 0.0, std)
    return m


class BaselineRunner:
    def __init__(self, c, lr, gpt2_init=False):
        import torch
        import torch.nn.functional as F
        self.torch, self.F, self.V = torch, F, c["V"]
        self.model = build_baseline(c, gpt2_init).cuda()
        decay = [p for p in self.model.parameters() if p.dim() >= 2]
        other = [p for p in self.model.parameters() if p.dim() < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1},
                                      {"params": other, "weight_decay": 0.0}],
                                     lr=lr, betas=(0.9, 0.95), fused=True)
        self.scaler = torch.amp.GradScaler("cuda")

    def step(self, x, y, lr):
        torch, F = self.torch, self.F
        for g in self.opt.param_groups:
            g["lr"] = lr
        with torch.autocast("cuda", dtype=torch.float16):
            loss = F.cross_entropy(self.model(x).view(-1, self.V).float(), y.reshape(-1))
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        return loss.detach()

    def logits(self, x):
        with self.torch.autocast("cuda", dtype=self.torch.float16):
            return self.model(x)

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())


class FxlmRunner:
    def __init__(self, c, lr, tie=False, attn="flash", norm="fused", ce="fused"):
        import torch
        from fxlm.model import GPT, GPTConfig
        from fxlm.optim import FlatAdamW
        self.torch = torch
        cfg = GPTConfig(vocab_size=c["V"], seq_len=c["T"], n_layer=c["L"], n_head=c["H"],
                        d_model=c["D"], tie_embeddings=tie, attn_impl=attn, norm_impl=norm, ce_impl=ce)
        self.model = GPT(cfg).cuda()
        self._np = sum(p.numel() for p in self.model.parameters())
        self.opt = FlatAdamW(self.model, lr=lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
                             max_grad_norm=1.0)

    def step(self, x, y, lr):
        loss = self.model(x, y, grad_scale=self.opt.scale)
        self.opt.backward(loss)
        self.opt.step(lr=lr)
        self.opt.zero_grad()
        return loss.detach()

    def logits(self, x):
        return self.model(x)

    def n_params(self):
        return self._np


def make_runner(variant, c, lr, gpt2_init=False):
    if variant == "baseline":
        return BaselineRunner(c, lr, gpt2_init)
    kw = {"fxlm": {}, "fxlm_sdpa": {"attn": "sdpa"}, "fxlm_torchnorm": {"norm": "torch"},
          "fxlm_torchce": {"ce": "torch"}, "fxlm_tied": {"tie": True},
          "fxlm_tunable": {}}[variant]
    return FxlmRunner(c, lr, **kw)


# ---------------------------------------------------------------- speed (child)
def child_speed(variant, cname):
    import torch
    c = CONFIGS[cname]
    torch.manual_seed(0)
    t_init = time.perf_counter()
    r = make_runner(variant, c, 3e-4)
    data = torch.randint(0, c["V"], (c["B"] * 32, c["T"] + 1), device="cuda")

    def batch(i):
        j = (i * c["B"]) % data.shape[0]
        s = data[j:j + c["B"]]
        return s[:, :-1], s[:, 1:]

    for i in range(WARM):
        l = r.step(*batch(i), 3e-4)
    torch.cuda.synchronize()
    t_warm = time.perf_counter() - t_init
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(ITERS):
        l = r.step(*batch(i), 3e-4)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS
    res = dict(variant=variant, config=cname, ms=dt * 1e3, tok_s=c["B"] * c["T"] / dt,
               peak_mb=torch.cuda.max_memory_allocated() / 2**20, loss=float(l),
               params_m=r.n_params() / 1e6, warm_s=t_warm)
    print("RESULT " + json.dumps(res), flush=True)


# ---------------------------------------------------------------- convergence (child)
def load_corpus(mb):
    import glob
    import numpy as np
    root = os.path.join(KB, "venv-rocm10", "Lib", "site-packages")
    buf, total = [], 0
    for p in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        with open(p, "rb") as f:
            b = f.read()
        buf.append(b)
        total += len(b)
        if total >= mb * 2**20:
            break
    arr = np.frombuffer(b"".join(buf), dtype=np.uint8).astype(np.int64)
    cut = int(len(arr) * 0.95)
    return arr[:cut], arr[cut:]


def child_converge(variant):
    import torch
    import torch.nn.functional as F
    c = CONV
    tr_np, va_np = load_corpus(c["corpus_mb"])
    tr = torch.from_numpy(tr_np).cuda()
    va = torch.from_numpy(va_np).cuda()
    torch.manual_seed(0)
    r = make_runner(variant, c, c["lr"], gpt2_init=True)
    ar = torch.arange(c["T"] + 1, device="cuda")
    g = torch.Generator(device="cuda")
    g.manual_seed(1234)
    gv = torch.Generator(device="cuda")
    gv.manual_seed(99)
    voffs = torch.randint(0, len(va) - c["T"] - 1, (c["eval_batches"], c["B"]), device="cuda", generator=gv)

    def lr_at(s):
        if s < c["warmup"]:
            return c["lr"] * (s + 1) / c["warmup"]
        pr = (s - c["warmup"]) / max(1, c["steps"] - c["warmup"])
        return c["lr"] * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * pr)))

    @torch.inference_mode()
    def evaluate():
        tot = 0.0
        for k in range(c["eval_batches"]):
            s = va[voffs[k][:, None] + ar]
            lg = r.logits(s[:, :-1])
            tot += float(F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), s[:, 1:].reshape(-1)))
        return tot / c["eval_batches"]

    evals, losses = [], torch.zeros(c["steps"], device="cuda")
    t_train = 0.0
    for s in range(c["steps"]):
        offs = torch.randint(0, len(tr) - c["T"] - 1, (c["B"],), device="cuda", generator=g)
        seq = tr[offs[:, None] + ar]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        losses[s] = r.step(seq[:, :-1], seq[:, 1:], lr_at(s))
        torch.cuda.synchronize()
        t_train += time.perf_counter() - t0
        if (s + 1) % c["eval_every"] == 0:
            evals.append((s + 1, round(evaluate(), 4)))
    lt = losses.tolist()
    res = dict(variant=variant, evals=evals, train_loss_last50=sum(lt[-50:]) / 50,
               finite=all(math.isfinite(v) for v in lt), train_s=t_train, params_m=r.n_params() / 1e6)
    print("RESULT " + json.dumps(res), flush=True)


# ---------------------------------------------------------------- driver
def run_child(args, tunable=False, timeout=300):
    env = dict(os.environ, HIP_VISIBLE_DEVICES="0", PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED="0",
               PYTORCH_TUNABLEOP_ENABLED="1" if tunable else "0")
    if tunable:
        env["PYTORCH_TUNABLEOP_FILENAME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                         "bench10_tunable.csv")
        env.setdefault("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS", "100")
    try:
        p = subprocess.run([sys.executable, "-u", os.path.abspath(__file__)] + args, env=env,
                           timeout=timeout, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return None, f"TIMEOUT {timeout}s"
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:]), ""
    tail = "\n".join((p.stdout + p.stderr).strip().splitlines()[-8:])
    return None, f"FAIL rc={p.returncode}\n{tail}"


def main():
    argv = sys.argv[1:]
    if argv and argv[0] == "--child-speed":
        return child_speed(argv[1], argv[2])
    if argv and argv[0] == "--child-converge":
        return child_converge(argv[1])

    if "--converge" in argv:
        c = CONV
        print(f"CONVERGENCE: byte-level text ({c['corpus_mb']} MB site-packages *.py), V={c['V']} D={c['D']} "
              f"L={c['L']} H={c['H']} T={c['T']} B={c['B']}, {c['steps']} steps, lr {c['lr']} warmup+cosine",
              flush=True)
        for v in ("baseline", "fxlm"):
            res, err = run_child(["--child-converge", v], timeout=1200)
            if res:
                print(f"{v:10s} params {res['params_m']:.2f}M | val loss {res['evals']} | "
                      f"train(last50) {res['train_loss_last50']:.4f} | finite={res['finite']} | "
                      f"training {res['train_s']:.1f}s", flush=True)
            else:
                print(f"{v:10s} {err}", flush=True)
        return

    variants = list(VARIANTS)
    configs = [a for a in argv if a in CONFIGS] or list(CONFIGS)
    only = [a for a in argv if a in VARIANTS]
    if only:
        variants = only
    results = {}
    for cname in configs:
        c = CONFIGS[cname]
        print(f"\n=== {cname}: V={c['V']} D={c['D']} L={c['L']} H={c['H']} T={c['T']} B={c['B']} "
              f"({c['B'] * c['T']} tok/step) ===", flush=True)
        runs = [(v, False) for v in variants]
        if "--tunable" in argv:
            runs += [("fxlm_tunable", True), ("fxlm_tunable", True)]
        for v, tun in runs:
            res, err = run_child(["--child-speed", v, cname], tunable=tun, timeout=900 if tun else 300)
            if res is None:
                print(f"  {v:15s} {err}", flush=True)
                continue
            base = results.get((cname, "baseline"))
            sp = f"{base['ms'] / res['ms']:.2f}x" if base else "-"
            results.setdefault((cname, v), res)
            print(f"  {v:15s} {res['ms']:8.1f} ms/step | {res['tok_s']:9,.0f} tok/s | {sp:>6s} | "
                  f"peak {res['peak_mb']:5.0f} MB | {res['params_m']:.1f}M param | "
                  f"warmup {res['warm_s']:.1f}s | loss {res['loss']:.3f}", flush=True)
    print("\nBENCH10 DONE", flush=True)


if __name__ == "__main__":
    main()
