"""fxlm.bench_kernels: each fused kernel vs its PyTorch equivalent (gfx1035).

    venv-rocm10\\Scripts\\python.exe ml\\fxlm\\bench_kernels.py

Results are written to ml\\fxlm\\bench_kernels_out.txt.
TunableOp OFF (setup_lm_env). All times are synchronized average ms.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxlm.env import setup_lm_env  # noqa: E402  (before torch)
setup_lm_env(tunable=False)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.nn as nn  # noqa: E402

from fxlm.norm import rms_norm  # noqa: E402
from fxlm.cross_entropy import fused_cross_entropy, linear_cross_entropy  # noqa: E402
from fxlm.attention import flash_attention  # noqa: E402
from fxlm.optim import FlatAdamW  # noqa: E402
from fxlm.model import GPT, GPTConfig  # noqa: E402

_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "bench_kernels_out.txt")
_fout = open(_OUT, "w", encoding="utf-8")


def out(s=""):
    print(s, flush=True)
    _fout.write(s + "\n")
    _fout.flush()


def bench(fn, iters=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def bench_attn(dev):
    out("=== flash attention vs SDPA(math) — fwd | fwd+bwd ms ===")
    for (B, T, H, Dh) in [(16, 256, 8, 64), (8, 512, 8, 64),
                          (4, 1024, 8, 64), (16, 256, 8, 128)]:
        q = torch.randn(B, T, H, Dh, device=dev, dtype=torch.float16,
                        requires_grad=True)
        k = q.clone().requires_grad_()
        v = q.clone().requires_grad_()
        do = torch.randn(B, T, H, Dh, device=dev, dtype=torch.float16)

        def f_fwd():
            with torch.no_grad():
                flash_attention(q, k, v, causal=True)

        def f_fb():
            o = flash_attention(q, k, v, causal=True)
            o.backward(do)
            q.grad = k.grad = v.grad = None

        qs = torch.randn(B, H, T, Dh, device=dev, dtype=torch.float16,
                         requires_grad=True)
        ks = qs.clone().requires_grad_()
        vs = qs.clone().requires_grad_()
        dos = torch.randn(B, H, T, Dh, device=dev, dtype=torch.float16)

        def s_fwd():
            with torch.no_grad():
                F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)

        def s_fb():
            o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True)
            o.backward(dos)
            qs.grad = ks.grad = vs.grad = None

        tf, tfb = bench(f_fwd), bench(f_fb)
        tsf, tsfb = bench(s_fwd), bench(s_fb)
        out(f"attn B{B} T{T} H{H} Dh{Dh}: flash {tf:7.2f} | {tfb:7.2f} ms  "
            f"sdpa {tsf:7.2f} | {tsfb:7.2f} ms  "
            f"speedup fwd {tsf/tf:.2f}x fb {tsfb/tfb:.2f}x")
        del q, k, v, do, qs, ks, vs, dos
        torch.cuda.empty_cache()


def bench_norm(dev):
    out("=== rms_norm fwd+bwd vs F.rms_norm — ms ===")
    for (N, D) in [(8192, 768), (8192, 512)]:
        x = torch.randn(N, D, device=dev, dtype=torch.float16,
                        requires_grad=True)
        w = torch.ones(D, device=dev, dtype=torch.float16,
                       requires_grad=True)
        dy = torch.randn(N, D, device=dev, dtype=torch.float16)

        def f_fb():
            y = rms_norm(x, w)
            y.backward(dy)
            x.grad = w.grad = None

        def t_fb():
            y = F.rms_norm(x, (D,), w)
            y.backward(dy)
            x.grad = w.grad = None

        tf, tt = bench(f_fb), bench(t_fb)
        out(f"rms {N}x{D}: fused {tf:6.2f} ms | torch {tt:6.2f} ms  "
            f"speedup {tt/tf:.2f}x")


def bench_ce(dev):
    out("=== cross_entropy fwd+bwd — ms ===")
    for (N, V) in [(4096, 8192), (4096, 16384)]:
        lg = torch.randn(N, V, device=dev, dtype=torch.float16,
                         requires_grad=True)
        tg = torch.randint(0, V, (N,), device=dev)

        def f_fb():
            fused_cross_entropy(lg, tg).backward()
            lg.grad = None

        def t_fb():
            F.cross_entropy(lg.float(), tg).backward()
            lg.grad = None

        tf, tt = bench(f_fb, iters=10, warm=3), bench(t_fb, iters=10, warm=3)
        out(f"ce {N}x{V}: fused {tf:6.2f} ms | torch-fp32 {tt:6.2f} ms  "
            f"speedup {tt/tf:.2f}x")
        del lg
        torch.cuda.empty_cache()

    out("=== linear_cross_entropy (fused head+CE) vs linear+CE — ms ===")
    for (N, D, V) in [(4096, 512, 8192)]:
        x = torch.randn(N, D, device=dev, dtype=torch.float16,
                        requires_grad=True)
        W = (torch.randn(V, D, device=dev) * 0.02).half()
        W.requires_grad_()
        tg = torch.randint(0, V, (N,), device=dev)

        def f_fb():
            linear_cross_entropy(x, W, tg).backward()
            x.grad = W.grad = None

        def t_fb():
            logits = F.linear(x, W)
            F.cross_entropy(logits.float(), tg).backward()
            x.grad = W.grad = None

        tf, tt = bench(f_fb, iters=10, warm=3), bench(t_fb, iters=10, warm=3)
        out(f"linearCE {N}x{D}x{V}: fused {tf:6.2f} ms | "
            f"torch {tt:6.2f} ms  speedup {tt/tf:.2f}x")


def bench_optim(dev):
    out("=== FlatAdamW.step vs GradScaler+clip+AdamW(fused) — ms ===")
    cfg = GPTConfig(vocab_size=8192, seq_len=256, n_layer=8, n_head=8,
                    d_model=512)
    m = GPT(cfg).to(dev)
    n = m.num_params()
    opt = FlatAdamW(m, lr=3e-4, weight_decay=0.1, max_grad_norm=1.0)
    ps = [p for p in m.parameters()]
    torch.manual_seed(0)
    for p in ps:
        p.grad = torch.randn(p.shape, device=dev).half()

    def f_step():
        opt.step(lr=3e-4)

    tf = bench(f_step, iters=10, warm=3)

    # torch path: fp32 param + scaler unscale + clip + fused AdamW
    m2 = GPT(cfg).to(dev)
    ps2 = list(m2.parameters())
    opt2 = torch.optim.AdamW(ps2, lr=3e-4, betas=(0.9, 0.95), fused=True)
    scaler = torch.amp.GradScaler("cuda")
    for p in ps2:
        p.grad = torch.randn(p.shape, device=dev)
    dummy = torch.ones((), device=dev)

    def t_step():
        scaler.scale(dummy)          # scale() is required before unscale_
        scaler.unscale_(opt2)
        torch.nn.utils.clip_grad_norm_(ps2, 1.0)
        scaler.step(opt2)
        scaler.update()

    tt = bench(t_step, iters=10, warm=3)
    out(f"adamw {n/1e6:.1f}M param: FlatAdamW {tf:6.2f} ms | "
        f"torch path {tt:6.2f} ms  speedup {tt/tf:.2f}x")


def main():
    assert torch.cuda.is_available(), "no CUDA"
    dev = "cuda"
    torch.manual_seed(0)
    out(f"fxlm bench_kernels — {torch.cuda.get_device_name(0)} "
        f"(TunableOp OFF) {time.strftime('%Y-%m-%d %H:%M')}")
    bench_attn(dev)
    bench_norm(dev)
    bench_ce(dev)
    bench_optim(dev)
    out("BENCH KERNELS DONE")
    _fout.close()


if __name__ == "__main__":
    main()
