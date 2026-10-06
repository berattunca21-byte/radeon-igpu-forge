"""fxlm self-test: fused kernel'arin PyTorch referanslariyla parity'si.

Kapsam (hepsi fp16 io, fp32 referans + torch-fp16 hata cubugu kurali
ile: bizim max-abs hata <= 2 * torch-fp16 hata + 1e-3):
  - rms_norm fwd/bwd (birka D ve satir sayisi; dx + dweight)
  - fused_cross_entropy (ignore_index satirli)
  - linear_cross_entropy (tensor grad_scale, ignore_index, N % chunk != 0,
    grad-kapali sadece-loss yolu)
  - flash attention fwd/bwd: causal + non-causal, Dh 64/128,
    tek T degerleri (1, 17, 513), strided qkv gorunumleri, qkv varyanti
  - FlatAdamW: grad-enjeksiyon parity vs torch AdamW+clip (rel < 1e-5,
    fp16 == master.half()), inf atlama (master/m/v/step ayni, scale
    yarilanir), temiz adimlarda scale buyumesi
  - minik GPT: fused vs torch impl ayni fp16 agirliklarla (loss + tum
    grad'lar kuralda), sonra 20 FlatAdamW adimi loss'u dusurmeli

Calistirma (KB kokunden):
    venv-rocm10\\Scripts\\python.exe ml\\fxlm\\selftest.py
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxlm.env import setup_lm_env  # noqa: E402  (torch'tan once)
setup_lm_env(tunable=False)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.nn as nn  # noqa: E402

from fxlm.norm import rms_norm  # noqa: E402
from fxlm.cross_entropy import fused_cross_entropy, linear_cross_entropy  # noqa: E402
from fxlm.attention import flash_attention, flash_attention_qkv  # noqa: E402
from fxlm.optim import FlatAdamW  # noqa: E402
from fxlm.model import GPT, GPTConfig  # noqa: E402

FAILS = []


def report(name, ok, extra=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {extra}", flush=True)
    if not ok:
        FAILS.append(name)


def rule(our, ref32, torch16):
    """bizim hata <= 2*torch-fp16 hata + 1e-3"""
    e_our = (our.float() - ref32).abs().max().item()
    e_t16 = (torch16.float() - ref32).abs().max().item()
    return e_our <= 2 * e_t16 + 1e-3, e_our, e_t16


def test_rms_norm(dev):
    for (N, D) in [(4, 8), (37, 64), (129, 768), (1000, 512)]:
        torch.manual_seed(N * 1000 + D)
        x = (torch.randn(N, D, device=dev) * 3).half()
        w = (torch.randn(D, device=dev) * 0.5 + 1).half()
        xq, wq = x.clone().requires_grad_(), w.clone().requires_grad_()
        xr, wr = x.float().requires_grad_(), w.float().requires_grad_()
        xt, wt = x.clone().requires_grad_(), w.clone().requires_grad_()
        y = rms_norm(xq, wq, 1e-6)
        yr = F.rms_norm(xr, (D,), wr, 1e-6)
        yt = F.rms_norm(xt, (D,), wt, 1e-6)
        g = torch.randn(N, D, device=dev).half()
        y.backward(g); yr.backward(g.float()); yt.backward(g)
        ok1, e1, b1 = rule(y, yr, yt)
        ok2, e2, b2 = rule(xq.grad, xr.grad, xt.grad)
        ok3, e3, b3 = rule(wq.grad, wr.grad, wt.grad)
        report(f"rms_norm {N}x{D}", ok1 and ok2 and ok3,
               f"y {e1:.4f}/{b1:.4f} dx {e2:.4f}/{b2:.4f} "
               f"dw {e3:.4f}/{b3:.4f}")


def test_ce(dev):
    for (N, V) in [(5, 32), (100, 512), (301, 8192)]:
        torch.manual_seed(N + V)
        lg = (torch.randn(N, V, device=dev) * 4).half()
        tg = torch.randint(0, V, (N,), device=dev)
        tg[::7] = -100                        # ignore_index satirlari
        l32 = F.cross_entropy(lg.float(), tg, ignore_index=-100)
        l16 = F.cross_entropy(lg, tg, ignore_index=-100)
        lq = fused_cross_entropy(lg, tg, -100)
        ok1, e1, b1 = rule(lq, l32, l16)
        # grad parity
        lg2 = lg.clone().requires_grad_()
        lg3 = lg.clone().requires_grad_()
        lg4 = lg.clone().requires_grad_()
        fused_cross_entropy(lg2, tg, -100).backward()
        F.cross_entropy(lg3.float(), tg, ignore_index=-100).backward()
        F.cross_entropy(lg4, tg, ignore_index=-100).backward()
        ok2, e2, b2 = rule(lg2.grad, lg3.grad, lg4.grad)
        report(f"fused_ce {N}x{V}", ok1 and ok2,
               f"loss {e1:.5f}/{b1:.5f} dgrad {e2:.4f}/{b2:.4f}")


def test_linear_ce(dev):
    for (N, D, V, gs) in [(50, 64, 128, 1.0), (301, 256, 8192, 1.0),
                          (1027, 512, 4096, None)]:
        torch.manual_seed(N + D + V)
        x = torch.randn(N, D, device=dev).half()
        W = (torch.randn(V, D, device=dev) * 0.02).half()
        tg = torch.randint(0, V, (N,), device=dev)
        tg[::11] = -100
        grad_scale = (torch.tensor(65536.0, device=dev)
                      if gs is None else gs)
        xq, Wq = x.clone().requires_grad_(), W.clone().requires_grad_()
        xr, Wr = x.float().requires_grad_(), W.float().requires_grad_()
        xt, Wt = x.clone().requires_grad_(), W.clone().requires_grad_()
        lq = linear_cross_entropy(xq, Wq, tg, grad_scale=grad_scale,
                                  chunk_rows=128)   # 128 | N degil
        lr = F.cross_entropy(xr @ Wr.t(), tg, ignore_index=-100)
        lt = F.cross_entropy((xt @ Wt.t()).float(), tg, ignore_index=-100)
        ok1, e1, b1 = rule(lq, lr, lt)
        lq.backward(); lr.backward(); lt.backward()
        ok2, e2, b2 = rule(xq.grad, xr.grad, xt.grad)
        ok3, e3, b3 = rule(Wq.grad, Wr.grad, Wt.grad)
        tag = "scale=65536(tensor)" if gs is None else f"scale={gs}"
        report(f"linear_ce {N}x{D}x{V} {tag}", ok1 and ok2 and ok3,
               f"loss {e1:.5f}/{b1:.5f} dx {e2:.4f}/{b2:.4f} "
               f"dW {e3:.4f}/{b3:.4f}")
    # grad-kapali sadece-loss yolu: ayni loss, grad_fn yok
    torch.manual_seed(7)
    x = torch.randn(200, 128, device=dev).half()
    W = (torch.randn(512, 128, device=dev) * 0.02).half()
    tg = torch.randint(0, 512, (200,), device=dev)
    lg = linear_cross_entropy(x, W, tg, grad_scale=1.0)
    with torch.no_grad():
        ln = linear_cross_entropy(x, W, tg)
    ok = ln.grad_fn is None and abs(float(ln) - float(lg)) < 1e-6
    report("linear_ce no-grad path", ok,
           f"loss {float(ln):.4f} vs {float(lg):.4f}")


def _ref_attn(q, k, v, causal, scale):
    qt = q.float().permute(0, 2, 1, 3)
    kt = k.float().permute(0, 2, 1, 3)
    vt = v.float().permute(0, 2, 1, 3)
    s = qt @ kt.transpose(-1, -2) * scale
    if causal:
        T = q.shape[1]
        mask = torch.ones(T, T, device=q.device, dtype=torch.bool).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return (torch.softmax(s, -1) @ vt).permute(0, 2, 1, 3)


def _sdpa_attn(q, k, v, causal, scale):
    o = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3), is_causal=causal, scale=scale)
    return o.permute(0, 2, 1, 3)


def test_attention(dev):
    cases = [(1, 1, 1, 64), (1, 17, 1, 64), (2, 64, 2, 64),
             (1, 100, 8, 64), (3, 256, 8, 64), (2, 513, 2, 64),
             (1, 130, 4, 128), (2, 200, 8, 128), (1, 33, 1, 128)]
    for causal in (True, False):
        for (B, T, H, Dh) in cases:
            torch.manual_seed(B * 7 + T * 13 + Dh)
            scale = 1.0 / math.sqrt(Dh)
            qkv = torch.randn(B, T, 3, H, Dh, device=dev).half()
            q, k, v = (qkv[:, :, i].requires_grad_() for i in range(3))
            o = flash_attention(q, k, v, causal=causal, scale=scale)
            oref = _ref_attn(q.detach(), k.detach(), v.detach(),
                             causal, scale)
            o16 = _sdpa_attn(q.detach(), k.detach(), v.detach(),
                             causal, scale)
            ok1, e1, b1 = rule(o, oref, o16)
            g = torch.randn_like(o)
            o.backward(g)
            qr = q.detach().float().requires_grad_()
            kr = k.detach().float().requires_grad_()
            vr = v.detach().float().requires_grad_()
            _ref_attn(qr, kr, vr, causal, scale).backward(g.float())
            qt = q.detach().clone().requires_grad_()
            kt = k.detach().clone().requires_grad_()
            vt = v.detach().clone().requires_grad_()
            _sdpa_attn(qt, kt, vt, causal, scale).backward(g)
            okq, eq, bq = rule(q.grad, qr.grad, qt.grad)
            okk, ek, bk = rule(k.grad, kr.grad, kt.grad)
            okv, ev, bv = rule(v.grad, vr.grad, vt.grad)
            report(f"attn B{B} T{T} H{H} Dh{Dh} causal={causal}",
                   ok1 and okq and okk and okv,
                   f"fwd {e1:.4f}/{b1:.4f} dq {eq:.4f}/{bq:.4f} "
                   f"dk {ek:.4f}/{bk:.4f} dv {ev:.4f}/{bv:.4f}")
    # qkv varyanti: tek dqkv buffer
    for causal in (True, False):
        torch.manual_seed(5)
        B, T, H, Dh = 2, 100, 4, 64
        qkv = torch.randn(B, T, 3, H, Dh, device=dev).half()
        qkv.requires_grad_()
        o = flash_attention_qkv(qkv, causal=causal)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        scale = 1.0 / math.sqrt(Dh)
        oref = _ref_attn(q.detach(), k.detach(), v.detach(), causal, scale)
        o16 = _sdpa_attn(q.detach(), k.detach(), v.detach(), causal, scale)
        ok1, e1, b1 = rule(o, oref, o16)
        g = torch.randn_like(o)
        o.backward(g)
        qr = q.detach().float().requires_grad_()
        kr = k.detach().float().requires_grad_()
        vr = v.detach().float().requires_grad_()
        _ref_attn(qr, kr, vr, causal, scale).backward(g.float())
        qt = q.detach().clone().requires_grad_()
        kt = k.detach().clone().requires_grad_()
        vt = v.detach().clone().requires_grad_()
        _sdpa_attn(qt, kt, vt, causal, scale).backward(g)
        oks = []
        for i, (rg, tg) in enumerate(((qr, qt), (kr, kt), (vr, vt))):
            ok, _e, _b = rule(qkv.grad[:, :, i], rg.grad, tg.grad)
            oks.append(ok)
        report(f"attn_qkv causal={causal}", ok1 and all(oks),
               f"fwd {e1:.4f}/{b1:.4f}")


def test_flat_adamw(dev):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 512, bias=False),
                          nn.LayerNorm(512),
                          nn.Linear(512, 128, bias=False)).to(dev)
    opt = FlatAdamW(model, lr=1e-3, betas=(0.9, 0.95), eps=1e-8,
                    weight_decay=0.1, max_grad_norm=0.5,
                    init_scale=8.0, growth_interval=4)
    ps = [p for p in model.parameters()]
    ref = [torch.nn.Parameter(
        opt.master[off:off + n].view(p.shape).clone())
        for _nm, p, off, n, _w in opt._info]
    ropt = torch.optim.AdamW(
        [{"params": [r for r, (_n, _p, _o, _e, wd) in zip(ref, opt._info)
                    if wd], "weight_decay": 0.1},
         {"params": [r for r, (_n, _p, _o, _e, wd) in zip(ref, opt._info)
                     if not wd], "weight_decay": 0.0}],
        lr=1e-3, betas=(0.9, 0.95), eps=1e-8)
    torch.manual_seed(1)
    maxrel = 0.0
    p16ok = True
    for st in range(6):
        grads = [torch.randn(p.shape, device=dev).half() * 0.1
                 for p in ps]
        scale = float(opt.scale)
        for p, g in zip(ps, grads):
            p.grad = g
        opt.step(lr=1e-3)
        for rp, g in zip(ref, grads):
            rp.grad = g.float() / scale
        torch.nn.utils.clip_grad_norm_(ref, 0.5)
        ropt.step()
        for (nm, p, off, n, _w) in opt._info:
            rp = ref[[q is p for q in ps].index(True)]
            m_our = opt.master[off:off + n].view(p.shape)
            err = (m_our - rp.detach()).abs().max().item()
            rel = err / rp.detach().abs().max().clamp(min=1e-6).item()
            maxrel = max(maxrel, rel)
        p16ok &= all(torch.equal(
            p, opt.master[o:o + ne].view(p.shape).half())
            for _n, p, o, ne, _w in opt._info)
    report("flat_adamw parity", maxrel < 1e-5 and p16ok,
           f"master rel-err {maxrel:.2e} fp16==half {p16ok}")

    # inf -> adim atlanir, scale yarilanir, step sayaci artmaz
    sc0, st0 = float(opt.scale), int(opt._step_ctr)
    m0, mm0, vv0 = opt.master.clone(), opt.m.clone(), opt.v.clone()
    for p in ps:
        p.grad = torch.zeros(p.shape, device=dev).half()
    ps[0].grad.view(-1)[0] = float("inf")
    stats = opt.step(lr=1e-3)
    ok = (bool(stats["found_inf"])
          and float(stats["scale"]) == sc0 * 0.5
          and int(opt._step_ctr) == st0
          and torch.equal(m0, opt.master) and torch.equal(mm0, opt.m)
          and torch.equal(vv0, opt.v))
    report("flat_adamw inf skip", ok,
           f"scale {sc0}->{float(stats['scale'])} "
           f"step {st0}->{int(opt._step_ctr)}")
    opt.zero_grad()

    # growth_interval temiz adimdan sonra scale 2x
    sc0 = float(opt.scale)
    for _ in range(opt.growth_interval):
        for p in ps:
            p.grad = torch.zeros(p.shape, device=dev).half()
        opt.step(lr=1e-3)
    report("flat_adamw scale growth",
           float(opt.scale) == sc0 * 2.0,
           f"scale {sc0}->{float(opt.scale)} "
           f"(interval {opt.growth_interval})")

    # grad accumulation: iki backward birikir
    ps[0].grad = None
    g1 = torch.randn(ps[0].shape, device=dev).half()
    ps[0].grad = g1.clone()
    ps[0].grad.add_(g1)          # ikinci micro-step gibi
    ok = torch.equal(ps[0].grad, (g1.float() * 2).half())
    report("flat_adamw grad accum", ok)
    opt.zero_grad()


def test_gpt(dev):
    torch.manual_seed(0)
    cfg_f = GPTConfig(vocab_size=256, seq_len=64, n_layer=2, n_head=2,
                      d_model=128, attn_impl="flash", norm_impl="fused",
                      ce_impl="fused")
    cfg_t = GPTConfig(vocab_size=256, seq_len=64, n_layer=2, n_head=2,
                      d_model=128, attn_impl="sdpa", norm_impl="torch",
                      ce_impl="torch")
    m_f = GPT(cfg_f).to(dev)
    m_t = GPT(cfg_t).to(dev)
    m_t.load_state_dict(m_f.state_dict())
    m_r = GPT(cfg_t).to(dev)
    m_r.load_state_dict(m_f.state_dict())   # fp32 referans
    m_f.half(); m_t.half()

    B, T = 8, 64
    idx = torch.randint(0, 256, (B, T), device=dev)
    tgt = torch.randint(0, 256, (B, T), device=dev)

    l_f = m_f(idx, tgt, grad_scale=1.0)
    l_t = m_t(idx, tgt)
    l_r = m_r(idx, tgt)
    ok1, e1, b1 = rule(l_f, l_r, l_t)
    l_f.backward(); l_t.backward(); l_r.backward()
    worst = (True, 0.0, "")
    for (nm, p) in m_f.named_parameters():
        pt = dict(m_t.named_parameters())[nm]
        pr = dict(m_r.named_parameters())[nm]
        ok, e, b = rule(p.grad, pr.grad, pt.grad)
        if not ok:
            worst = (False, e, nm)
    report("gpt fused-vs-torch parity", ok1 and worst[0],
           f"loss {e1:.4f}/{b1:.4f} worst-grad {worst[2]} {worst[1]:.4f}")

    # 20 FlatAdamW adimi: loss belirgin dusmeli, skip olmamali
    torch.manual_seed(0)
    m = GPT(cfg_f).to(dev)
    opt = FlatAdamW(m, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1,
                    max_grad_norm=1.0, init_scale=8.0,
                    growth_interval=1000)
    losses = []
    skips = 0
    for s in range(20):
        loss = m(idx, tgt, grad_scale=opt.scale)
        opt.backward(loss)
        stats = opt.step(lr=3e-3)
        opt.zero_grad()
        skips += int(bool(stats["found_inf"]))
        losses.append(float(loss.detach()))
    ok = losses[-1] < losses[0] * 0.9 and skips == 0
    report("gpt 20-step learn", ok,
           f"loss {losses[0]:.3f} -> {losses[-1]:.3f} skip {skips}")


def main():
    t0 = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA yok"
    dev = "cuda"
    print(f"fxlm selftest — {torch.cuda.get_device_name(0)}", flush=True)
    test_rms_norm(dev)
    test_ce(dev)
    test_linear_ce(dev)
    test_attention(dev)
    test_flat_adamw(dev)
    test_gpt(dev)
    dt = time.perf_counter() - t0
    if FAILS:
        print(f"SELFTEST FAIL ({len(FAILS)} test): {FAILS} [{dt:.1f}s]")
        sys.exit(1)
    print(f"SELFTEST OK [{dt:.1f}s]")


if __name__ == "__main__":
    main()
