"""fxgpu self-test: FusedGatherLinear parity vs the unfold+Fn.linear reference.

Run (from the repo root):
    GPU_PYTHON.cmd ml\\fxgpu\\selftest.py
or inside ml\\fxgpu with the champion env:  python selftest.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxgpu.env import setup_env  # noqa: E402  (before torch)
setup_env()

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from fxgpu import FusedGatherLinear, fused_gather_linear  # noqa: E402


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")
    R, F, W, HID = 100_000, 40, 30, 256
    K = W * F
    B = 8192

    X16 = torch.randn(R, F, device=device, dtype=torch.float16)
    weight = torch.randn(HID, K, device=device, dtype=torch.float16) * 0.03
    bias = torch.randn(HID, device=device, dtype=torch.float16) * 0.01
    idx32 = torch.randint(0, R - W + 1, (B,), device=device,
                          dtype=torch.int32)

    # reference: unfold view + Fn.linear (flattened window-major)
    Xu = X16.unfold(0, W, 1)                          # (R-W+1, F, W)
    xb = Xu[idx32.long()].transpose(1, 2).reshape(B, K)
    w_ref = weight.detach().clone().requires_grad_()
    b_ref = bias.detach().clone().requires_grad_()
    out_ref = Fn.linear(xb, w_ref, b_ref)

    w_f = weight.detach().clone().requires_grad_()
    b_f = bias.detach().clone().requires_grad_()
    out_f = fused_gather_linear(X16, idx32, w_f, b_f)

    fwd_err = (out_f.double() - out_ref.double()).abs().max().item()
    print(f"fwd max abs err = {fwd_err:.6f}  (threshold 0.05)")
    assert fwd_err < 0.05, "forward parity broken"

    g = torch.randn(B, HID, device=device, dtype=torch.float16)
    out_ref.backward(g)
    out_f.backward(g)
    dW_err = (w_f.grad.double() - w_ref.grad.double()).abs().max().item()
    db_err = (b_f.grad.double() - b_ref.grad.double()).abs().max().item()
    print(f"dW err = {dW_err:.6f} | db err = {db_err:.6f}  (threshold 1e-3)")
    assert dW_err < 1e-3 and db_err < 1e-3, "grad parity broken"

    # an F % 8 != 0 error must be triggered
    try:
        bad = torch.randn(1000, 37, device=device, dtype=torch.float16)
        w37 = torch.randn(HID, 37 * W, device=device,
                          dtype=torch.float16)
        FusedGatherLinear.apply(bad, idx32[:8].clamp(max=1000 - W),
                                w37, bias)
    except ValueError as e:
        assert "multiple of 8" in str(e)
        print(f"F=37 rejected OK: {e}")
    else:
        raise AssertionError("no ValueError was raised for F=37")

    # forward_dense parity (WindowMLP)
    from fxgpu import WindowMLP
    torch.manual_seed(1)
    for master in (True, False):
        m = WindowMLP(F, W, master=master).to(device)
        o1 = m(X16, idx32)
        o2 = m.forward_dense(xb)
        d = (o1.double() - o2.double()).abs().max().item()
        print(f"WindowMLP master={master}: fused vs dense err = {d:.6f}")
        assert d < 0.05

    # inference_mode: null-Xg path — fused vs forward_dense parity + speed
    torch.manual_seed(2)
    m = WindowMLP(F, W, master=True).to(device).eval()
    big = torch.randint(0, R - W + 1, (32768,), device=device,
                        dtype=torch.int32)

    def _time(fn, iters=20, warm=5):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000

    ms_grad = _time(lambda: m(X16, big))          # path with the Xg write-out
    with torch.inference_mode():
        o_inf = m(X16, big)
        xb_big = Xu[big.long()].transpose(1, 2).reshape(32768, K)
        o_dense = m.forward_dense(xb_big)
        d_inf = (o_inf.double() - o_dense.double()).abs().max().item()
        ms_inf = _time(lambda: m(X16, big))       # Xg=NULL path
    print(f"inference_mode fused vs dense err = {d_inf:.6f}")
    assert d_inf < 0.05, "inference_mode (null-Xg) parity broken"
    print(f"eval fwd B=32768: grad-on {ms_grad:.3f} ms | "
          f"inference_mode {ms_inf:.3f} ms")

    print("SELFTEST OK")


if __name__ == "__main__":
    main()
