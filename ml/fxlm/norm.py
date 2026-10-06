"""fxlm.norm: fused RMSNorm (fp16 io, fp32 math).

y = x * rsqrt(mean(x^2) + eps) * w — ileri kernel satir-cache'i LDS'te
tutar; geri dx + dweight (fp32 partial + sum). Hedef: 8192x768 fwd+bwd
~2 ms (F.rms_norm 4.7 ms).
"""
import ctypes
import os

import torch

from fxgpu.hip_kernel import compile_module, launch

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kernels", "norm.hip")
_MODS = {}   # arch -> {isim: func}
_NAMES = ["rms_norm_fwd", "rms_norm_bwd_dx", "rms_norm_bwd_dw"]
_RPB = 64    # norm.hip'teki RPB ile ayni
_MAX_LDS = 60 * 1024


def _get(arch):
    if arch not in _MODS:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        _MODS[arch] = compile_module(src, _NAMES, arch=arch)
    return _MODS[arch]


def _check(x, w):
    if x.dtype != torch.float16 or w.dtype != torch.float16:
        raise TypeError(f"x ve weight fp16 olmali ({x.dtype}, {w.dtype})")
    if not (x.is_cuda and w.is_cuda):
        raise TypeError("x ve weight cuda'da olmali")
    D = x.shape[-1]
    if w.shape != (D,):
        raise ValueError(f"weight ({D},) olmali, {tuple(w.shape)}")
    if D % 8 != 0 or D > 8192:
        raise ValueError(f"D={D} icin D%8==0 ve D<=8192 sart")
    if not x.is_contiguous() or not w.is_contiguous():
        raise ValueError("x ve weight contiguous olmali")
    return D


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        D = _check(x, w)
        N = x.numel() // D
        x2 = x.view(N, D)
        y = torch.empty_like(x2)
        rstd = torch.empty(N, device=x.device, dtype=torch.float32)
        fns = _get(torch.cuda.get_device_properties(x.device).gcnArchName)
        launch(fns["rms_norm_fwd"], (N, 1, 1), (256, 1, 1),
               [ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(w.data_ptr()),
                ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(rstd.data_ptr()),
                ctypes.c_int(D), ctypes.c_float(eps)],
               shared=D * 2)
        ctx.save_for_backward(x2, w, rstd)
        ctx.eps = eps
        ctx.oshape = x.shape
        return y.view(x.shape)

    @staticmethod
    def backward(ctx, dy):
        x2, w, rstd = ctx.saved_tensors
        N, D = x2.shape
        dy2 = dy.contiguous().view(N, D)
        dx = torch.empty_like(x2)
        fns = _get(torch.cuda.get_device_properties(x2.device).gcnArchName)
        launch(fns["rms_norm_bwd_dx"], (N, 1, 1), (256, 1, 1),
               [ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(dy2.data_ptr()),
                ctypes.c_void_p(w.data_ptr()), ctypes.c_void_p(rstd.data_ptr()),
                ctypes.c_void_p(dx.data_ptr()), ctypes.c_int(D)],
               shared=2 * D * 2)
        nstrips = (N + _RPB - 1) // _RPB
        part = torch.empty(nstrips, D, device=x2.device,
                           dtype=torch.float32)
        launch(fns["rms_norm_bwd_dw"], (nstrips, 1, 1), (256, 1, 1),
               [ctypes.c_void_p(x2.data_ptr()), ctypes.c_void_p(dy2.data_ptr()),
                ctypes.c_void_p(rstd.data_ptr()),
                ctypes.c_void_p(part.data_ptr()),
                ctypes.c_int(N), ctypes.c_int(D), ctypes.c_int(N)])
        dw = part.sum(0).to(w.dtype)
        return dx.view(ctx.oshape), dw, None


def rms_norm(x, weight, eps=1e-6):
    """x (...,D) fp16, weight (D,) fp16 -> y fp16."""
    return _RMSNorm.apply(x, weight, float(eps))
