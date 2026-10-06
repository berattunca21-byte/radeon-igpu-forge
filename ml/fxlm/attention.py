"""fxlm.attention: flash-attention (FA2 tarzi) HIP kernel sarilari.

q,k,v (B,T,H,Dh) fp16, son eksen stride=1, diger stride'lar serbest
(fused qkv buffer'in gorunumu kopyasiz calisir). o (B,T,H,Dh) fp16
contiguous doner; lse (B,H,T) fp32 backward icin saklanir.
Dh in {64,128}; T >= 1 serbest (kuyruk bloklari maskelenir).
"""
import ctypes
import math
import os

import torch

from fxgpu.hip_kernel import compile_module, launch

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kernels", "attention.hip")
_MODS = {}   # (arch, Dh) -> {isim: func}
_NAMES = ["attn_fwd", "attn_bwd_prep", "attn_bwd_dkv", "attn_bwd_dq"]


class _AttnArgs(ctypes.Structure):
    _fields_ = [
        ("q", ctypes.c_void_p), ("k", ctypes.c_void_p), ("v", ctypes.c_void_p),
        ("o", ctypes.c_void_p), ("dout", ctypes.c_void_p),
        ("o_out", ctypes.c_void_p),
        ("dq", ctypes.c_void_p), ("dk", ctypes.c_void_p), ("dv", ctypes.c_void_p),
        ("lse", ctypes.c_void_p), ("lse_out", ctypes.c_void_p),
        ("delta", ctypes.c_void_p), ("delta_out", ctypes.c_void_p),
        ("qsb", ctypes.c_int64), ("qst", ctypes.c_int64), ("qsh", ctypes.c_int64),
        ("ksb", ctypes.c_int64), ("kst", ctypes.c_int64), ("ksh", ctypes.c_int64),
        ("vsb", ctypes.c_int64), ("vst", ctypes.c_int64), ("vsh", ctypes.c_int64),
        ("dqb", ctypes.c_int64), ("dqt", ctypes.c_int64), ("dqh", ctypes.c_int64),
        ("dkb", ctypes.c_int64), ("dkt", ctypes.c_int64), ("dkh", ctypes.c_int64),
        ("dvb", ctypes.c_int64), ("dvt", ctypes.c_int64), ("dvh", ctypes.c_int64),
        ("B", ctypes.c_int), ("T", ctypes.c_int), ("H", ctypes.c_int),
        ("scale", ctypes.c_float), ("causal", ctypes.c_int),
    ]


def _get(arch, dh):
    key = (arch, dh)
    if key not in _MODS:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        br = bc = 64 if dh == 64 else 32
        _MODS[key] = compile_module(
            src, _NAMES, arch=arch,
            extra_opts=[f"-DHEAD_DIM={dh}", f"-DBR={br}", f"-DBC={bc}"])
    return _MODS[key]


def _br_bc(dh):
    return (64, 64) if dh == 64 else (32, 32)


def _check_qkv(q, k, v):
    if q.dim() != 4 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"q,k,v ayni 4B sekilde olmali: {tuple(q.shape)} "
            f"{tuple(k.shape)} {tuple(v.shape)}")
    B, T, H, Dh = q.shape
    if Dh not in (64, 128):
        raise ValueError(f"head_dim {Dh} desteklenmiyor (64 veya 128 olmali)")
    if T < 1 or B < 1 or H < 1:
        raise ValueError(f"gecersiz sekil: {tuple(q.shape)}")
    for nm, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.float16:
            raise TypeError(f"{nm} fp16 olmali, {t.dtype} verildi")
        if not t.is_cuda:
            raise TypeError(f"{nm} cuda tensoru olmali")
        if t.stride(-1) != 1:
            raise ValueError(f"{nm} son ekseni contiguous olmali "
                             f"(stride(-1)={t.stride(-1)})")
    if q.numel() >= 2**31:
        raise ValueError("tensor cok buyuk (2^31 eleman siniri)")
    if B * H > 65535:
        raise ValueError(f"B*H={B * H} > 65535 (grid.y siniri)")
    return B, T, H, Dh


def _strides3(t):
    s = t.stride()
    return s[0], s[1], s[2]


def _fwd(q, k, v, causal, scale):
    B, T, H, Dh = _check_qkv(q, k, v)
    dev = q.device
    o = torch.empty(B, T, H, Dh, device=dev, dtype=torch.float16)
    lse = torch.empty(B, H, T, device=dev, dtype=torch.float32)
    arch = torch.cuda.get_device_properties(dev).gcnArchName
    fns = _get(arch, Dh)
    BR, _ = _br_bc(Dh)
    a = _AttnArgs()
    a.q = q.data_ptr(); a.k = k.data_ptr(); a.v = v.data_ptr()
    a.o_out = o.data_ptr(); a.lse_out = lse.data_ptr()
    a.qsb, a.qst, a.qsh = _strides3(q)
    a.ksb, a.kst, a.ksh = _strides3(k)
    a.vsb, a.vst, a.vsh = _strides3(v)
    a.B, a.T, a.H = B, T, H
    a.scale = scale; a.causal = int(causal)
    launch(fns["attn_fwd"], ((T + BR - 1) // BR, B * H, 1), (256, 1, 1), [a])
    return o, lse


def _bwd(q, k, v, o, lse, do, causal, scale, dq, dk, dv):
    B, T, H, Dh = q.shape
    do = do.contiguous()
    dev = q.device
    delta = torch.empty(B, H, T, device=dev, dtype=torch.float32)
    arch = torch.cuda.get_device_properties(dev).gcnArchName
    fns = _get(arch, Dh)
    BR, BC = _br_bc(Dh)
    a = _AttnArgs()
    a.q = q.data_ptr(); a.k = k.data_ptr(); a.v = v.data_ptr()
    a.o = o.data_ptr(); a.dout = do.data_ptr()
    a.lse = lse.data_ptr(); a.delta_out = delta.data_ptr()
    a.qsb, a.qst, a.qsh = _strides3(q)
    a.ksb, a.kst, a.ksh = _strides3(k)
    a.vsb, a.vst, a.vsh = _strides3(v)
    a.B, a.T, a.H = B, T, H
    a.scale = scale; a.causal = int(causal)
    launch(fns["attn_bwd_prep"], (B * H, 1, 1), (256, 1, 1), [a])
    a.delta = delta.data_ptr()
    a.dq = dq.data_ptr(); a.dk = dk.data_ptr(); a.dv = dv.data_ptr()
    a.dqb, a.dqt, a.dqh = _strides3(dq)
    a.dkb, a.dkt, a.dkh = _strides3(dk)
    a.dvb, a.dvt, a.dvh = _strides3(dv)
    nkt = (T + BC - 1) // BC
    nqt = (T + BR - 1) // BR
    launch(fns["attn_bwd_dkv"], (nkt, B * H, 1), (256, 1, 1), [a])
    launch(fns["attn_bwd_dq"], (nqt, B * H, 1), (256, 1, 1), [a])
    return dq, dk, dv


class _FlashAttn(torch.autograd.Function):
    """flash_attention(q,k,v): o (B,T,H,Dh) contiguous fp16."""

    @staticmethod
    def forward(ctx, q, k, v, causal, scale):
        o, lse = _fwd(q, k, v, causal, scale)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        B, T, H, Dh = q.shape
        dq = torch.empty(B, T, H, Dh, device=q.device, dtype=torch.float16)
        dk = torch.empty(B, T, H, Dh, device=q.device, dtype=torch.float16)
        dv = torch.empty(B, T, H, Dh, device=q.device, dtype=torch.float16)
        _bwd(q, k, v, o, lse, do, ctx.causal, ctx.scale, dq, dk, dv)
        return dq, dk, dv, None, None


class _FlashAttnQKV(torch.autograd.Function):
    """flash_attention_qkv(qkv): qkv (B,T,3,H,Dh) fp16 contiguous."""

    @staticmethod
    def forward(ctx, qkv, causal, scale):
        if qkv.dim() != 5 or qkv.shape[2] != 3:
            raise ValueError(f"qkv (B,T,3,H,Dh) olmali, {tuple(qkv.shape)}")
        if not qkv.is_contiguous():
            raise ValueError("qkv contiguous olmali")
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        o, lse = _fwd(q, k, v, causal, scale)
        ctx.save_for_backward(qkv, o, lse)
        ctx.causal, ctx.scale = causal, scale
        return o

    @staticmethod
    def backward(ctx, do):
        qkv, o, lse = ctx.saved_tensors
        B, T = qkv.shape[0], qkv.shape[1]
        dqkv = torch.empty_like(qkv)
        dq, dk, dv = dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2]
        _bwd(qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2], o, lse, do,
             ctx.causal, ctx.scale, dq, dk, dv)
        return dqkv, None, None


def flash_attention(q, k, v, causal=True, scale=None):
    """q,k,v (B,T,H,Dh) fp16 -> o (B,T,H,Dh) fp16."""
    Dh = q.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(Dh)
    return _FlashAttn.apply(q, k, v, causal, float(scale))


def flash_attention_qkv(qkv, causal=True, scale=None):
    """qkv (B,T,3,H,Dh) fp16 contiguous -> o (B,T,H,Dh) fp16."""
    if qkv.dim() != 5:
        raise ValueError(f"qkv (B,T,3,H,Dh) olmali, {tuple(qkv.shape)}")
    Dh = qkv.shape[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(Dh)
    return _FlashAttnQKV.apply(qkv, causal, float(scale))
