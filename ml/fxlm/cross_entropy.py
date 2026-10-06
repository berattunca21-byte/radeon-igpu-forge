"""fxlm.cross_entropy: fused CE + linear-CE (the full (N,V) logits are not written).

The ce_fwd_grad kernel computes an online logsumexp (fp32) per row;
it writes the grad as (softmax - onehot) * mul in fp16 (mul = scale/n_valid,
a device scalar — no host sync). linear_cross_entropy produces
logits_c = x_c @ W.T over row chunks and gets dx/dW in the same loop.
"""
import ctypes
import math
import os

import torch

from fxgpu.hip_kernel import compile_module, launch

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kernels", "ce.hip")
_MODS = {}
_NAMES = ["ce_fwd_grad"]
_LDS_LIMIT = 60 * 1024     # upper bound for the row cache (V*2 <= this)


class _CEArgs(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.c_void_p), ("target", ctypes.c_void_p),
        ("grad", ctypes.c_void_p), ("loss", ctypes.c_void_p),
        ("mul", ctypes.c_void_p),
        ("N", ctypes.c_int), ("V", ctypes.c_int),
        ("ignore_index", ctypes.c_int64), ("use_lds", ctypes.c_int),
    ]


def _get(arch):
    if arch not in _MODS:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        _MODS[arch] = compile_module(src, _NAMES, arch=arch)
    return _MODS[arch]


def _inv_nvalid(target, ignore_index):
    """1/n_valid device tensor (n_valid==0 -> 0). No host sync."""
    nv = (target != ignore_index).sum(dtype=torch.float32)
    return torch.where(nv > 0, nv.reciprocal(), nv)


def _ce_rows(logits, target, grad, loss_sum, mul, ignore_index):
    """single launch: accumulate loss+grad over (C,V) logits."""
    N, V = logits.shape
    fns = _get(torch.cuda.get_device_properties(logits.device).gcnArchName)
    a = _CEArgs()
    a.logits = logits.data_ptr(); a.target = target.data_ptr()
    a.grad = grad.data_ptr() if grad is not None else 0
    a.loss = loss_sum.data_ptr(); a.mul = mul.data_ptr()
    a.N, a.V = N, V
    a.ignore_index = int(ignore_index)
    a.use_lds = 1 if V * 2 <= _LDS_LIMIT else 0
    launch(fns["ce_fwd_grad"], (N, 1, 1), (256, 1, 1), [a],
           shared=V * 2 if a.use_lds else 0)


def _check_logits(logits, target):
    if logits.dtype != torch.float16:
        raise TypeError(f"logits must be fp16, got {logits.dtype}")
    if not (logits.is_cuda and target.is_cuda):
        raise TypeError("logits and target must be on cuda")
    if not logits.is_contiguous() or not target.is_contiguous():
        raise ValueError("logits and target must be contiguous")
    if target.dtype != torch.int64:
        raise TypeError(f"target must be int64, got {target.dtype}")
    if logits.dim() != 2 or target.dim() != 1 \
            or target.shape[0] != logits.shape[0]:
        raise ValueError(f"expected logits (N,V), target (N,): "
                         f"{tuple(logits.shape)} vs {tuple(target.shape)}")


class _FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index):
        _check_logits(logits, target)
        N, V = logits.shape
        dev = logits.device
        inv_n = _inv_nvalid(target, ignore_index)
        loss_sum = torch.zeros((), device=dev, dtype=torch.float32)
        grad = torch.empty(N, V, device=dev, dtype=torch.float16)
        _ce_rows(logits, target, grad, loss_sum, inv_n, ignore_index)
        ctx.save_for_backward(grad)
        return loss_sum * inv_n

    @staticmethod
    def backward(ctx, g):
        (grad,) = ctx.saved_tensors
        return grad * g, None, None


def fused_cross_entropy(logits, target, ignore_index=-100):
    """logits (N,V) fp16, target (N,) int64 -> fp32 scalar loss
    (mean over the non-ignored rows)."""
    return _FusedCE.apply(logits, target, int(ignore_index))


def _auto_chunk(D, V):
    c = int(2.5e10 / (D * V) / 256) * 256
    return min(4096, max(256, c))


class _LinearCE(torch.autograd.Function):
    """loss = CE(x @ W.T, target) — the (N,V) logits are never materialized."""

    @staticmethod
    def forward(ctx, x, weight, target, grad_scale, ignore_index,
                chunk_rows):
        N, D, V = _check_lin(x, weight, target)
        dev = x.device
        if chunk_rows is None:
            chunk_rows = _auto_chunk(D, V)
        inv_n = _inv_nvalid(target, ignore_index)
        gs_t = (grad_scale if torch.is_tensor(grad_scale)
                else torch.tensor(float(grad_scale), device=dev,
                                  dtype=torch.float32))
        mul = inv_n * gs_t.float()
        loss_sum = torch.zeros((), device=dev, dtype=torch.float32)
        dx = torch.empty(N, D, device=dev, dtype=torch.float16)
        dW = torch.zeros(V, D, device=dev, dtype=torch.float32)
        dl_buf = torch.empty(min(chunk_rows, N), V, device=dev,
                             dtype=torch.float16)
        for r0 in range(0, N, chunk_rows):
            r1 = min(r0 + chunk_rows, N)
            xc = x[r0:r1]
            tc = target[r0:r1]
            logits_c = torch.mm(xc, weight.t())          # (C,V) fp16
            dl = dl_buf[:r1 - r0]
            _ce_rows(logits_c, tc, dl, loss_sum, mul, ignore_index)
            torch.mm(dl, weight, out=dx[r0:r1])          # dx_c (fp16)
            dW.add_(torch.mm(dl.t(), xc).float())        # fp32 accumulate
        ctx.save_for_backward(dx, dW)
        ctx.grad_scale = grad_scale
        ctx.wdtype = weight.dtype
        return loss_sum * inv_n

    @staticmethod
    def backward(ctx, g):
        dx, dW = ctx.saved_tensors
        gs = ctx.grad_scale
        inv = (torch.reciprocal(gs.float()) if torch.is_tensor(gs)
               else 1.0 / gs)
        gx = dx * (g * inv)
        gw = (dW * (g * inv)).to(ctx.wdtype)
        return gx, gw, None, None, None, None


def _check_lin(x, weight, target):
    if x.dtype != torch.float16 or weight.dtype != torch.float16:
        raise TypeError("x and weight must be fp16")
    if not (x.is_cuda and weight.is_cuda and target.is_cuda):
        raise TypeError("all tensors must be on cuda")
    if not (x.is_contiguous() and weight.is_contiguous()
            and target.is_contiguous()):
        raise ValueError("x, weight, target must be contiguous")
    if target.dtype != torch.int64:
        raise TypeError("target must be int64")
    N, D = x.shape
    V, Dw = weight.shape
    if Dw != D or target.shape != (N,):
        raise ValueError(f"shape mismatch: x{tuple(x.shape)} "
                         f"W{tuple(weight.shape)} t{tuple(target.shape)}")
    return N, D, V


def _linear_ce_loss_only(x, weight, target, ignore_index, chunk_rows):
    """no-grad path: dx/dW and dlogits are never allocated."""
    N, D, V = _check_lin(x, weight, target)
    dev = x.device
    if chunk_rows is None:
        chunk_rows = _auto_chunk(D, V)
    inv_n = _inv_nvalid(target, ignore_index)
    loss_sum = torch.zeros((), device=dev, dtype=torch.float32)
    for r0 in range(0, N, chunk_rows):
        r1 = min(r0 + chunk_rows, N)
        logits_c = torch.mm(x[r0:r1], weight.t())      # (C,V) fp16
        _ce_rows(logits_c, target[r0:r1], None, loss_sum, inv_n,
                 ignore_index)
    return loss_sum * inv_n


def linear_cross_entropy(x, weight, target, grad_scale=1.0,
                         ignore_index=-100, chunk_rows=None):
    """x (N,D) fp16, weight (V,D) fp16, target (N,) int64 -> loss fp32.

    grad_scale: python float or a 0-dim fp32 cuda tensor (loss scale).
    chunk_rows: None -> automatic (capped at ~50 GFLOP).
    If no grad is needed the loss-only path is used (dx/dW not allocated).
    """
    if not (torch.is_grad_enabled()
            and (x.requires_grad or weight.requires_grad)):
        return _linear_ce_loss_only(x, weight, target, ignore_index,
                                    chunk_rows)
    return _LinearCE.apply(x, weight, target, grad_scale,
                           int(ignore_index), chunk_rows)
