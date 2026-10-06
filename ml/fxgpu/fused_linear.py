"""fxgpu.fused_linear: FusedGatherLinear — fused gather+GEMM autograd katmani.

out = (X[idx].unfold penceresi, window-major duz) @ W.T + bias — ileri
yonde tek HIP kernel (gather_gemm.hip), geri yonde rocBLAS mm ile dW.
bench7_fused_e2e.py'deki FusedGatherLinear'in genellestirilmis hali.
"""
import ctypes
import os

import torch

from .hip_kernel import compile_kernel, launch

_KERNELS = {}          # arch -> (module, func); proses basina bir derleme
_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "gather_gemm.hip")


def _get_kernel(arch):
    if arch not in _KERNELS:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        _KERNELS[arch] = compile_kernel(src, "gather_gemm", arch=arch)
    return _KERNELS[arch][1]


class FusedGatherLinear(torch.autograd.Function):
    """out[b, n] = sum_k Xg[b, k] * weight[n, k] + bias[n]

    Xg[b, w*F + f] = X[idx[b] + w, f] — pencere-major (window-major) duz.
    X16: (R, F) fp16 contiguous; idx32: (B,) int32 contiguous;
    weight: (N, K) fp16 contiguous, K = W*F; bias: (N,) fp16.
    """

    @staticmethod
    def forward(ctx, X16, idx32, weight, bias):
        F = X16.shape[1]
        K = weight.shape[1]
        if K % F != 0:
            raise ValueError(
                f"weight.shape[1]={K} X16.shape[1]={F}'in kati degil "
                f"(K = window * F olmali)")
        if F % 8 != 0:
            raise ValueError(
                "F'yi 8'in katina sifir-sutunla doldur "
                f"(simdiki F={F}; kernel 16B-hizali float4 yukleri ister)")
        if X16.dtype != torch.float16:
            raise TypeError(f"X16 fp16 olmali, {X16.dtype} verildi")
        if idx32.dtype != torch.int32:
            raise TypeError(f"idx32 int32 olmali, {idx32.dtype} verildi")
        if weight.dtype != torch.float16 or bias.dtype != torch.float16:
            raise TypeError("weight ve bias fp16 olmali")
        if not (X16.is_cuda and idx32.is_cuda and weight.is_cuda
                and bias.is_cuda):
            raise TypeError("tum tensorlar GPU'da (cuda) olmali")
        if not (X16.is_contiguous() and idx32.is_contiguous()
                and weight.is_contiguous() and bias.is_contiguous()):
            raise ValueError("tum tensorlar contiguous olmali "
                             "(.contiguous() cagir)")
        if idx32.dim() != 1:
            raise ValueError(f"idx32 (B,) olmali, shape={tuple(idx32.shape)}")
        if weight.dim() != 2 or bias.dim() != 1 \
                or bias.shape[0] != weight.shape[0]:
            raise ValueError("weight (N,K), bias (N,) olmali")

        Bsz = idx32.numel()
        N_out = weight.shape[0]
        C = torch.empty(Bsz, N_out, device=X16.device, dtype=torch.float16)
        need_grad = any(ctx.needs_input_grad)
        # Xg sadece backward (dW) icin lazim — inference'da ayirma,
        # kernel Xg==NULL dalinda write-out'u atliyor.
        Xg = (torch.empty(Bsz, K, device=X16.device, dtype=torch.float16)
              if need_grad else None)
        arch = torch.cuda.get_device_properties(X16.device).gcnArchName
        fn = _get_kernel(arch)
        grid = ((N_out + 127) // 128, (Bsz + 127) // 128, 1)
        args = [ctypes.c_void_p(X16.data_ptr()),
                ctypes.c_void_p(idx32.data_ptr()),
                ctypes.c_void_p(weight.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(C.data_ptr()),
                ctypes.c_void_p(Xg.data_ptr() if need_grad else 0),
                ctypes.c_int(Bsz), ctypes.c_int(N_out),
                ctypes.c_int(K), ctypes.c_int(F)]
        launch(fn, grid, (256, 1, 1), args)
        if need_grad:
            ctx.save_for_backward(Xg)
        return C

    @staticmethod
    def backward(ctx, g):
        (Xg,) = ctx.saved_tensors
        g = g.contiguous()
        dW = torch.mm(g.t(), Xg)          # (N, K)
        db = g.sum(0)
        return None, None, dW, db


def fused_gather_linear(X16, idx32, weight, bias):
    """FusedGatherLinear.apply icin ince sarmalayici."""
    return FusedGatherLinear.apply(X16, idx32, weight, bias)
