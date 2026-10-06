"""fxgpu.model: WindowMLP — a 3-layer MLP for windowed input.

L1 = FusedGatherLinear (HIP gather_gemm): X16 (R,F) fp16 + idx32 (B,)
window start rows -> (B, hidden). Init scales are the same as bench7.
If master=True the parameters are kept in fp32 and computed in fp16 via .half()
in forward (autograd carries the grads back to the fp32 master).
"""
import torch
from torch import nn
import torch.nn.functional as Fn

from .fused_linear import FusedGatherLinear


class WindowMLP(nn.Module):
    def __init__(self, n_feat, window, hidden=256, hidden2=128,
                 master=True, dropout=0.0):
        super().__init__()
        self.n_feat = n_feat
        self.window = window
        self.master = master
        self.dropout = dropout
        dtype = torch.float32 if master else torch.float16
        K = window * n_feat
        self.W1 = nn.Parameter(torch.randn(hidden, K, dtype=dtype) * 0.03)
        self.b1 = nn.Parameter(torch.zeros(hidden, dtype=dtype))
        self.W2 = nn.Parameter(torch.randn(hidden2, hidden, dtype=dtype) * 0.06)
        self.b2 = nn.Parameter(torch.zeros(hidden2, dtype=dtype))
        self.W3 = nn.Parameter(torch.randn(1, hidden2, dtype=dtype) * 0.1)
        self.b3 = nn.Parameter(torch.zeros(1, dtype=dtype))

    def _half_params(self):
        ps = (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3)
        return [p.half() for p in ps] if self.master else list(ps)

    def _trunk(self, h1, W2, b2, W3, b3):
        h = Fn.silu(h1)
        if self.dropout > 0 and self.training:
            h = Fn.dropout(h, p=self.dropout)
        h = Fn.silu(Fn.linear(h, W2, b2))
        return Fn.linear(h, W3, b3)                    # (B, 1) fp16

    def forward(self, X16, idx32):
        """X16 (R,F) fp16 + idx32 (B,) -> (B,1) fp16 logit."""
        W1, b1, W2, b2, W3, b3 = self._half_params()
        h1 = FusedGatherLinear.apply(X16, idx32, W1, b1)
        return self._trunk(h1, W2, b2, W3, b3)

    def forward_dense(self, xb16):
        """Same computation with an already-gathered (B,K) fp16 batch.

        For eval parity / inference without HIP (plain Fn.linear instead of
        FusedGatherLinear).
        """
        W1, b1, W2, b2, W3, b3 = self._half_params()
        h1 = Fn.linear(xb16, W1, b1)
        return self._trunk(h1, W2, b2, W3, b3)
