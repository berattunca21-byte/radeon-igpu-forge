"""fxlm.optim: FlatAdamW — fp32 master + fp16 param + dynamic loss scale.

At construction the parameters are copied into a single flat fp32 master, fp32
m/v and a flat fp16 param buffer are allocated; every model parameter points to
a view of flat16 (p.data = flat16[off:off+n].view_as(p)) — the forward
computation is pure fp16, with no per-step cast.

step(): adamw_sumsq (multi-tensor sumsq + nonfinite) -> adamw_final
(found_inf, norm, clip_coef, step++) -> adamw_update (flat AdamW, grad
pointers passed by value via a TensorList) -> torch._amp_update_scale_.
No host sync / H2D copy at all.
"""
import ctypes
import os

import torch

from fxgpu.hip_kernel import compile_module, launch

_SRC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "kernels", "optim.hip")
_MODS = {}
_NAMES = ["adamw_sumsq", "adamw_final", "adamw_update"]
_MAX_T = 64
_EPB = 2048    # optim.hip ELEMS_PER_BLOCK


class _TensorList(ctypes.Structure):
    _fields_ = [
        ("g", ctypes.c_void_p * _MAX_T),
        ("off", ctypes.c_int64 * _MAX_T),
        ("n", ctypes.c_int32 * _MAX_T),
        ("wd", ctypes.c_int32 * _MAX_T),
        ("count", ctypes.c_int32),
    ]


def _get(arch):
    if arch not in _MODS:
        with open(_SRC_PATH, "r", encoding="utf-8") as f:
            src = f.read()
        _MODS[arch] = compile_module(src, _NAMES, arch=arch)
    return _MODS[arch]


def _align8(n):
    return (n + 7) & ~7


class FlatAdamW:
    """fp32 master + fp16 param + dynamic loss scaling + global clip.

    opt = FlatAdamW(model, lr=3e-4, betas=(0.9,0.95), eps=1e-8,
                    weight_decay=0.1, max_grad_norm=1.0,
                    init_scale=2.**16, growth_interval=1000)
    opt.backward(loss)              # (loss * opt.scale).backward()
    stats = opt.step(lr=...)        # {grad_norm, found_inf, scale} 0-dim cuda
    opt.zero_grad()                 # set_to_none
    """

    def __init__(self, model, lr=3e-4, betas=(0.9, 0.95), eps=1e-8,
                 weight_decay=0.1, max_grad_norm=1.0, init_scale=2.0**16,
                 growth_interval=1000):
        params = [p for p in model.parameters() if p.requires_grad]
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        if not params:
            raise ValueError("the model has no requires_grad parameters")
        dev = params[0].device
        if not dev.type == "cuda":
            raise TypeError("parameters must be on cuda")
        if any(p.dtype != torch.float32 for p in params):
            raise TypeError("FlatAdamW expects fp32 parameters "
                            "(it converts them to the fp16 flat itself)")
        self.lr, self.betas, self.eps = lr, betas, eps
        self.weight_decay, self.max_grad_norm = weight_decay, max_grad_norm
        self.growth_interval = int(growth_interval)
        self._fns = _get(torch.cuda.get_device_properties(dev).gcnArchName)

        self._info = []     # (name, p, off, numel, wd_flag)
        total = 0
        for name, p in zip(names, params):
            self._info.append((name, p, total, p.numel(),
                               1 if (p.dim() >= 2 and weight_decay != 0)
                               else 0))
            total += _align8(p.numel())
        self.master = torch.empty(total, device=dev, dtype=torch.float32)
        self.m = torch.zeros(total, device=dev, dtype=torch.float32)
        self.v = torch.zeros(total, device=dev, dtype=torch.float32)
        self.flat16 = torch.empty(total, device=dev, dtype=torch.float16)
        for name, p, off, n, _wd in self._info:
            self.master[off:off + n] = p.data.reshape(-1)
        self.flat16.copy_(self.master)
        for name, p, off, n, _wd in self._info:
            p.data = self.flat16[off:off + n].view(p.shape)

        self.scratch = torch.zeros(2, device=dev, dtype=torch.float32)
        self._flags = torch.zeros((), device=dev, dtype=torch.int32)
        self._step_ctr = torch.zeros((), device=dev, dtype=torch.int32)
        self._found_inf = torch.zeros((), device=dev, dtype=torch.float32)
        self._norm = torch.zeros((), device=dev, dtype=torch.float32)
        self.scale = torch.full((), float(init_scale), device=dev,
                                dtype=torch.float32)
        self._growth = torch.zeros((), device=dev, dtype=torch.int32)

    # -- helpers -------------------------------------------------------------

    def _chunks(self):
        """Build TensorList chunks from the parameters that have a grad."""
        items = []
        for name, p, off, n, wd in self._info:
            g = p.grad
            if g is None:
                continue
            if not g.is_contiguous():
                g = g.contiguous()
            if g.data_ptr() % 16:
                raise RuntimeError(f"{name}: grad is not 16B aligned")
            if g.dtype != torch.float16:
                raise TypeError(f"{name}: grad is not fp16 ({g.dtype})")
            items.append((g, off, n, wd))
        for i in range(0, len(items), _MAX_T):
            tl = _TensorList()
            for j, (g, off, n, wd) in enumerate(items[i:i + _MAX_T]):
                tl.g[j] = g.data_ptr()
                tl.off[j] = off
                tl.n[j] = n
                tl.wd[j] = wd
            tl.count = min(len(items) - i, _MAX_T)
            yield tl, items[i:i + _MAX_T]

    # -- API ------------------------------------------------------------------

    def backward(self, loss):
        (loss * self.scale).backward()

    def step(self, lr=None):
        fns = self._fns
        lr_v = self.lr if lr is None else float(lr)
        for tl, items in self._chunks():
            maxn = max(n for _g, _o, n, _w in items)
            nblk = (maxn + _EPB - 1) // _EPB
            launch(fns["adamw_sumsq"], (nblk, tl.count, 1), (256, 1, 1),
                   [tl, ctypes.c_void_p(self.scratch.data_ptr()),
                    ctypes.c_void_p(self._flags.data_ptr())])
        launch(fns["adamw_final"], (1, 1, 1), (32, 1, 1),
               [ctypes.c_void_p(self.scratch.data_ptr()),
                ctypes.c_void_p(self._flags.data_ptr()),
                ctypes.c_void_p(self._step_ctr.data_ptr()),
                ctypes.c_void_p(self._found_inf.data_ptr()),
                ctypes.c_void_p(self._norm.data_ptr()),
                ctypes.c_void_p(self.scale.data_ptr()),
                ctypes.c_float(self.max_grad_norm)])
        for tl, items in self._chunks():
            maxn = max(n for _g, _o, n, _w in items)
            nblk = (maxn + _EPB - 1) // _EPB
            launch(fns["adamw_update"], (nblk, tl.count, 1), (256, 1, 1),
                   [tl, ctypes.c_void_p(self.master.data_ptr()),
                    ctypes.c_void_p(self.m.data_ptr()),
                    ctypes.c_void_p(self.v.data_ptr()),
                    ctypes.c_void_p(self.flat16.data_ptr()),
                    ctypes.c_void_p(self.scratch.data_ptr()),
                    ctypes.c_void_p(self._found_inf.data_ptr()),
                    ctypes.c_void_p(self.scale.data_ptr()),
                    ctypes.c_void_p(self._step_ctr.data_ptr()),
                    ctypes.c_float(lr_v),
                    ctypes.c_float(self.betas[0]),
                    ctypes.c_float(self.betas[1]),
                    ctypes.c_float(self.eps),
                    ctypes.c_float(self.weight_decay)])
        torch._amp_update_scale_(self.scale, self._growth, self._found_inf,
                                 2.0, 0.5, self.growth_interval)
        return {"grad_norm": self._norm, "found_inf": self._found_inf,
                "scale": self.scale}

    def zero_grad(self):
        for _n, p, _o, _el, _w in self._info:
            p.grad = None

    # -- state ----------------------------------------------------------------

    def state_dict(self):
        return {"master": self.master, "m": self.m, "v": self.v,
                "step": self._step_ctr, "scale": self.scale,
                "growth": self._growth,
                "offsets": [(n_, o, el) for n_, _p, o, el, _w in self._info]}

    def load_state_dict(self, sd):
        self.master.copy_(sd["master"]); self.m.copy_(sd["m"])
        self.v.copy_(sd["v"]); self._step_ctr.copy_(sd["step"])
        self.scale.copy_(sd["scale"]); self._growth.copy_(sd["growth"])
        self.flat16.copy_(self.master)     # refresh the fp16 views

    def master_state_dict(self):
        """fp32 master weights, keyed like model.state_dict()."""
        return {name: self.master[off:off + n].view(p.shape)
                for name, p, off, n, _w in self._info}
