"""fxlm.optim: FlatAdamW — fp32 master + fp16 param + dinamik loss scale.

Kurulumda parametreler tek flat fp32 master'a kopyalanir, m/v fp32 ve
flat fp16 param buffer'i ayrilir; her model parametresi flat16'nin bir
goruntusune isaret eder (p.data = flat16[off:off+n].view_as(p)) — ileri
hesap saf fp16, adim basina cast yok.

step(): adamw_sumsq (multi-tensor sumsq + nonfinite) -> adamw_final
(found_inf, norm, clip_coef, step++) -> adamw_update (flat AdamW, grad
pointer'lari by-value TensorList ile) -> torch._amp_update_scale_.
Hic host sync / H2D kopya yok.
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
    """fp32 master + fp16 param + dinamik loss scaling + global clip.

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
            raise ValueError("model'de requires_grad parametre yok")
        dev = params[0].device
        if not dev.type == "cuda":
            raise TypeError("parametreler cuda'da olmali")
        if any(p.dtype != torch.float32 for p in params):
            raise TypeError("FlatAdamW fp32 parametre bekler "
                            "(kendisi fp16 flat'e donusturur)")
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

    # -- yardimcilar ---------------------------------------------------------

    def _chunks(self):
        """grad'i olan parametrelerden TensorList chunk'lari uret."""
        items = []
        for name, p, off, n, wd in self._info:
            g = p.grad
            if g is None:
                continue
            if not g.is_contiguous():
                g = g.contiguous()
            if g.data_ptr() % 16:
                raise RuntimeError(f"{name}: grad 16B hizali degil")
            if g.dtype != torch.float16:
                raise TypeError(f"{name}: grad fp16 degil ({g.dtype})")
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
        self.flat16.copy_(self.master)     # fp16 goruntuleri tazele

    def master_state_dict(self):
        """fp32 master agirliklar, model.state_dict() anahtarlariyla."""
        return {name: self.master[off:off + n].view(p.shape)
                for name, p, off, n, _w in self._info}
