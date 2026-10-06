"""fxgpu: gfx1035 (Radeon 680M) icin fused-HIP-kernel ML yardimcilari.

Tembel export'lar: torch'a bagli moduller (fused_linear, model) ilk
erisimde yuklenir — boylece `from fxgpu.env import setup_env` torch'u
import etmeden ortam degiskenlerini kurabilir.
"""

__all__ = [
    "FusedGatherLinear",
    "fused_gather_linear",
    "WindowMLP",
    "setup_env",
    "require_gpu",
]


def __getattr__(name):
    if name in ("FusedGatherLinear", "fused_gather_linear"):
        from . import fused_linear as _m
        return getattr(_m, name)
    if name == "WindowMLP":
        from .model import WindowMLP
        return WindowMLP
    if name in ("setup_env", "require_gpu"):
        from . import env as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
