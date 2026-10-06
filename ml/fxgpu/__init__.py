"""fxgpu: fused-HIP-kernel ML helpers for gfx1035 (Radeon 680M).

Lazy exports: modules that depend on torch (fused_linear, model) are loaded on
first access — so `from fxgpu.env import setup_env` can set up the
environment variables without importing torch.
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
