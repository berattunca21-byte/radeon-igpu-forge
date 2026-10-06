"""fxlm: small GPT training package for gfx1035 (Radeon 680M).

Hand-written HIP kernels (compiled at runtime with hiprtc):
  - flash attention (fwd+bwd, fp16 io / fp32 softmax)
  - fused cross-entropy + linear-CE (the full (N,V) logits are never materialized)
  - fused RMSNorm
  - FlatAdamW (fp32 master + fp16 param + dynamic loss scale, fused)

Lazy exports: `from fxlm.env import setup_lm_env` sets up the environment
variables without importing torch.
"""

__all__ = [
    "setup_lm_env",
    "require_gpu",
    "flash_attention",
    "flash_attention_qkv",
    "fused_cross_entropy",
    "linear_cross_entropy",
    "rms_norm",
    "FlatAdamW",
    "GPT",
    "GPTConfig",
]


def __getattr__(name):
    if name in ("setup_lm_env", "require_gpu"):
        from . import env as _m
        return getattr(_m, name)
    if name in ("flash_attention", "flash_attention_qkv"):
        from . import attention as _m
        return getattr(_m, name)
    if name in ("fused_cross_entropy", "linear_cross_entropy"):
        from . import cross_entropy as _m
        return getattr(_m, name)
    if name == "rms_norm":
        from .norm import rms_norm
        return rms_norm
    if name == "FlatAdamW":
        from .optim import FlatAdamW
        return FlatAdamW
    if name in ("GPT", "GPTConfig"):
        from . import model as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
