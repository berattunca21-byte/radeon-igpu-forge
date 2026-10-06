"""fxgpu.env: environment variables + GPU verification.

setup_env() must be called BEFORE `import torch` (the TunableOp settings must be
in the environment before torch initializes). This module does NOT import torch.
"""
import os

TUNABLEOP_DEFAULTS = {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",   # the Lt path crashes on gfx1035
    "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": "200",
    "HIP_VISIBLE_DEVICES": "0",
}


def setup_env():
    """Set the champion environment variables via setdefault (existing values are left alone)."""
    for k, v in TUNABLEOP_DEFAULTS.items():
        os.environ.setdefault(k, v)


def require_gpu():
    """Verify gfx1035 + TunableOp; imports torch lazily.

    Raises SystemExit if there is no GPU or the architecture is not gfx1035. On
    success it prints one line of torch/hip/arch info and returns torch.device("cuda").
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "No GPU found: torch.cuda.is_available() is False. "
            "Check the ROCm environment (right venv + drivers).")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "?"))
    if not arch.startswith("gfx1035"):
        raise SystemExit(
            f"Expected gfx1035 but saw {arch} ({props.name}). "
            "These scripts/kernels are verified for gfx1035.")
    print(f"torch {torch.__version__} | HIP {torch.version.hip} | "
          f"{props.name} ({arch})")
    if not torch.cuda.tunable.is_enabled():
        print("WARNING: TunableOp is off — setup_env() may not have been called "
              "before torch; fp16 GEMM runs ~38% slower.")
    return torch.device("cuda")
