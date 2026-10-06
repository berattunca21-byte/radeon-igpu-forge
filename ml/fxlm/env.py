"""fxlm.env: environment variables + GPU verification for LM training.

setup_lm_env() must be called BEFORE `import torch`. Difference from fxgpu.env:
TunableOp is OFF by default (development/safety: avoids the iGPU freezing
during a long tuning run); it is enabled with tunable=True and a CSV path
can be given via tune_file. This module does NOT import torch.
"""
import os


def require_gpu():
    """gfx1035 verification (the fxlm version of fxgpu.env.require_gpu).

    Difference from fxgpu's: it does not print the "~38% slower" WARNING when
    TunableOp is off — fxlm turns TunableOp off on purpose (setup_lm_env).
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
    return torch.device("cuda")


def setup_lm_env(tunable=False, tune_file=None):
    """Set up the environment variables (OVERRIDES existing ones).

    tunable=False: TunableOp off (safe development mode).
    tunable=True : TunableOp on; if tune_file is given
        PYTORCH_TUNABLEOP_FILENAME points to that file — the existing CSV
        is reused and only new shapes go through tuning
        (the TUNING=0 + CSV combination is broken — not used).
    """
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1" if tunable else "0"
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"  # crashes on gfx1035
    os.environ.setdefault("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS", "100")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    if tune_file is not None:
        os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(tune_file)
