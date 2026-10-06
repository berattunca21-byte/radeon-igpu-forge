"""fxgpu.env: ortam degiskenleri + GPU dogrulamasi.

setup_env() `import torch`'dan ONCE cagrilmali (TunableOp ayarlari torch
baslatilmadan once ortamda olmali). Bu modul torch IMPORT ETMEZ.
"""
import os

TUNABLEOP_DEFAULTS = {
    "PYTORCH_TUNABLEOP_ENABLED": "1",
    "PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED": "0",   # Lt yolu gfx1035'te patlar
    "PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS": "200",
    "HIP_VISIBLE_DEVICES": "0",
}


def setup_env():
    """Sampiyon ortam degiskenlerini setdefault ile kur (var olana dokunmaz)."""
    for k, v in TUNABLEOP_DEFAULTS.items():
        os.environ.setdefault(k, v)


def require_gpu():
    """gfx1035 + TunableOp dogrulamasi; torch'u tembel import eder.

    GPU yoksa ya da mimari gfx1035 degilse SystemExit. Basariliysa bir
    satir torch/hip/arch bilgisi basar ve torch.device("cuda") dondurur.
    """
    import torch

    if not torch.cuda.is_available():
        raise SystemExit(
            "GPU bulunamadi: torch.cuda.is_available() False. "
            "ROCm ortamini kontrol et (dogru venv + suruculer).")
    props = torch.cuda.get_device_properties(0)
    arch = str(getattr(props, "gcnArchName", "?"))
    if not arch.startswith("gfx1035"):
        raise SystemExit(
            f"Beklenen gfx1035 yerine {arch} goruldu ({props.name}). "
            "Bu script/kernel'lar gfx1035 icin dogrulanmistir.")
    print(f"torch {torch.__version__} | HIP {torch.version.hip} | "
          f"{props.name} ({arch})")
    if not torch.cuda.tunable.is_enabled():
        print("UYARI: TunableOp kapali — setup_env() torch'tan once "
              "cagrilmamis olabilir; fp16 GEMM ~%38 yavas calisir.")
    return torch.device("cuda")
