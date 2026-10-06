"""fxlm.env: LM egitimi icin ortam degiskenleri + GPU dogrulamasi.

setup_lm_env() `import torch`'dan ONCE cagrilmali. fxgpu.env'den farki:
TunableOp varsayilan KAPALI (gelistirme/guvenlik: iGPU'da uzun tuning
kilitlenmesi onlenir); tunable=True ile acilir ve tune_file ile CSV
yolu verilebilir. Bu modul torch IMPORT ETMEZ.
"""
import os


def require_gpu():
    """gfx1035 dogrulamasi (fxgpu.env.require_gpu'nun fxlm surumu).

    fxgpu'nunkinden farki: TunableOp kapaliyken "~%38 yavas" UYARISI
    basmaz — fxlm TunableOp'u bilerek kapatir (setup_lm_env).
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
    return torch.device("cuda")


def setup_lm_env(tunable=False, tune_file=None):
    """Ortam degiskenlerini kur (var olanlari EZER — override).

    tunable=False: TunableOp kapali (guvenli gelistirme modu).
    tunable=True : TunableOp acik; tune_file verilirse
        PYTORCH_TUNABLEOP_FILENAME o dosyaya isaret eder — mevcut CSV
        yeniden kullanilir, sadece yeni sekiller tuning'e girer
        (TUNING=0 + CSV kombinasyonu bozuk — kullanilmaz).
    """
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1" if tunable else "0"
    os.environ["PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED"] = "0"  # gfx1035'te coker
    os.environ.setdefault("PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS", "100")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "0")
    if tune_file is not None:
        os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(tune_file)
