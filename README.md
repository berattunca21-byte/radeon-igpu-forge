# Radeon Forge — Radeon 680M (gfx1035) iGPU'da Maksimum Hızlı Yapay Zeka Eğitimi

> ROCm/HIP + PyTorch ile entegre AMD GPU'da ölçülmüş hız reçeteleri, özel HIP kernel'leri (fused linear, flash attention, fused CE, FlatAdamW) ve çalışan kurulum kaydı.

Son doğrulama: **27 Eylül 2026** (RAM çift kanala geçirildi, Sec.2 güncellendi; fxlm dil modeli paketi 24 Eyl 2026'da eklendi ve ölçüldü; tablo rakamları 20 Eylül 2026'da yeniden ölçüldü)
Kapsam: bu makinede **PyTorch eğitimi** — (1) tablo / zaman-serisi pencere modelleri (`ml\fxgpu\`), (2) küçük dil modeli / transformer (`ml\fxlm\`, Sec.12). Oyun/görüntü CNN proje notları bu bilgi bankasından çıkarıldı (yerel arşiv, repoda yok); gfx1035'e genel CNN dersleri Sec.9'da.

> Tek cümle: Bu laptopta Radeon 680M üzerinde ROCm/HIP PyTorch eğitimi **çalışıyor**; ölçülmüş en
> hızlı yol saf fp16 + TunableOp(rocBLAS) + el yazısı HIP fused L1 katmanı = **~1.40M örnek/sn**
> (FP32 eager'ın 3.1 katı). Bunun hazır kodu `ml\fxgpu\` paketinde, çalıştırıcısı `GPU_PYTHON.cmd`.
> Dil modeli / transformer için `ml\fxlm\` (el yazısı flash attention + fused CE + FlatAdamW)
> naif PyTorch GPT'ye göre adım başına **1.9–2.7×** hızlı, aynı öğrenme kalitesiyle (Sec.12).

---

## 1. Zorunlu kurallar (insan veya AI için)

1. **Doğru yorumlayıcı:** `<PROJE>\venv-rocm10\Scripts\python.exe`
   Sistem `python` komutu **GPU'suzdur**; eğitimde asla kullanılmaz. En kolayı: KB kökündeki
   `GPU_PYTHON.cmd script.py ...` — doğru python'u **ve** zorunlu TunableOp env var'larını birlikte verir.
2. `torch.cuda.*` adı Radeon'da normaldir; bu kurulum CUDA değil **ROCm/HIP**'tir. `nvidia-smi` yoktur.
3. `torch.cuda.is_available()` False ise CPU'ya sessizce geçme — ortamı düzelt (Sec.9).
4. Çalışan venv'i (`venv-rocm10`) **silme, yükseltme, içine pip install yapma.** Deney için ayrı venv kur.
5. Radeon 680M = **`gfx1035`**. Başka GFX paketi kurma; `HSA_OVERRIDE_GFX_VERSION` ayarlama.
6. Eğitimde **hız reçetesine uy** (Sec.5): DataLoader yok, Conv1d yok, batch 16384–32768, saf fp16,
   `loss.item()` batch içinde yok, `isfinite` kontrolü var. (tablo reçetesi; dil modeli kuralları Sec.12)
7. `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` **şart** — hipBLASLt yolu gfx1035'te kernel dosyası
   eksikliğiyle çöker (Sec.5.3).
8. Her yeni eğitim ayrı çıktı dosyasına yazılır; kullanıcı istemeden eski model/checkpoint silinmez.
9. Aynı anda **tek** GPU işi. iGPU sistem RAM'ini paylaşır; ikinci süreç hem hızı hem ölçümü bozar.
10. NaN/Inf görülürse eğitim durur, son iyi checkpoint korunur, lr düşürülür / fp32 master'a geçilir (Sec.6).

## 2. Donanım gerçeği (20 Eyl 2026'da bu makinede okundu)

| Bileşen | Değer |
|---|---|
| Makine | Acer Aspire **AG15-41P** (laptop) |
| CPU | AMD **Ryzen 7 7735HS** — 8 çekirdek / 16 iş parçacığı, 3.2 GHz taban |
| GPU | AMD Radeon 680M (`AMD Radeon(TM) Graphics`), **gfx1035**, RDNA2, **12 CU** (torch `multi_processor_count=6` = WGP sayısı) |
| GPU'nun gördüğü bellek | **7.14 GB** (paylaşımlı sistem RAM'i; Windows `AdapterRAM=2 GB` yanıltıcıdır) |
| RAM | **2× DIMM 16 GB = 32 GB DDR5-5600 SODIMM** — Hynix `HMCG78AGBSA092N` (`P0 CHANNEL A`) + Micron `MTC8C1084S1SC56BD1` (`P0 CHANNEL B`), **4800 MT/s**'de çalışıyor. **27 Eyl 2026'da ikinci SODIMM takıldı** |
| Boş RAM slotu | **YOK — iki slot da dolu** (Channel A + B), anakart maks **64 GB** |
| Bellek kanalı | **ÇİFT KANAL AKTİF (27 Eyl 2026)** — eski kayıt "tek kanal ≈ 38 GB/s teorik, ana darboğaz" idi; bant genişliği beklenen ~2×. Kazancın eğitim hızına yansıması ilk bench7 koşusunda doğrulanacak |
| Sürücü | AMD `32.0.21045.9003` (27 Ağu 2026) — KB'nin eski kaydı 21043.19003'ten güncellenmiş |
| OS | Windows 11 Pro build 26200 |
| Güç planı | **"Nihai Performans"** aktif (doğrulandı). Laptop prize takılı olmalı. |
| Toplam RAM | **29.75 GB fiziksel / ~27.7 GB görünür** (27 Eyl 2026; eski kayıt 14.7/13.7 GB idi) |

**Donanım tavsiyesi — GERÇEKLEŞTİ (27 Eyl 2026):** ikinci SODIMM (Hynix DDR5-5600 16 GB) boş
slota takıldı → **çift kanal aktif**, bant genişliği beklenen ~2×. iGPU eğitimi (özellikle
gather/unfold ve dW adımları) bant sınırlıydı; bu, yazılımla ulaşılamayan sıçramaydı. Eski
ölçümler (Sec.5'teki ~1.40M örnek/sn şampiyon dahil) TEK KANALDA alınmıştır — çift kanal
kazancı bench7 ile yeniden ölçülüp not edilmeli (ölçüm gelince Sec.5 güncellenir).

## 3. Yazılım ortamı

### 3.1 Birincil ortam — `venv-rocm10` (bu klasörde, doğrulandı)

| Bileşen | Değer |
|---|---|
| Python | 3.12.10 64-bit |
| torch | `2.15.0a0+rocm10.1.0a20260822` (+ `amd-torch-device-gfx1035`) |
| HIP | `7.16.26332` |
| ROCm SDK | `10.1.0a20260822` (core / device-gfx1035 / libraries) |
| Ek | numpy 2.4.4, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0, onnx 1.23.0, onnxruntime 1.30.0 |
| GPU | `is_available()=True`, `gcnArchName=gfx1035` |
| Offline yedek | `ml\wheels-rocm10\` (34 wheel, 1.2 GB) — internetsiz yeniden kurulum `ml\README.txt` |

### 3.2 Yedek ortam — `<ESKI_VENV_THEROCK>`

torch `2.12.0+rocm7.14.0a20260519`, HIP 7.13. Benchmark'ta venv-rocm10'dan %3–7 daha hızlı
ölçülmüştü (fp32 455k vs 424k); kullanıcı yine de güncel build'i birincil seçti. Sorun çıkarsa
geri dönülebilir. Dokunma.

### 3.3 Yedek — ROCm 10.0.0 KARARLI sürüm (`ml\wheels-rocm10-stable\`, venv kurulu değil)

20 Eyl 2026'da ayrı venv'e kuruldu, test edildi (kayıt: `ml\bench\stable_rocm10_test.txt`), karar verildikten sonra
venv silindi; **wheel arşivi duruyor** — gerekirse 2 dakikada offline kurulur (`ml\README.txt`). venv-rocm10'a dokunulmadı.

| Bileşen | Değer |
|---|---|
| torch | `2.13.0+rocm10.0.0` (+ `amd-torch-device-gfx1035 2.13.0+rocm10.0.0`) |
| HIP | `7.15.26333` |
| ROCm SDK | `10.0.0` (core / device-gfx1035 / libraries), rocm-bootstrap 0.2.0, numpy 2.5.3 |
| GPU | `True`, `gfx1035`, 7.14 GB — `GPU_SMOKE_TEST` geçti, hiprtc harness geçti, `selftest` geçti |
| DLL adları | `amdhip64_7.dll`, **`hiprtc0715.dll`** (10.1 nightly'de `hiprtc0716.dll`) → `hip_kernel.py` artık glob ile bulur |
| Offline arşiv | `ml\wheels-rocm10-stable\` (17 wheel, 1.1 GB) |

Aynı oturum A/B (`bench7_fused_e2e.py`, 3 deneme):

| Ölçüm | venv-rocm10 (torch 2.15 / HIP 7.16) | venv-rocm10-stable (torch 2.13 / ROCm 10.0.0) |
|---|---|---|
| **fused şampiyon** (HIP L1) | **1,442,474 / 1,430,773 / 1,421,515** | 1,342,144 / 1,355,358 / 1,366,183 (**−5%**) |
| saf PyTorch baseline (`BNCH_FUSED=0`, fp16+TunableOp) | 1,304,097 / 1,301,029 / 1,308,710 | 1,305,473 / 1,309,739 / 1,311,824 (eşit) |
| hipBLASLt açık (`HIPBLASLT_ENABLED=1`) | çöker | **hâlâ çöker** (`TensileLibrary_lazy_gfx1035.dat` yok → `HIPBLAS_STATUS_INVALID_VALUE`) |
| selftest fwd err | 0.000000 | 0.001953 (fp16 yuvarlama; grad err 0) |

**Karar:** birincil **venv-rocm10 kalır** (fused yolda %5 önde, tüm paketler kurulu). Stable wheel arşivi
pinlenmiş / tekrar üretilebilir **yedek**tir; nightly bir gün bozulursa buradan kurulur. Kernel aynı olduğu
için fark backward'daki rocBLAS `mm`'den geliyor olmalı.

**Kurulum tuzağı (önemli):** `pip install` → `WinError 206` (yol çok uzun). Stable wheel'in `licenses/`
ağacı ~230 karakter; `LongPathsEnabled=0` ve admin yok. Çözüm: `subst S: "<PROJE>"`
→ `S:\venv-rocm10-stable\Scripts\python.exe -m pip install ...` → `subst S: /d`. (Ya da admin ile
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`.)

### 3.4 Paket kaynakları — **URL yerleşimi değişti (20 Eyl 2026 tespiti)**

AMD, TheRock paket indekslerini `repo.amd.com` altına taşıdı. KB'nin eski adresi artık **legacy**:

| Kanal | Adres | gfx1035 Windows durumu |
|---|---|---|
| **Stable (kararlı)** | `https://stable.repo.amd.com/rocm/whl-next/` | **ROCm 10.0.0** — torch 2.11.0 / 2.12.0 / 2.13.0 (+torchvision), cp310–cp314 win_amd64. gfx1035 için **ilk kararlı sürüm.** |
| Nightly | `https://nightly.repo.amd.com/rocm/whl-next/` | ROCm 10.2.0a nightly; torch 2.12/2.13/2.14/2.15a0; en son 20260920 |
| Legacy nightly (KB'nin eskisi) | `https://rocm.nightlies.amd.com/whl-multi-arch/` | **Donmuş** — gfx1035 için son wheel 20260822 (kurulu olanın aynısı). Yeni sürüm gelmez. |

Notlar:
- `SUPPORTED_GPUS.md` (TheRock): gfx1035 Windows = Build ✅ / Sanity Tested ✅ / **Release Ready ✅**.
- **triton** her iki indekste de yalnız **linux_x86_64** → Windows'ta `torch.compile(inductor)` hâlâ yok.
- Kurulum komutu (yeni venv'e, kararlı):
  ```bat
  py -3.12 -m venv venv-X
  venv-X\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy pandas scikit-learn lightgbm
  ```
  Nightly için `--pre` ve `nightly.repo.amd.com` kullan. Kurulum sonrası **mutlaka** `GPU_SMOKE_TEST.py` + `ml\fxgpu\selftest.py`.

Resmî kaynaklar: `https://github.com/ROCm/TheRock/blob/main/RELEASES.md`,
`https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md`,
`https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md`.

## 4. Her oturumun başında

```bat
cd <PROJE>
GPU_PYTHON.cmd -c "import torch;print(torch.__version__,torch.version.hip,torch.cuda.is_available(),torch.cuda.get_device_properties(0).gcnArchName,torch.cuda.tunable.is_enabled())"
```
Beklenen: `... True gfx1035 True`. Şüphe varsa: `GPU_PYTHON.cmd GPU_SMOKE_TEST.py` (fp32 CNN smoke) ve
`GPU_PYTHON.cmd ml\fxgpu\selftest.py` (HIP fused kernel doğruluk). Tam pipeline sağlığı:
`GPU_PYTHON.cmd ml\bench\bench7_fused_e2e.py` → 3 deneme **~1.39–1.44M örnek/sn**, `err=0.0000`.
Dil modeli paketi: `GPU_PYTHON.cmd ml\fxlm\selftest.py` → `SELFTEST OK` (~6 s).

`GPU_PYTHON.cmd` şunları ayarlar: `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`,
`PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=200`, `HIP_VISIBLE_DEVICES=0`; sonra `venv-rocm10` python'unu çağırır.

## 5. HIZ REÇETESİ — ölçülmüş merdiven (500k × 40 özellik, pencere W=30, bs=32768)

| Adım | örnek/sn | Kazanç | Not |
|---|---|---|---|
| FP32 eager, DataLoader'sız GPU-index reçetesi | ~455k | 1.0× | eski baseline (tipik AI scriptlerinden zaten 10–50× hızlı) |
| + saf fp16 (`model.half()`, `X.half()`, loss `.float()`) | ~900–930k | 2.0× | RDNA2 paketli fp16 |
| + TunableOp (rocBLAS yolu, Lt kapalı) | ~1.24–1.28M | 2.7× | fp16 GEMM'de +%38 |
| + dropout kaldır, NN layout (saf PyTorch tavanı) | ~1.24M (Eylül başı) → **~1.30M** (20 Eyl aynı oturum, `BNCH_FUSED=0`) | 2.9× | dropout ~%10 yer |
| **+ el yazısı HIP fused gather+GEMM L1** (`FusedGatherLinear`) | **~1.40–1.44M** | **3.1×** | saf PyTorch'a +%8–10; grad'lar bit-bit eşit |
| + fp32 master ağırlık (kararlılık için, Sec.6) | ~1.41M | 3.1× | maliyet ~%1 |
| 2M-parametreli model (1200→1024→768→1) aynı yolla | ~268k | — | ~1.84 s/epok (500k satır) |

20 Eyl 2026 tekrar ölçümleri (`bench7_fused_e2e.py`, aynı gün 3 koşu): 1,400,028 / 1,408,611 / 1,395,568 →
1,407,791 / 1,398,414 / 1,393,630 → 1,442,474 / 1,430,773 / 1,421,515 örnek/sn; hepsinde `fwd/dW/db err = 0.0000`.
Koşudan koşuya ±%2 varyans normaldir (TunableOp tuning + termal).

### 5.1 Reçetenin kuralları (her biri bu donanımda ölçüldü)

1. **DataLoader YOK.** Tüm X/y tensörleri baştan GPU'da; batch = `X[rows]` GPU indexleme. Windows'ta
   DataLoader+workers yavaş/kilitlenir.
2. **Pencere = `X.unfold(0, W, 1)` görünümü** (kopya yok) ya da daha iyisi **fused kernel** (pencere
   hiç materyalize edilmez).
3. **Conv1d YASAK** (gfx1035 MIOpen generic → ~50× yavaş). Pencereli model = `wflat`: `(B,W,C)→(B,W·C)→MLP`.
4. **Büyük batch:** 16384–32768. 65536 daha yavaş ölçüldü. Küçük batch (2048) 4.6× daha yavaş.
5. **Saf fp16**, autocast/AMP değil. Loss fp32'de: `lossf(out.float(), y)`. `clip_grad_norm_(5.0)`.
6. **Sync yok:** `loss.item()`/`float()` batch içinde yok; `loss_sum += loss.detach()` GPU'da birikir,
   `synchronize()` sadece epok sonunda. `isfinite` kontrolü seyrek (örn. her 50 batch'te bir).
7. `torch.set_float32_matmul_precision("high")`, `AdamW(fused=True)`, `randperm`/`randint` device'da.
8. Eval `torch.inference_mode()` + bs 32768; AUC GPU'da rank ile hesaplanır (`train_template.py`).
9. **Dropout yerine weight_decay** (dropout ~%10 hız yer).

### 5.2 TunableOp — doğru kullanım

- Env var'lar `import torch`'tan **önce** ayarlanmalı → `GPU_PYTHON.cmd` veya `fxgpu.setup_env()`.
- `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` şart: varsayılan hipBLASLt yolu gfx1035'te
  `TensileLibrary_lazy_gfx1035.dat / Kernels.so-000-gfx1035.hsaco file not found` ile çöker (bu
  arch için Lt kernel dosyaları pakette yok). rocBLAS yolu çalışır ve fp16 GEMM'de ~%38 kazandırır.
- Tuning ilk iterasyonlarda olur → isınma döngüsü bırak (15 adım yeter). `MAX_TUNING_DURATION_MS=150–200` stabil.
- **CSV'yi tekrar kullanma** (`TUNING=0` + kayıtlı csv ~933k'ya düşürdü; solution-id'ler farklı
  kernel'a eşleniyor). Her koşuda taze tuning.
- Her koşu çalışma dizinine `tunableop_results0.csv` yazar (TunableOp varsayılanı). Zararsızdır,
  silinebilir; tuning açıkken aynı şekiller için sonuçları yeniden kullanır (bench7'de 1.40M ile
  doğrulandı). Hız açıklanamaz biçimde düşerse önce bu dosyayı sil.

### 5.3 Fused HIP kernel (`ml\fxgpu\gather_gemm.hip`)

`out[b,n] = bias[n] + Σ_{w,f} X16[idx[b]+w, f] · W1[n, w·F+f]` — gather + bias + GEMM tek kernel;
`Xg` (B,K) fp16 kopyasını da yazar ki backward'da `dW = gᵀ·Xg` rocBLAS ile hızlı olsun (fused dW
denendi: doğru ama 14.4 ms vs rocBLAS 8.5 ms → kullanılmadı). hiprtc ile **runtime derlenir**,
host derleyici (hipcc/MSVC) **gerekmez**. Teknik: k-çifti paketli sütun-major LDS + interleaved lane
mapping (bank-conflict-free), `v_dot2` (`__builtin_amdgcn_fdot2`) fp32 akümülasyon, 128×128×32 tile,
8×8 thread tile, çift buffer. İzole GEMM 3.5 TF/s (rocBLAS 3.19).

Kısıtlar: `F % 8 == 0` (16 B hizalı float4 yükleme; özellik sayısını sıfır sütunla 8'in katına
doldur — `train_template.py` bunu otomatik yapar), `idx` int32 ve `idx + W - 1 < R`, tüm tensörler
contiguous fp16, tek GPU.

## 6. Kararlılık: fp16'nın riski ve fp32 master ağırlık reçetesi

- fp16 AdamW rastgele etiketlerde ~10–15 adımda ıraksadı (|w| 0.09 → 9.4 ilk adımda). Kernel temiz
  (GEMM rel err 3.4e-4). Gerçek veride: **lr warmup + fp32 master ağırlık** kullan.
- **fp32 master** = parametreler fp32'de (AdamW fp32), her adımda `.half()` cast ile fp16 hesap;
  autograd grad'ları fp32 master'a geri akıtır. `WindowMLP(master=True)` bunu yapar.

**Ölçüm (20 Eyl 2026, `ml\bench\bench9_fp32master.py`, çıktı `ml\bench\bench9_out.txt`):** aynı model/veri,
öğrenilebilir etiket (`y = X[t,0] + 0.5·X[t-1,1] > 0`), lr 1e-3, 300 adım:

| Mod | örnek/sn | fp16'ya göre | 300 adım sonra |
|---|---|---|---|
| saf fp16 (AdamW fp16) | 1,422,089 | 100% | **NaN — 50. adımdan önce ıraksadı** |
| **fp32 master** (AdamW fp32, `.half()` cast) | 1,406,516 | **98.9%** | loss 0.246 → 0.0012, sonlu, \|w\|max 0.37 |
| fp32 master + statik loss scale 256 | 1,393,445 | 98.0% | loss 0.246 → 0.0010 |

**Sonuç: fp32 master ağırlık ~%1 hıza mal olur ve ıraksamayı bitirir → varsayılan budur** (`WindowMLP(master=True)`,
`train_template.py` varsayılanı). Saf fp16 AdamW gerçek eğitimde kullanılmaz. Statik loss scale ekstra
%1 daha yer; yalnız grad underflow şüphesinde (`--loss-scale 256`). `train_template.py --no-master` sentetik
veride de ıraksadı (loss 0.71 → 1.88) — KB'nin fp16 uyarısıyla tutarlı.

Kural: ilk gerçek eğitimde `isfinite` izle; NaN görülürse sırasıyla (1) lr düşür / warmup uzat,
(2) `--loss-scale 256` (statik), (3) fp32'ye dön (`WindowMLP(master=False)` değil — tamamen fp32 yol,
Sec.5 hızının yarısı).

## 7. `ml\fxgpu\` paketi — tablo/pencere MLP başlangıç noktası

```text
ml\fxgpu\
  __init__.py        FusedGatherLinear, fused_gather_linear, WindowMLP, setup_env, require_gpu
  env.py             setup_env() -> TunableOp env var'ları (import torch'tan ÖNCE çağır); require_gpu()
  hip_kernel.py      hiprtc + ctypes harness (DLL'leri glob ile bulur: amdhip64*.dll, hiprtc*.dll)
  gather_gemm.hip    fused kernel kaynağı (tek kopya — kernel burada yaşar)
  fused_linear.py    FusedGatherLinear autograd.Function — K/F/W şekilden türetilir, F%8 kontrolü,
                     inference_mode'da Xg yazmaz (%15 hızlı eval)
  model.py           WindowMLP(n_feat, window, hidden, hidden2, master=True): L1 fused, L2/L3 Fn.linear
  train_template.py  NPZ -> normalize -> fp16 GPU -> fused eğitim -> rank-AUC eval -> model.pt
  selftest.py        fused vs referans (unfold + Fn.linear) doğruluk testi
ml\bench\
  bench7_fused_e2e.py   şampiyon uçtan uca benchmark = sağlık kontrolü (BNCH_FUSED=0 → saf PyTorch baseline)
  bench8_2m.py          2M-parametreli model hız testi
  bench9_fp32master.py  saf fp16 vs fp32 master hız + kararlılık
  bench10_lm.py         GPT hız + yakınsama benchmark'ı (fxlm, Sec.12) + bench10*_out.txt kayıtları
  bench9_out.txt, stable_rocm10_test.txt   ölçüm kayıtları
```
bench7–9 `fxgpu`'yu import eder — paket bozulursa bench7 de bozulur (istenen: tek gerçek kaynak).

Kullanım:

```python
from fxgpu import setup_env; setup_env()      # import torch'tan ONCE
import torch
from fxgpu import require_gpu, WindowMLP
dev = require_gpu()
X16 = X_norm.half().to(dev)                    # (R, F) F%8==0
model = WindowMLP(n_feat=F, window=30, master=True).to(dev)
idx32 = torch.randint(0, R - 30 + 1, (32768,), device=dev, dtype=torch.int32)
out = model(X16, idx32)                        # (B,1) fp16 ; label = y[idx32 + 29]
```

Hazır script:

```bat
GPU_PYTHON.cmd ml\fxgpu\train_template.py --npz VERI.npz --window 30 --epochs 5 --batch 32768
```
NPZ sözleşmesi: `X (n,f) float`, `y (n,) 0/1` (y[t], t satırındaki pencere sonu için **gelecek**
hedef olmalı — sızıntı sorumluluğu veri hazırlayanda), isteğe bağlı `ts`, `cols`. Son %20 test
(zaman sıralı), normalize yalnız train istatistikleriyle. Varsayılanlar: fp32 master açık, warmup
otomatik (`min(200, max(10, toplam_adım/10))`), weight_decay 1e-4, dropout 0, `--check-every 50`.

**Referans koşu (20 Eyl 2026, sentetik 500k×40, öğrenilebilir etiket):**

```text
bs=32768 --epochs 5 : warmup=10 (auto) | 13 adim/epok, 65 toplam adim
  ep1 auc=0.7016 [1.5s]  ep2 0.8551 [0.3s 1,333,698/s]  ep3 0.9095  ep4 0.9428  ep5 0.9601 [0.3s 1,328,026/s]
bs=8192  --epochs 3 : warmup=14 (auto) | 49 adim/epok, 147 toplam adim
  ep1 auc=0.9188 [1.6s]  ep2 0.9720 [0.5s 852,437/s]  ep3 0.9824 [0.5s 867,305/s]
```

**Ders — örnek/sn tek ölçüt değil:** bs=32768 en yüksek throughput'u verir ama 400k satırda epok başına
yalnız 13 güncelleme yapar; bs=8192 %35 daha yavaş akar ama 4× güncelleme ile **daha az duvar saatinde
daha yüksek AUC**'ye ulaşır. Veri < ~2M satırsa **bs 8192–16384** ile başla; veri büyüdükçe 32768'e çık.
İlk epok her koşuda ~1.5 s daha uzun (hiprtc derleme + TunableOp tuning) — normal.

## 8. Denenip ELENENLER (tekrar deneme — zaman kaybı)

| Deneme | Sonuç |
|---|---|
| `torch.compile` backend=`cudagraphs` | ~352k (Dynamo overhead) |
| `torch.compile` inductor | çalışmaz — Windows'ta triton yok (stable/nightly indekslerde de yok) |
| CUDA/HIP Graph (statik buffer) | fp32'de +%5, fp16'da kazanç yok (~925k), tam-adım graph ~859k |
| bs=65536 | ~1.15M (32768'den yavaş) |
| Lagged matris `Xf` materyalize | ~881k (daha yavaş) |
| `fp16_accum=1` | fark yok |
| Triton fused gather+GEMM | `tl.dot` gfx1035'te FMA yoluna düşüyor — yavaş |
| TunableOp CSV yeniden kullanımı | ~933k (bozuk) |
| Fused dW HIP kernel | doğru ama 14.4 ms vs rocBLAS 8.5 ms |
| Conv1d | ~50× yavaş |
| DataLoader + num_workers | Windows'ta yavaş / kilitlenme |
| LightGBM GPU (OpenCL) | 16-thread CPU'dan hızlı değil → GBM CPU'da |
| hipBLASLt (TunableOp varsayılanı) | gfx1035'te çöker |
| hipcc host derleme toolchain (amdclang/msvc-sdk/xwin, 2.4 GB) | gereksiz — hiprtc runtime derleme yeter; 20 Eyl 2026'da silindi |
| SDPA flash / mem-efficient backend (gfx1035) | "available" görünür ama "No available kernel" ile math'e düşer — el yazısı flash kernel 4.5-5.8× daha hızlı (fxlm) |
| Fused RMSNorm HIP kernel (fxlm) | F.rms_norm'dan hızlı değil (0.84-0.96×, 8192×768 fwd+bwd 3.5 vs 3.4 ms) → varsayılan `norm_impl="torch"` |

## 9. Sık sorunlar

**`torch.cuda.is_available() == False`** → sırayla: doğru python mu (`venv-rocm10`)? AMD sürücü
Aygıt Yöneticisi'nde etkin mi? `pip show amd-torch-device-gfx1035` var mı? `torch.version.hip` boş mu?
Yeniden başlat. Hâlâ yoksa ortamı `ml\wheels-rocm10`'dan yeniden kur (`ml\README.txt`).

**TunableOp `...hsaco file not found` / `TensileLibrary_lazy_gfx1035.dat`** → `HIPBLASLT_ENABLED=0` unutulmuş.

**`hiprtcCompileProgram failed`** → `[hiprtc log]` satırını oku; genelde include yolu
(`_rocm_sdk_core\include`) veya `--offload-arch=gfx1035`. DLL adı değişmişse `fxgpu\hip_kernel.py`
glob ile bulur; bulamazsa bin klasörünü listeler.

**Bellek yetersiz** → batch yarıya; tarayıcı/oyun kapat; tek eğitim; veri shard'la (500k×40 fp16 = 40 MB,
Xg buffer 32768×1200 fp16 = 78 MB — genelde sorun olmaz).

**Hız 1.4M'nin çok altında** → (1) env var'lar set mi (`torch.cuda.tunable.is_enabled()`)? (2) başka
GPU süreci var mı? (3) laptop prizde + "Nihai Performans"? (4) isınma yapıldı mı? (5) F%8 slow-path'e
düşmüş olabilir (F'yi doldur).

**MIOpen `CK grouped conv library not found`** → uyarı, conv yine çalışır (bizim yolda conv yok).

**PC dondu / ekran takıldı (uzun GPU işi)** → iGPU ekranı da sürdüğü için uzun ve sınırsız GPU
denemeleri (özellikle ilk TunableOp tuning'i) sistemi kilitleyebilir; 24 Eyl 2026'da bir kez donma
yaşandı, reset gerekti. Deneyleri önce küçük şekillerle, zaman aşımıyla ve çıktıyı dosyaya yazarak
çalıştır; tek kernel çağrısı ~0.5 sn'yi geçmesin (Windows TDR ~2 sn).

**Görüntü/CNN modelleri (oyun/görüntü CNN deneyimi, gfx1035 Windows)** → `BatchNorm2d` bazı kurulumlarda
askıda kaldı → GroupNorm tercih et; ilk MIOpen çekirdek hazırlığı 30–60 sn sürebilir (donma değil);
CNN'de FP32 varsayılan (autocast/saf fp16 görüntü modelinde doğrulanmadı); `channels_last` kapalı;
`num_workers=0`; batch 4–8 ile başla.

## 10. Tablo / pencere modeli için yöntem notları

- **Tablo özellik + GBM (LightGBM, 16 thread CPU)** çoğu tablo verisinde NN'den iyi/eşit ve
  saniyeler sürer — **önce baseline olarak bunu kur**. GPU NN'i pencere/sekans modelleri için kullan.
- Zaman sıralı veride **zaman sıralı bölme** şart (son %20 test), rastgele split sızıntı verir. Normalizasyon yalnız train'den.
- Metrik: AUC + top-%10 güven doğruluğu (`train_template.py` yazar); accuracy tek başına yanıltır.
- fp16 için özellikleri normalize et (z-score); ham değerler fp16'da taşar.
- Küçük model + weight_decay > büyük model + dropout (hem hız hem genelleme bu veri rejiminde).

## 11. Tablo eğitimi için karar ağacı ve kontrol listesi

```text
GPU_PYTHON.cmd ile True + gfx1035 + tunable True mu?   Hayır -> Sec.9
selftest.py geçti mi (fused == referans)?               Hayır -> ortam/kernel; eğitim yok
Veri: zaman sıralı split, sızıntı yok mu?               Hayır -> veriyi düzelt
Baseline LightGBM AUC'si alındı mı?                     Hayır -> önce onu al
NN: train_template --epochs 1 ile hız ~1M+/sn ve loss sonlu mu?  Hayır -> Sec.9 hız / Sec.6 NaN
Uzun eğitim -> test AUC baseline'ı geçiyor mu?          Hayır -> model/özellik; hızla oynamayı bırak
```

- [ ] `GPU_PYTHON.cmd` kullanıyorum (doğru python + env var'lar).
- [ ] `venv-rocm10`'a dokunmadım.
- [ ] Tek GPU süreci; laptop prizde; "Nihai Performans".
- [ ] Saf fp16 + fp32 master (`master=True`), warmup açık, `isfinite` seyrek kontrol.
- [ ] DataLoader/Conv1d/dropout yok; bs 16384–32768.
- [ ] F % 8 == 0 (dolduruldu), `idx + W - 1 < R`.
- [ ] Zaman sıralı split, train-istatistikli normalize.
- [ ] Ayrı çıktı dosyası; eski model silinmedi.

## 12. Dil modeli / transformer eğitimi — `ml\fxlm\` (24 Eyl 2026)

Küçük GPT tarzı dil modeli eğitim paketi; kernel'lar hiprtc ile runtime derlenen
el yazısı HIP (host derleyici yok). Naif PyTorch GPT baseline'ına göre **1.6-2.7×
hızlı** (aşağıdaki tablo).

**İçerik** (`ml\fxlm\`):

| Dosya | İş |
|---|---|
| `kernels\attention.hip` | flash-attention fwd + bwd (prep/dkv/dq), fp16 io, fp32 softmax, fdot2 |
| `kernels\norm.hip` | fused RMSNorm fwd/bwd (F.rms_norm'dan hızlı değil — varsayılan torch) |
| `kernels\ce.hip` | satır-başına online logsumexp CE + fp16 grad yazımı |
| `kernels\optim.hip` | FlatAdamW: sumsq+nonfinite, final, flat update |
| `attention.py` | `flash_attention(q,k,v)` ve `flash_attention_qkv(qkv)` — (B,T,H,Dh) fp16, strided görünümler kopyasız |
| `norm.py` | `rms_norm(x, w)` |
| `cross_entropy.py` | `fused_cross_entropy`, `linear_cross_entropy` (tam (N,V) logits hiç yazılmaz; grad_scale float veya 0-dim CUDA tensor) |
| `optim.py` | `FlatAdamW` — fp32 master + fp16 flat param + dinamik loss scale + global clip, host sync yok |
| `model.py` | `GPTConfig` + `GPT` (pre-norm, bias yok, tied embedding varsayılan) |
| `train_lm.py` | eğitim CLI — GPU'da token dizisi, device randint batch, cosine LR, atomic ckpt, --resume |
| `prepare_bytes.py` | metin/glob/dizin → byte-level uint16 .bin + .json (--ext dizin filtresi, NUL'lu binary'ler atlanır) |
| `sample.py` | ckpt + prompt → metin üretimi (temperature/top-k) |
| `selftest.py` | tüm kernel'ların PyTorch parity testi (~6 s) |
| `bench_kernels.py` | kernel mikrobenchmark'ları → `bench_kernels_out.txt` |
| `kernel_regs.py` | kernel başına vgpr/scratch/LDS (ELF .kd parse) — spill teşhisi |

**fxlm'e özgü kurallar:**

- TunableOp **varsayılan kapalı** (`setup_lm_env` GPU_PYTHON.cmd'nin =1'ini ezer);
  açmak için `train_lm.py --tunable` — ilk tuning dakikalar sürer, ekran takılabilir.
- Micro-step başına `batch × seq_len ≤ 16384`; üstü için `--accum`.
- Parametreler saf fp16 (flat buffer görünümleri), master fp32 + dinamik loss scale;
  autocast yok, adım başı cast yok.
- Her eğitim yeni `--out` klasörü; var olan ckpt'li klasör `--resume` olmadan reddedilir.
- Tek GPU süreci (iGPU ekranı da sürüyor).

**Hız** (`ml\bench\bench10_out.txt`, TunableOp kapalı, ms/adım):

| Config | baseline (SDPA+LN+autocast) | fxlm (tümü fused) | fxlm_torchnorm (önerilen) |
|---|---|---|---|
| T256 B16 | 788.5 ms — 5,195 tok/s | 492.4 ms — 8,318 tok/s — 1.60× | 420.2 ms — 9,748 tok/s — 1.88× |
| T512 B8 | 981.1 ms — 4,175 tok/s | 530.1 ms — 7,726 tok/s — 1.85× | 450.8 ms — 9,086 tok/s — 2.18× |
| T1024 B4 | 1406.3 ms — 2,913 tok/s | 513.7 ms — 7,974 tok/s — 2.74× | 513.8 ms — 7,973 tok/s — 2.74× |

Batch ölçekleme (`bench10_bigbatch_out.txt`): T256B32'de fxlm_torchnorm
796.6 ms — **10,284 tok/s** (baseline 5,405); T512B16'da 859.0 ms —
**9,536 tok/s** (baseline 4,299). fxlm batch'le ölçekleniyor, baseline etmiyor.

Yakınsama (`bench10_converge_out.txt`, byte-level 24 MB site-packages, 600 adım):
baseline 201.6 s → val 1.6323; **fxlm 97.6 s → val 1.6448** (2.07×, aynı öğrenme).

Uçtan uca deneme (24 Eyl 2026, `GPU_PYTHON.cmd` ile): 10.8M param (D=384 L=6 H=6, T256 B16,
byte-level ~30 MB .py metni), ~27,800 tok/s, 0 skip; val 2.4556@100 → 2.3536@200 → 2.2159@300;
`--resume` ile 300→400 kesintisiz devam, val 2.1444@400. Bu kadar kısa eğitimde üretilen metin
zayıftır — normal.

Kernel mikrobenchmark (`bench_kernels_out.txt`):

| Kernel | fused | PyTorch esdeğeri | hız |
|---|---|---|---|
| flash attn fwd+bwd B16 T256 Dh64 | 7.5 ms | SDPA math 34.1 ms | 4.5× |
| flash attn fwd+bwd B4 T1024 Dh64 | 19.0 ms | 107.1 ms | 5.7× |
| CE 4096×8192 fwd+bwd | 10.0 ms | fp32 cast 39.7 ms | 3.9× |
| CE 4096×16384 | 19.7 ms | 86.7 ms | 4.4× |
| linearCE 4096×512×8192 | 58.4 ms | linear+CE 82.6 ms | 1.4× |
| FlatAdamW.step 29.5M | 35.4 ms | scaler+clip+fused AdamW 47.2 ms | 1.3× |
| rms_norm 8192×768 fwd+bwd | 3.5 ms | F.rms_norm 3.4 ms | 0.96× (kazanç yok) |

**Dersler:**

a) gfx1035'te SDPA'nın flash/mem-efficient backend'i "available" raporlasa bile
   "No available kernel" hatasıyla math'e düşüyor — math backend'i yavaş
   (B16 T256 fwd+bwd ~34 ms). El yazısı flash kernel Dh64'te 4.5-5.8×,
   Dh128'de ~2.4× daha hızlı.
b) İç GEMM döngülerinde TAM `#pragma unroll` → VGPR tavanı (248) + thread başına
   ~1.3 KB scratch spill → backward 10-30× yavaşladı (dkv 81 ms, dq 24.6 ms idi;
   düzeltme sonrası fwd+bwd 6.7 ms). `#pragma unroll 4` çözdü. Spill teşhisi:
   `kernel_regs.py` (hipFuncGetAttributes modül handle'larında çalışmıyor —
   ELF .kd parse ediliyor).
c) El yazısı fused RMSNorm F.rms_norm'dan hızlı değil → varsayılan
   `norm_impl="torch"`; fused yol `--norm fused` ile duruyor.
d) Kalan adım süresi çoğunlukla rocBLAS matmul'lar. Sıradaki kaldıraçlar:
   daha büyük token batch (yukarıdaki bigbatch tablosu), TunableOp
   (riskli/ilk koşu yavaş), çift kanal RAM — **27 Eyl 2026'da yapıldı**, kazanç ilk ölçüde doğrulanacak.
