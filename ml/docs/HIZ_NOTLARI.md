# Radeon 680M (gfx1035) — En Hizli ML Egitimi: Olculmus Bulgular

Son guncelleme: 20 Eyl 2026 (tum sampiyon rakamlar bugun yeniden olculdu); 24 Eyl 2026: fxlm (dil modeli) Sec.7-8 eklendi.
Standart is yuku: 500k x 40 ozellik, wflat W=30, HID=256, bs=32768, BCE.
Ozet ve kurallar ana `README.md` Sec.5-6'da; bu dosya OLCUM KAYDIDIR.

## 0. GUNCEL SAMPIYON — fused HIP L1 + saf fp16 + TunableOp(rocBLAS) = ~1.40M ornek/sn

```bat
cd <PROJE>
GPU_PYTHON.cmd ml\bench\bench7_fused_e2e.py
```
20 Eyl 2026: `t0 1,400,028 | t1 1,408,611 | t2 1,395,568 ornek/sn`, `fwd/dW/db err = 0.0000`.
Yeniden kullanilabilir hali: `ml\fxgpu\` (FusedGatherLinear, WindowMLP, train_template.py).

| Teknik | ornek/sn | Not |
|---|---|---|
| **fused HIP L1 + fp16 + TunableOp(rocBLAS)** | **~1.40M** | SAMPIYON, tekrarlanabilir |
| fp16 + TunableOp + nodrop + NN-layout (PyTorch-only) | ~1.24M | HIP kernel'siz en iyi |
| fp16 + TunableOp(rocBLAS) | ~1.06-1.28M | tuning sansina bagli varyans |
| saf fp16 eager | ~900-930k | 2x FP32 |
| FP32 eager | ~455k | eski baseline |
| 2M-param model (1200-1024-768-1), fused yol | ~268k | ~1.84 s/epok |

### fp32 master agirlik maliyeti (bench9_fp32master.py, 20 Eyl 2026)

Ayni model/veri, ogrenilebilir etiket (`y = X[t,0] + 0.5*X[t-1,1] > 0`), lr 1e-3, 300 adim.
Tam cikti: `ml\bench\bench9_out.txt`.

| Mod | ornek/sn (medyan) | fp16'ya gore | 300 adim sonra |
|---|---|---|---|
| saf fp16 (AdamW fp16) | 1,422,089 | 100% | NaN — 50. adimdan once iraksadi, |w|max=nan |
| **fp32 master** (AdamW fp32, .half() cast) | 1,406,516 | **98.9%** | 0.2464 0.0773 0.0261 0.0082 0.0027 0.0012 (sonlu, |w|max 0.371) |
| fp32 master + loss scale 256 | 1,393,445 | 98.0% | 0.2460 0.0767 0.0255 0.0079 0.0025 0.0010 |

Peak alloc: 319 MB (fp16) vs 321 MB (master) — bellek farki yok.
KARAR: fp32 master varsayilan (~%1 maliyet, iraksama biter). Saf fp16 AdamW gercek egitimde YOK.

### Stable ROCm 10.0.0 (torch 2.13) kiyasi (20 Eyl 2026)

`venv-rocm10-stable` (torch 2.13.0+rocm10.0.0, HIP 7.15.26333). Kayit: `ml\bench\stable_rocm10_test.txt`.
Ayni oturum A/B, bench7, 3 deneme:

| Olcum | venv-rocm10 (2.15 / HIP 7.16) | venv-rocm10-stable (2.13 / 10.0.0) |
|---|---|---|
| fused sampiyon (HIP L1) | 1,442,474 / 1,430,773 / 1,421,515 | 1,342,144 / 1,355,358 / 1,366,183 (-%5) |
| saf PyTorch baseline (BNCH_FUSED=0) | 1,304,097 / 1,301,029 / 1,308,710 | 1,305,473 / 1,309,739 / 1,311,824 (esit) |
| hipBLASLt acik | coker | HALA coker (TensileLibrary_lazy_gfx1035.dat yok) |
| GPU_SMOKE / hiprtc / selftest | gecti | gecti (fwd err 0.001953, grad err 0) |

Not: saf PyTorch baseline bugun iki ortamda da ~1.30M — HIZ_NOTLARI'nin eski 1.22-1.24M kaydi
guncellendi (fused kazanci +%8-10). Fused fark kernel ayni oldugu icin backward rocBLAS mm'den.
KARAR: birincil venv-rocm10; stable = pinlenmis yedek. Kurulumda WinError 206 (uzun yol) ->
`subst S:` ile kur (README Sec.3.3).

### Inference (Xg yazmadan) — selftest.py, B=32768 forward

| Yol | ms |
|---|---|
| grad acik (Xg write-out var) | 12.05 |
| inference_mode (Xg=NULL, write-out yok) | 10.29 (-%15) |

### Batch vs guncelleme sayisi — train_template.py, sentetik 500k x 40 (ogrenilebilir etiket)

| bs | ornek/sn | adim/epok | AUC |
|---|---|---|---|
| 32768 | ~1.33M | 13 | ep3 0.9095, ep5 0.9601 |
| 8192 | ~0.86M | 49 | ep3 **0.9824** |

Throughput tek olcut degil: kucuk veri setinde (< ~2M satir) bs 8192-16384 daha az duvar saatinde
daha yuksek AUC verir. `--no-master` (saf fp16 AdamW) ayni veride iraksadi (loss 0.71 -> 1.88).

## 1. Elenen teknikler (olculdu, tekrar deneme)

| Teknik | ornek/sn | Neden |
|---|---|---|
| fp16 CUDA-graph (statik buf) | ~925-930k | replay avantajsiz; statik buffer kopyasi kazanci yiyor |
| fp16 tam-adim graph (randint dahil) | ~859k | ayni |
| FP32 + graph | ~484k | sadece +%5 |
| fp16 bs=65536 (+TunableOp) | ~1.15M | bs buyutme zararli |
| fp16 + Xf materyalize (lagged matris) | ~881k | daha yavas |
| fp16 + TunableOp, TUNING=0 + CSV | ~933k | CSV solution-id'ler baska kernel'a esleniyor |
| torch.compile("cudagraphs") | ~352k | Dynamo overhead |
| torch.compile inductor | calismaz | triton Windows'ta yok (stable/nightly indekslerde de yok) |
| Triton fused gather+GEMM | yikama | tl.dot gfx1035'te FMA yolu |
| fp16_accum=1 | fark yok | |
| TunableOp varsayilan (hipBLASLt acik) | COKER | `TensileLibrary_lazy_gfx1035.dat` / `.hsaco` yok |
| Fused gather+dW HIP kernel | 14.4 ms | rocBLAS dW 8.5 ms daha hizli (m-reduction tile geride) |
| dropout 0.1 | -%10 | 32768x256 RNG maskesi + kernel'lar |
| Conv1d | ~50x yavas | MIOpen generic |
| LightGBM GPU (OpenCL) | CPU'dan yavas | 16-thread CPU'da kal |

## 2. Tur-3 bulgulari (PyTorch seviyesi)

- **Dropout ~%10 yiyor.** nodrop 1.22M; NN-layout ile 1.24M. Gercek modelde weight_decay tercih.
- **NaN loss = benchmark artefakti:** fp16 AdamW rastgele etiketlerde ~10-15 adimda diverge
  (wmax 0.09->9.44 ilk adimda). Kernel dogrulamasi temiz (GEMM rel err 3.4e-4). Gercek veride
  lr warmup + fp32 master agirlik (bench9 sonucu yukarida).
- **TunableOp CSV tekrar-kullanimi bozuk.** Her kosuda fresh tuning; MAX_TUNING_DURATION_MS=150-200.
- **GEMM yonu:** microbench mm(x,WT) NN 4.6ms vs addmm-TN 6.6ms; uctan uca +%1-2 (varyans icinde).
- **CPU enqueue ~%50 ama kritik degil:** graph replay bile ~930k — GPU isi baskin.
- **Limitler:** ileri GEMM ~%92 peak, dW1 ~%55 peak, gather bant-sinirli (~5.2ms/iter, ~30GB/s
  tek kanal duvari) -> fused kernel bu gather'i yok etti.

## 3. Ham HIP kernel projesi (hiprtc) — gelistirme kaydi

hiprtc + ctypes ile torch tensor'larina direkt yazan HIP kernel'lari (`fxgpu\hip_kernel.py` harness;
host derleyici GEREKMEZ). NOT (20 Eyl 2026 temizligi): asagida adi gecen gelistirme scriptleri
(`bench_hip_gemm*.py`, `bench_hip_dW*.py`, `bench_hip_fused.py`, `bench_hip_uarch.py`) ve PyTorch-seviyesi
deneme scriptleri (`bench_gpu_speed.py`, `bench2..6`) kullanici karariyla SILINDI; nihai kernel
`fxgpu\gather_gemm.hip`'te, uctan uca olcum `bench\bench7_fused_e2e.py`'de yasar. Bu bolum, rakamlarin
nasil elde edildiginin kaydidir.

- **fdot2 == hfma2 rate** (ikisi tam-rate). Mikrobench dusuk saatte olculur — kiyaslamalari ayni
  islemde yap.
- **Bank-conflict analizi kritikti:** `Bs[tx*8+j][kk]` 8-16 yollu conflict (~1TF tavan). Cozum:
  pair-packed k-major shared layout (`As2[k/2][m]` half2) + interleaved lane mapping -> conflict-free.
- **Kazanan yapi:** load(buf^1) ONCE -> compute(buf) -> TEK sync. Register-staging spill yapti
  (LOCAL>0 -> performans coker).
- **Izole GEMM** (`bench_hip_gemm5.py`, 128x128x32/T8x8/pad0): **5.70 ms = 3.53 TF/s** vs rocBLAS
  6.3 ms / 3.19 TF — el yazisi kernel rocBLAS'i gecti.
- **Fused gather+GEMM** (`bench_hip_fused.py`): `out = X[idx+w][f] @ Wt` gather'i GEMM icine
  fuzeler, 78 MB materyalizasyon yok. 6.57 ms (Xg yazmadan) / 8.8 ms (Xg fp16 write-out ile) vs
  eski yol gather 5.2 + mm 5.9 = 11.1 ms.
- **gather_dW** (`bench_hip_dW*.py`): dogru (rel err 7e-7) ama 14.4 ms; rocBLAS dW 8.5 ms -> dW
  rocBLAS'ta kaldi, Xg write-out bu yuzden var.
- **Uctan uca** (`bench7_fused_e2e.py`): fused L1 ~1.39-1.41M vs baseline (index_select + Linear)
  ~1.22M = **+%14**; grad'lar bit-bit esit.
- Acik is: fused-dW 8.5 ms altina inerse Xg write-out kalkar (fwd 8.8->6.6 ms, toplam ~2 ms daha).
  SIMD sayisi (12 CU) ve LDS 64 KB -> 1 blok/CU siniri.

## 4. Donanim gercegi (darbogaz)

- **Tek DIMM:** 16 GB DDR5-5600 @4800, `P0 CHANNEL B / DIMM 0`, ikinci slot BOS, maks 64 GB.
  TEK KANAL ~38 GB/s. Ikinci SODIMM -> ~2x bant. Bu PC'deki en buyuk tek hamleli hiz artisi.
- iGPU = sistem RAM'i (7.14 GB gorunur); egitim sirasinda agir uygulamalari kapat.
- Laptop prize takili + "Nihai Performans" (20 Eyl 2026 dogrulandi aktif).
- Ryzen 7 7735HS 8C/16T; LightGBM/CPU isleri icin 16 thread.

## 5. Ortamlar

- **Birincil:** `..\venv-rocm10` torch 2.15.0a0+rocm10.1.0a20260822, HIP 7.16 (fp32 ~424k,
  fp16 ~874k, fused ~1.40M). `GPU_PYTHON.cmd` bunu cagirir.
- **Yedek:** `<ESKI_VENV_THEROCK>` torch 2.12.0+rocm7.14.0a20260519 — PyTorch-only
  benchmark'ta %3-7 onde (fp32 ~455k, fp16 ~900k).
- **Yedek (kurulu degil):** ROCm 10.0.0 kararli wheel arsivi `wheels-rocm10-stable\` — sonuc Sec.0;
  test venv'i karar sonrasi silindi.
- triton hicbirinde yok -> inductor kullanma.
- Inference: ONNX Runtime DirectML EP Windows'ta AMD iGPU'yu gorur (egitim degil).

## 6. Algoritma notu (tablo verisi)

- Tablo ozellik + GBM (LightGBM, 16 thread CPU): saniyeler, NN'den sik sik daha iyi -> baseline.
- Sekans/pencere modelleri -> GPU'da wflat fp16 fused (~1.4M/sn), `ml\fxgpu\train_template.py`.

## 7. fxlm — kucuk GPT egitimi (24 Eyl 2026)

`ml\fxlm\`: hiprtc-derlemeli HIP kernel'lari (flash attention, fused CE/linear-CE,
FlatAdamW fp32-master) ile GPT egitimi. Baseline = autocast fp16 + SDPA(MATH) +
LayerNorm + GradScaler + fused AdamW. Tum olcumler TunableOp KAPALI.
Tek-kosu ablation'larin ~%15 kosu-kosu varyansi var — farklari yorumlarken dikkat.

Uctan uca hiz (`ml\bench\bench10_out.txt`, her varyant ayri alt-surec,
8 isinma + 20 olculen adim):

    === T256: V=8192 D=512 L=8 H=8 T=256 B=16 (4096 tok/adim) ===
      baseline           788.5 ms/adim |     5,195 tok/s |      - | peak  1853 MB | 33.7M param | isinma 11.0s | loss 9.143
      fxlm               492.4 ms/adim |     8,318 tok/s |  1.60x | peak  1209 MB | 33.7M param | isinma 6.4s | loss 9.077
      fxlm_sdpa          701.9 ms/adim |     5,835 tok/s |  1.12x | peak  1560 MB | 33.7M param | isinma 9.0s | loss 9.076
      fxlm_torchnorm     420.2 ms/adim |     9,748 tok/s |  1.88x | peak  1209 MB | 33.7M param | isinma 6.2s | loss 9.075
      fxlm_torchce       449.5 ms/adim |     9,112 tok/s |  1.75x | peak  1421 MB | 33.7M param | isinma 6.5s | loss 9.075
      fxlm_tied          489.0 ms/adim |     8,375 tok/s |  1.61x | peak  1153 MB | 29.5M param | isinma 6.2s | loss 9.090

    === T512: V=8192 D=512 L=8 H=8 T=512 B=8 (4096 tok/adim) ===
      baseline           981.1 ms/adim |     4,175 tok/s |      - | peak  2111 MB | 33.8M param | isinma 12.6s | loss 9.144
      fxlm               530.1 ms/adim |     7,726 tok/s |  1.85x | peak  1213 MB | 33.8M param | isinma 6.5s | loss 9.072
      fxlm_sdpa          892.3 ms/adim |     4,591 tok/s |  1.10x | peak  1820 MB | 33.8M param | isinma 10.6s | loss 9.073
      fxlm_torchnorm     450.8 ms/adim |     9,086 tok/s |  2.18x | peak  1213 MB | 33.8M param | isinma 6.3s | loss 9.073
      fxlm_torchce       557.9 ms/adim |     7,342 tok/s |  1.76x | peak  1425 MB | 33.8M param | isinma 6.7s | loss 9.072
      fxlm_tied          447.3 ms/adim |     9,157 tok/s |  2.19x | peak  1157 MB | 29.6M param | isinma 6.5s | loss 9.093

    === T1024: V=8192 D=512 L=8 H=8 T=1024 B=4 (4096 tok/adim) ===
      baseline          1406.3 ms/adim |     2,913 tok/s |      - | peak  2625 MB | 34.1M param | isinma 16.5s | loss 9.138
      fxlm               513.7 ms/adim |     7,974 tok/s |  2.74x | peak  1215 MB | 34.1M param | isinma 7.0s | loss 9.077
      fxlm_sdpa         1334.0 ms/adim |     3,071 tok/s |  1.05x | peak  2511 MB | 34.1M param | isinma 14.0s | loss 9.078
      fxlm_torchnorm     513.8 ms/adim |     7,973 tok/s |  2.74x | peak  1215 MB | 34.1M param | isinma 6.9s | loss 9.078
      fxlm_torchce       547.8 ms/adim |     7,477 tok/s |  2.57x | peak  1427 MB | 34.1M param | isinma 7.2s | loss 9.077
      fxlm_tied          583.2 ms/adim |     7,023 tok/s |  2.41x | peak  1159 MB | 29.9M param | isinma 6.9s | loss 9.093

Batch olcekleme (`ml\bench\bench10_bigbatch_out.txt`):

    === T256: V=8192 D=512 L=8 H=8 T=256 B=16 (4096 tok/adim) ===
      baseline           786.6 ms/adim |     5,208 tok/s |      - | peak  1853 MB | 33.7M param
      fxlm_torchnorm     420.2 ms/adim |     9,749 tok/s |  1.87x | peak  1209 MB | 33.7M param

    === T256B32: V=8192 D=512 L=8 H=8 T=256 B=32 (8192 tok/adim) ===
      baseline          1515.7 ms/adim |     5,405 tok/s |      - | peak  3165 MB | 33.7M param
      fxlm_torchnorm     796.6 ms/adim |    10,284 tok/s |  1.90x | peak  1774 MB | 33.7M param

    === T512B16: V=8192 D=512 L=8 H=8 T=512 B=16 (8192 tok/adim) ===
      baseline          1905.7 ms/adim |     4,299 tok/s |      - | peak  3678 MB | 33.8M param
      fxlm_torchnorm     859.0 ms/adim |     9,536 tok/s |  2.22x | peak  1779 MB | 33.8M param

Yakinsama (`ml\bench\bench10_converge_out.txt`; byte-level, 24 MB site-packages
*.py, V=256 D=384 L=6 H=6 T=256 B=16, 600 adim lr 1e-3):

    baseline   params 10.92M | val loss [[150, 2.1697], [300, 1.9896], [450, 1.7548], [600, 1.6323]] | train(son50) 1.8442 | sonlu=True | egitim 201.6s
    fxlm       params 10.92M | val loss [[150, 2.165], [300, 1.9678], [450, 1.7476], [600, 1.6448]] | train(son50) 1.8526 | sonlu=True | egitim 97.6s

Kernel mikrobenchmark (`ml\fxlm\bench_kernels_out.txt`):

    === flash attention vs SDPA(math) — fwd | fwd+bwd ms ===
    attn B16 T256 H8 Dh64: flash    2.06 |    7.54 ms  sdpa   16.77 |   34.10 ms  speedup fwd 8.13x fb 4.52x
    attn B8 T512 H8 Dh64: flash    3.11 |   11.23 ms  sdpa   29.76 |   57.58 ms  speedup fwd 9.56x fb 5.13x
    attn B4 T1024 H8 Dh64: flash    5.45 |   18.95 ms  sdpa   57.49 |  107.06 ms  speedup fwd 10.55x fb 5.65x
    attn B16 T256 H8 Dh128: flash    6.11 |   20.87 ms  sdpa   23.36 |   50.71 ms  speedup fwd 3.82x fb 2.43x
    === rms_norm fwd+bwd vs F.rms_norm — ms ===
    rms 8192x768: fused   3.50 ms | torch   3.36 ms  speedup 0.96x
    rms 8192x512: fused   2.81 ms | torch   2.37 ms  speedup 0.84x
    === cross_entropy fwd+bwd — ms ===
    ce 4096x8192: fused  10.03 ms | torch-fp32  39.65 ms  speedup 3.95x
    ce 4096x16384: fused  19.68 ms | torch-fp32  86.65 ms  speedup 4.40x
    === linear_cross_entropy (fused head+CE) vs linear+CE — ms ===
    linearCE 4096x512x8192: fused  58.44 ms | torch  82.57 ms  speedup 1.41x
    === FlatAdamW.step vs GradScaler+clip+AdamW(fused) — ms ===
    adamw 29.5M param: FlatAdamW  35.42 ms | torch path  47.16 ms  speedup 1.33x

Attention backward VGPR-spill vakasi (once/sonra — ayni kernel, sadece
`#pragma unroll` -> `#pragma unroll 4` + lse/delta hoist + ds tek-hesap):

    ONCE (probe olcumu, 24 Eyl 2026 — bwd_dkv 81 ms, dq 24.6 ms, scratch 1332-1360B/thread):
    flash B16 T256 H8 Dh64 causal: fwd   2.512 ms | fwd+bwd 113.969 ms
    flash B8 T512 H8 Dh64 causal: fwd   3.621 ms | fwd+bwd 198.014 ms
    flash B4 T1024 H8 Dh64 causal: fwd   5.737 ms | fwd+bwd 497.986 ms
    flash B16 T256 H8 Dh128 causal: fwd   6.520 ms | fwd+bwd 294.964 ms
    flash B8 T512 H8 Dh128 causal: fwd  10.811 ms | fwd+bwd 558.145 ms

    SONRA (probe olcumu, 24 Eyl 2026 — scratch 0B, vgpr 192):
    B16 T256 H8 Dh64: fwd 2.101 | fwd+bwd 6.716 ms  (sdpa fb 33.311 ms -> 4.96x)
    B8 T512 H8 Dh64:  fwd 3.133 | fwd+bwd 10.404 ms (sdpa fb 56.386 -> 5.42x)
    B4 T1024 H8 Dh64: fwd 5.402 | fwd+bwd 18.120 ms (sdpa fb 105.140 -> 5.80x)
    B16 T256 H8 Dh128: fwd 6.207 | fwd+bwd 19.537 ms (sdpa fb 49.660 -> 2.54x)

    Teshis: `ml\fxlm\kernel_regs.py` (ELF .kd parse — vgpr/scratch/lds).
    hipFuncGetAttributes modul handle'larinda calismiyor (hipError 13).

## 8. Transformer darbogaz olcumleri — fxlm oncesi baseline (24 Eyl 2026, probe)

fxlm yazilmadan once naif PyTorch GPT'nin nerede vakit harcadigini olcen
probe serisi. Bu olcumler fxlm'in hedeflerini (attention, CE, norm,
optimizer, cast/kopya yuku) belirledi.

probe_tf.py: ortam/env taramasi + bant + fp16 GEMM, TunableOp ACIK
(GPU_PYTHON ortami). Cikti GEMM bolumunde kesiliyor — son sekil
8192x768x32768 (~412 GFLOP) tuning sirasinda PC'yi dondurdu; sonraki
probe'lar TunableOp KAPALI ve kucuk sekillerle yazildi:

```
2.15.0a0+rocm10.1.0a20260822 7.16.26332 AMD Radeon(TM) Graphics gfx1035 tunable True
sdp flags: flash True mem True math True
is_flash_attention_available True
copy 256MB: 17.99 ms -> 29.8 GB/s
mm 8192x768x3072 fp16: 8.67 ms 4.46 TF/s
mm 8192x3072x768 fp16: 9.69 ms 3.99 TF/s
mm 8192x768x768 fp16: 2.45 ms 3.94 TF/s
mm 8192x512x2048 fp16: 4.16 ms 4.13 TF/s
mm 8192x768x8192 fp16: 23.10 ms 4.46 TF/s
```

probe2.py: guvenli tekrar — her bolum ayri alt-surecte, TunableOp KAPALI.
fp16 GEMM (3.3-3.6 TF/s); SDPA backend'leri: MATH calisiyor, FLASH ve
EFFICIENT "No available kernel" ile dusuyor (B16 H8 T256 D64 fwd+bwd
35.7 ms; B8 T512 59.9 ms); LayerNorm/RMSNorm/manual-RMS fwd+bwd;
fp32-cast CE fwd+bwd + peak; kernel launch ~10.8 us; CUDA graph kazanci yok:

```
=== gemm ===
tunable: False
mm 8192x768x3072 fp16 (no tunable): 10.76 ms 3.59 TF/s
mm 8192x768x768 fp16 (no tunable): 2.71 ms 3.57 TF/s
mm 4096x768x16384 fp16 (no tunable): 31.19 ms 3.30 TF/s
=== attn_small ===
tunable: False
sdpa MATH                 B16 H8 T256 D64:   35.71 ms peak 248 MB
sdpa EFFICIENT_ATTENTION : FAIL RuntimeError: No available kernel. Aborting execution.
sdpa FLASH_ATTENTION     : FAIL RuntimeError: No available kernel. Aborting execution.
sdpa DEFAULT: 35.32 ms
=== attn_mid ===
tunable: False
sdpa MATH                 B8 H8 T512 D64:   59.94 ms peak 376 MB
sdpa EFFICIENT_ATTENTION : FAIL RuntimeError: No available kernel. Aborting execution.
sdpa FLASH_ATTENTION     : FAIL RuntimeError: No available kernel. Aborting execution.
sdpa DEFAULT: 59.64 ms
=== norm ===
tunable: False
layer_norm 8192x768 fwd+bwd: 4.940 ms (~10 GB/s min-traffic)
rms_norm 8192x768 fwd+bwd: 4.675 ms (~11 GB/s min-traffic)
rms_manual 8192x768 fwd+bwd: 27.915 ms (~2 GB/s min-traffic)
silu(a)*b 8192x3072: 8.185 ms | gelu_tanh: 3.599 ms
=== ce ===
tunable: False
CE 4096xV8192 fp32-cast fwd+bwd: 46.82 ms peak 512 MB
CE 4096xV16384 fp32-cast fwd+bwd: 99.74 ms peak 1024 MB
=== launch ===
tunable: False
tiny kernel: 10.8 us/launch
=== graph ===
tunable: False
100 small ops: eager 1.534 ms | graph 1.717 ms
PROBE2 DONE
```

probe3_gpt.py: naif baseline GPT adim suresi + ablation, her varyant ayri
alt-surecte, TunableOp KAPALI. Model: V=8192 D=512 L=8 H=8 T=256 B=16
(33.7M) — LayerNorm + SDPA + autocast fp16 + F.cross_entropy(fp32) +
GradScaler + fused AdamW. Varyantlar: noattn (attention cikti, y=v),
noce (CE yerine logits.mean()), nonorm (LayerNorm'lar bypass),
attn_manual (acik matmul+softmax attention), ce16 (fp16 logits'e CE),
aotriton (TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1), full_tunable
(TunableOp acik -> TIMEOUT). Not: probe3'teki 'no CE' varyanti gecerli
bir egitim hedefi degil (loss negatif) — sadece maliyet olcumu.

```
=== full ===
tunable: False
full         params 33.7M |   778.1 ms/step |    5,264 tok/s | peak 1915 MB | loss 9.141
=== noattn ===
tunable: False
noattn       params 33.7M |   487.5 ms/step |    8,402 tok/s | peak 1527 MB | loss 9.168
=== noce ===
tunable: False
noce         params 33.7M |   759.1 ms/step |    5,396 tok/s | peak 1785 MB | loss -4.057
=== nonorm ===
tunable: False
nonorm       params 33.7M |   749.2 ms/step |    5,467 tok/s | peak 1780 MB | loss 9.266
=== attn_manual ===
tunable: False
attn_manual  params 33.7M |   699.8 ms/step |    5,853 tok/s | peak 1951 MB | loss 9.141
=== ce16 ===
tunable: False
ce16         params 33.7M |   766.0 ms/step |    5,347 tok/s | peak 1851 MB | loss 9.141
=== aotriton ===
tunable: False
full         params 33.7M |   775.2 ms/step |    5,284 tok/s | peak 1915 MB | loss 9.141
=== full_tunable ===
full_tunable: TIMEOUT
PROBE3 DONE
```

probe4_prof.py: ayni baseline GPT adiminin torch.profiler (CPU+CUDA)
profili, 5 adim ortalamasi, self_device_time_total sirali (ilk 15 satir).
Attention math'in bmm/softmax/transpose/copy dagilimi ve autocast'in
_to_copy/_to yuku goruluyor; toplam self device suresi 2433 ms/5-adim
(launch bosluklari dahil duvar suresi 778 ms'ten buyuk — streamler
paralel degil, tablo self-time toplamidir):

```
----------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                                  Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
----------------------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                              aten::mm        10.52%     138.883ms        10.52%     138.883ms     280.573us        1.496s        12.30%        1.496s       3.023ms           495  
                                                           aten::copy_         5.61%      73.998ms         5.61%      73.998ms      63.463us     815.017ms         6.70%     815.017ms     698.985us          1166  
                                                              aten::to         2.12%      28.011ms        15.27%     201.543ms     145.204us     656.770ms         5.40%        1.861s       1.341ms          1388  
                                                         aten::reshape         3.48%      45.895ms         9.29%     122.692ms     119.118us     540.130ms         4.44%        1.140s       1.107ms          1030  
                                                             aten::bmm         4.68%      61.801ms         4.68%      61.801ms     257.504us     489.100ms         4.02%     489.100ms       2.038ms           240  
                                                       aten::transpose         3.59%      47.402ms         4.38%      57.773ms      50.457us     427.284ms         3.51%     657.798ms     574.496us          1145  
                                                        aten::_to_copy         6.12%      80.737ms        13.15%     173.532ms     178.899us     401.810ms         3.30%        1.205s       1.242ms           970  
                                                               aten::t         2.49%      32.911ms         5.33%      70.383ms      85.312us     385.186ms         3.17%     834.346ms       1.011ms           825  
                                                      aten::as_strided         1.10%      14.509ms         1.10%      14.509ms       7.705us     371.981ms         3.06%     371.981ms     197.547us          1883  
                                                             aten::mul         2.40%      31.689ms         2.40%      31.689ms     152.353us     259.214ms         2.13%     259.214ms       1.246ms           208  
                                      aten::native_layer_norm_backward         1.80%      23.701ms         2.44%      32.233ms     379.218us     244.283ms         2.01%     301.015ms       3.541ms            85  
                                                          aten::linear         2.10%      27.705ms        22.53%     297.458ms     901.389us     221.666ms         1.82%        2.553s       7.737ms           330  
                                                   aten::empty_strided         2.52%      33.319ms         2.52%      33.319ms      31.433us     202.775ms         1.67%     202.775ms     191.297us          1060  
                                               aten::native_layer_norm         1.77%      23.401ms         2.27%      29.980ms     352.700us     183.470ms         1.51%     233.544ms       2.748ms            85  

Self CPU time total: 1.320s
Self CUDA time total: 12.166s
TOTAL self device time per step: 2433.2 ms
```
