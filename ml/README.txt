BU KLASOR — hazir kod (fxgpu: tablo/pencere MLP, fxlm: dil modeli/transformer), guncel benchmark'lar, olcum kaydi ve offline kurulum arsivi.
(24 Eyl 2026 guncel. 20 Eyl temizligi: eski HIP iterasyonlari silindi, sonuclari docs\HIZ_NOTLARI.md'de.)

ICERIK
------
fxgpu\                >>> TABLO / PENCERE MLP ICIN YENIDEN KULLANILABILIR PAKET <<<
  __init__.py           FusedGatherLinear, fused_gather_linear, WindowMLP, setup_env, require_gpu
  env.py                setup_env(): TunableOp env var'lari (import torch'tan ONCE cagir); require_gpu()
  hip_kernel.py         hiprtc+ctypes harness (DLL'leri glob ile bulur; host derleyici gerekmez)
  gather_gemm.hip       fused gather+bias+GEMM+Xg kernel kaynagi (TEK kopya — kernel burada yasar)
  fused_linear.py       FusedGatherLinear autograd.Function (sekilden K/F/W turetir, F%8 kontrolu,
                        inference'da Xg yazmaz)
  model.py              WindowMLP(n_feat, window, hidden, hidden2, master=True, dropout=0)
  train_template.py     NPZ -> normalize -> fp16 GPU -> fused egitim -> rank-AUC -> model.pt
  selftest.py           fused vs referans (unfold + Fn.linear) dogruluk testi — kurulum sonrasi calistir

fxlm\                 >>> KUCUK GPT / TRANSFORMER EGITIM PAKETI (24 Eyl 2026) <<<
  __init__.py           tembel export'lar (setup_lm_env, flash_attention*, *_cross_entropy,
                        rms_norm, FlatAdamW, GPT, GPTConfig)
  env.py                setup_lm_env(tunable=False) — TunableOp varsayilan KAPALI (override)
  kernels\attention.hip flash-attention fwd+bwd HIP kernel'lari (fp16 io, fp32 softmax, fdot2)
  kernels\norm.hip      fused RMSNorm fwd/bwd (F.rms_norm'dan hizli degil — default torch)
  kernels\ce.hip        satir-basina online logsumexp CE + fp16 grad
  kernels\optim.hip     FlatAdamW kernel'lari (sumsq / final / flat update)
  attention.py          flash_attention(q,k,v) + flash_attention_qkv(qkv) — (B,T,H,Dh) fp16
  norm.py               rms_norm(x, w)
  cross_entropy.py      fused_cross_entropy + linear_cross_entropy ((N,V) logits yazilmaz)
  optim.py              FlatAdamW: fp32 master + fp16 flat param + dinamik loss scale
  model.py              GPTConfig + GPT (pre-norm, bias yok, tied emb)
  train_lm.py           egitim CLI (GPU token dizisi, cosine LR, atomic ckpt, --resume, --tunable)
  prepare_bytes.py      metin/glob/dizin -> byte-level .bin + .json (--ext, binary atlar)
  sample.py             ckpt + prompt -> metin (temperature/top-k)
  selftest.py           tum kernel parity testleri (~6 s; kurulum sonrasi calistir)
  bench_kernels.py      kernel mikrobenchmark -> bench_kernels_out.txt
  kernel_regs.py        kernel basina vgpr/scratch/LDS raporu (spill teshisi)

bench\                GUNCEL benchmark'lar (bench7-9 fxgpu'yu, bench10 fxlm'i kullanir) + olcum kayitlari
  bench7_fused_e2e.py   UCTAN UCA SAMPIYON BENCHMARK — ~1.40M or/sn; SAGLIK KONTROLU olarak kullan
                        (BNCH_FUSED=0 -> saf PyTorch baseline ~1.30M)
  bench8_2m.py          2M-parametre hiz testi (~268k or/sn)
  bench9_fp32master.py  saf fp16 vs fp32-master-agirlik hiz + kararlilik (BNCH_MODE=fp16,master,master_ls)
  bench9_out.txt        bench9 ciktisi (20 Eyl 2026)
  bench10_lm.py         GPT hiz benchmark'i: naif PyTorch vs fxlm (hiz + --converge yakinsama)
  bench10_out.txt       bench10 hiz ciktisi (24 Eyl 2026)
  bench10_converge_out.txt  bench10 yakinsama ciktisi (24 Eyl 2026)
  bench10_bigbatch_out.txt  batch-olcekleme ciktisi (T256B32, T512B16; 24 Eyl 2026)
  stable_rocm10_test.txt  ROCm 10.0.0 kararli surum test kaydi (20 Eyl 2026)

docs\
  HIZ_NOTLARI.md        TUM olculmus hiz tablolari (olcum kaydi); kurallar ana README.md'de

wheels-rocm10\          venv-rocm10'un tum wheel'lari — OFFLINE kurulum (1.2 GB, 34 wheel)
wheels-rocm10-stable\   ROCm 10.0.0 KARARLI (torch 2.13) wheel'lari (1.1 GB, 17 wheel) — yedek ortam
                        gerekirse buradan kurulur; venv'i silindi (test bitti, README Sec.3.3)

CALISAN ORTAM
-------------
Ust klasorde HAZIR kurulu venv var; onerilen cagirma:
  ..\GPU_PYTHON.cmd script.py            (venv-rocm10 python + TunableOp env var'lari)
  ..\venv-rocm10\Scripts\python.exe      (torch 2.15.0a0+rocm10.1, HIP 7.16)

Calistirma (KB kokunden):
  GPU_PYTHON.cmd ml\fxgpu\selftest.py
  GPU_PYTHON.cmd ml\bench\bench7_fused_e2e.py
  GPU_PYTHON.cmd ml\fxgpu\train_template.py --npz VERI.npz
  GPU_PYTHON.cmd ml\fxlm\selftest.py
  GPU_PYTHON.cmd ml\fxlm\train_lm.py --data data\X_train.bin --out runs\run_001
  GPU_PYTHON.cmd ml\bench\bench10_lm.py

ORTAM SIFIRDAN KURULACAKSA (offline, internetsiz) — birebir ayni surumler
------------------------------------------------------------------------
py -3.12 -m venv venv-rocm10
venv-rocm10\Scripts\python.exe -m pip install --no-index --pre ^
  --find-links "...\ml\wheels-rocm10" ^
  torch amd-torch-device-gfx1035 rocm rocm-sdk-core rocm-sdk-device-gfx1035 ^
  rocm-sdk-libraries rocm-bootstrap numpy pandas scikit-learn lightgbm ^
  onnx onnxruntime

  NOT: --pre SART — arsivdeki networkx 3.7rc0 wheel'i pre-release.
  (20 Eyl 2026'da bu komutla kurulup dogrulandi: GPU+HIP ok, fused bench ~1.4M or/sn.)

YEDEK: KARARLI ROCm 10.0.0 ORTAMI (offline, wheels-rocm10-stable'dan)
--------------------------------------------------------------------
py -3.12 -m venv venv-rocm10-stable
venv-rocm10-stable\Scripts\python.exe -m pip install --no-index ^
  --find-links "...\ml\wheels-rocm10-stable" "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy

  TUZAK: WinError 206 (yol cok uzun; LongPathsEnabled=0). Cozum:
    subst S: "<PROJE>"
    S:\venv-rocm10-stable\Scripts\python.exe -m pip install ...
    subst S: /d
  Olculmus: fused sampiyon venv-rocm10'dan %5 yavas, saf PyTorch yolu esit, hipBLASLt hala coker.

ONLINE KURULUM — PAKET INDEKSI DEGISTI (20 Eyl 2026)
----------------------------------------------------
ESKI adres https://rocm.nightlies.amd.com/whl-multi-arch/ artik LEGACY: gfx1035 icin son
wheel 20260822 (kurulu olanla ayni). Yeni surum buradan GELMEZ.

KARARLI (ROCm 10.0.0, torch 2.11/2.12/2.13):
  venv-X\Scripts\python.exe -m pip install ^
    --index-url https://stable.repo.amd.com/rocm/whl-next/ ^
    --extra-index-url https://pypi.org/simple ^
    "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy pandas scikit-learn lightgbm onnx onnxruntime

NIGHTLY (ROCm 10.2.0a; farkli surum gelebilir, once ayri venv'de dene):
  venv-X\Scripts\python.exe -m pip install --pre ^
    --index-url https://nightly.repo.amd.com/rocm/whl-next/ ^
    --extra-index-url https://pypi.org/simple ^
    "torch[device-gfx1035]" numpy

Kaynak: https://github.com/ROCm/TheRock/blob/main/RELEASES.md  (repo.amd.com yerlesimi)

KURULUM SONRASI DOGRULAMA
-------------------------
venv-X\Scripts\python.exe -c "import torch; print(torch.__version__,
torch.version.hip, torch.cuda.is_available()); print(
torch.cuda.get_device_properties(0).gcnArchName)"
-> Beklenen: True ve gfx1035
Sonra (env var'larla): python ..\GPU_SMOKE_TEST.py  ve  python fxgpu\selftest.py
Dil modeli: python fxlm\selftest.py  (SELFTEST OK bekle)
Tam pipeline: python bench\bench7_fused_e2e.py  (~1.4M bekle)
