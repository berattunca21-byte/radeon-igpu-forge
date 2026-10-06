THIS FOLDER — ready-made code (fxgpu: tabular/window MLP, fxlm: language model/transformer), current benchmarks, measurement logs and the offline install archive.
(Updated 24 Sep 2026. Cleanup on 20 Sep: old HIP iterations were deleted, their results are in docs\SPEED_NOTES.md.)

CONTENTS
--------
fxgpu\                >>> REUSABLE PACKAGE FOR TABULAR / WINDOW MLPs <<<
  __init__.py           FusedGatherLinear, fused_gather_linear, WindowMLP, setup_env, require_gpu
  env.py                setup_env(): TunableOp env vars (call BEFORE import torch); require_gpu()
  hip_kernel.py         hiprtc+ctypes harness (finds the DLLs via glob; no host compiler needed)
  gather_gemm.hip       fused gather+bias+GEMM+Xg kernel source (SINGLE copy — the kernel lives here)
  fused_linear.py       FusedGatherLinear autograd.Function (derives K/F/W from the shape, F%8 check,
                        does not write Xg in inference)
  model.py              WindowMLP(n_feat, window, hidden, hidden2, master=True, dropout=0)
  train_template.py     NPZ -> normalize -> fp16 GPU -> fused training -> rank-AUC -> model.pt
  selftest.py           fused vs reference (unfold + Fn.linear) correctness test — run after install

fxlm\                 >>> SMALL GPT / TRANSFORMER TRAINING PACKAGE (24 Sep 2026) <<<
  __init__.py           lazy exports (setup_lm_env, flash_attention*, *_cross_entropy,
                        rms_norm, FlatAdamW, GPT, GPTConfig)
  env.py                setup_lm_env(tunable=False) — TunableOp OFF by default (override)
  kernels\attention.hip flash-attention fwd+bwd HIP kernels (fp16 io, fp32 softmax, fdot2)
  kernels\norm.hip      fused RMSNorm fwd/bwd (not faster than F.rms_norm — default is torch)
  kernels\ce.hip        per-row online logsumexp CE + fp16 grad
  kernels\optim.hip     FlatAdamW kernels (sumsq / final / flat update)
  attention.py          flash_attention(q,k,v) + flash_attention_qkv(qkv) — (B,T,H,Dh) fp16
  norm.py               rms_norm(x, w)
  cross_entropy.py      fused_cross_entropy + linear_cross_entropy ((N,V) logits are not written)
  optim.py              FlatAdamW: fp32 master + fp16 flat param + dynamic loss scale
  model.py              GPTConfig + GPT (pre-norm, no bias, tied emb)
  train_lm.py           training CLI (GPU token sequence, cosine LR, atomic ckpt, --resume, --tunable)
  prepare_bytes.py      text/glob/directory -> byte-level .bin + .json (--ext, skips binaries)
  sample.py             ckpt + prompt -> text (temperature/top-k)
  selftest.py           all kernel parity tests (~6 s; run after install)
  bench_kernels.py      kernel microbenchmark -> bench_kernels_out.txt
  kernel_regs.py        per-kernel vgpr/scratch/LDS report (spill diagnosis)

bench\                CURRENT benchmarks (bench7-9 use fxgpu, bench10 uses fxlm) + measurement records
  bench7_fused_e2e.py   END-TO-END CHAMPION BENCHMARK — ~1.40M samples/s; use it as the HEALTH CHECK
                        (BNCH_FUSED=0 -> pure PyTorch baseline ~1.30M)
  bench8_2m.py          2M-parameter speed test (~268k samples/s)
  bench9_fp32master.py  pure fp16 vs fp32-master-weights speed + stability (BNCH_MODE=fp16,master,master_ls)
  bench9_out.txt        bench9 output (20 Sep 2026)
  bench10_lm.py         GPT speed benchmark: naive PyTorch vs fxlm (speed + --converge convergence)
  bench10_out.txt       bench10 speed output (24 Sep 2026)
  bench10_converge_out.txt  bench10 convergence output (24 Sep 2026)
  bench10_bigbatch_out.txt  batch-scaling output (T256B32, T512B16; 24 Sep 2026)
  stable_rocm10_test.txt  ROCm 10.0.0 stable test record (20 Sep 2026)

docs\
  SPEED_NOTES.md        ALL measured speed tables (measurement log); the rules are in the main README.md

wheels-rocm10\          all wheels of venv-rocm10 — OFFLINE install (1.2 GB, 34 wheels)   [local only, not in the repo]
wheels-rocm10-stable\   ROCm 10.0.0 STABLE (torch 2.13) wheels (1.1 GB, 17 wheels) — if a backup environment
                        is needed it installs from here; its venv was deleted (test finished, README Sec.3.3)   [local only]

WORKING ENVIRONMENT
-------------------
The parent folder holds a READY installed venv; recommended invocation:
  ..\GPU_PYTHON.cmd script.py            (venv-rocm10 python + TunableOp env vars)
  ..\venv-rocm10\Scripts\python.exe      (torch 2.15.0a0+rocm10.1, HIP 7.16)

Running (from the repo root):
  GPU_PYTHON.cmd ml\fxgpu\selftest.py
  GPU_PYTHON.cmd ml\bench\bench7_fused_e2e.py
  GPU_PYTHON.cmd ml\fxgpu\train_template.py --npz DATA.npz
  GPU_PYTHON.cmd ml\fxlm\selftest.py
  GPU_PYTHON.cmd ml\fxlm\train_lm.py --data data\X_train.bin --out runs\run_001
  GPU_PYTHON.cmd ml\bench\bench10_lm.py

SETTING THE ENVIRONMENT UP FROM SCRATCH (offline, no internet) — identical versions
-----------------------------------------------------------------------------------
py -3.12 -m venv venv-rocm10
venv-rocm10\Scripts\python.exe -m pip install --no-index --pre ^
  --find-links "...\ml\wheels-rocm10" ^
  torch amd-torch-device-gfx1035 rocm rocm-sdk-core rocm-sdk-device-gfx1035 ^
  rocm-sdk-libraries rocm-bootstrap numpy pandas scikit-learn lightgbm ^
  onnx onnxruntime

  NOTE: --pre is REQUIRED — the networkx 3.7rc0 wheel in the archive is a pre-release.
  (Installed and verified with this command on 20 Sep 2026: GPU+HIP ok, fused bench ~1.4M samples/s.)

BACKUP: STABLE ROCm 10.0.0 ENVIRONMENT (offline, from wheels-rocm10-stable)
---------------------------------------------------------------------------
py -3.12 -m venv venv-rocm10-stable
venv-rocm10-stable\Scripts\python.exe -m pip install --no-index ^
  --find-links "...\ml\wheels-rocm10-stable" "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy

  PITFALL: WinError 206 (path too long; LongPathsEnabled=0). Fix:
    subst S: "<PROJECT>"
    S:\venv-rocm10-stable\Scripts\python.exe -m pip install ...
    subst S: /d
  Measured: fused champion is 5% slower than venv-rocm10, the pure PyTorch path is equal, hipBLASLt still crashes.

ONLINE INSTALL — PACKAGE INDEX CHANGED (20 Sep 2026)
----------------------------------------------------
The OLD address https://rocm.nightlies.amd.com/whl-multi-arch/ is now LEGACY: the last wheel for
gfx1035 is 20260822 (same as the installed one). New versions will NOT come from there.

STABLE (ROCm 10.0.0, torch 2.11/2.12/2.13):
  venv-X\Scripts\python.exe -m pip install ^
    --index-url https://stable.repo.amd.com/rocm/whl-next/ ^
    --extra-index-url https://pypi.org/simple ^
    "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy pandas scikit-learn lightgbm onnx onnxruntime

NIGHTLY (ROCm 10.2.0a; a different version may come, try it in a separate venv first):
  venv-X\Scripts\python.exe -m pip install --pre ^
    --index-url https://nightly.repo.amd.com/rocm/whl-next/ ^
    --extra-index-url https://pypi.org/simple ^
    "torch[device-gfx1035]" numpy

Source: https://github.com/ROCm/TheRock/blob/main/RELEASES.md  (repo.amd.com layout)

VERIFICATION AFTER INSTALL
--------------------------
venv-X\Scripts\python.exe -c "import torch; print(torch.__version__,
torch.version.hip, torch.cuda.is_available()); print(
torch.cuda.get_device_properties(0).gcnArchName)"
-> Expected: True and gfx1035
Then (with the env vars): python ..\GPU_SMOKE_TEST.py  and  python fxgpu\selftest.py
Language model: python fxlm\selftest.py  (expect SELFTEST OK)
Full pipeline: python bench\bench7_fused_e2e.py  (expect ~1.4M)
