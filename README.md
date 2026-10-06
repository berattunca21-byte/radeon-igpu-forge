# Radeon Forge — Maximum-Speed AI Training on the Radeon 680M (gfx1035) iGPU

> Measured speed recipes for PyTorch training on an integrated AMD GPU with ROCm/HIP, hand-written HIP kernels (fused linear, flash attention, fused CE, FlatAdamW), and a record of the working setup.

Last verified: **27 Sep 2026** (RAM moved to dual channel, Sec.2 updated; the fxlm language-model package was added and measured on 24 Sep 2026; the tabular numbers were re-measured on 20 Sep 2026).
Scope: **PyTorch training on this machine** — (1) tabular / time-series window models (`ml\fxgpu\`), (2) small language model / transformer (`ml\fxlm\`, Sec.12). Game/image CNN project notes were removed from this knowledge base (local archive, not in the repo); general CNN lessons for gfx1035 are in Sec.9.

> In one sentence: ROCm/HIP PyTorch training **works** on the Radeon 680M in this laptop; the fastest
> measured path is pure fp16 + TunableOp (rocBLAS) + a hand-written HIP fused L1 layer = **~1.40M samples/s**
> (3.1x FP32 eager). Ready-made code is in the `ml\fxgpu\` package, the launcher is `GPU_PYTHON.cmd`.
> For language models / transformers, `ml\fxlm\` (hand-written flash attention + fused CE + FlatAdamW)
> is **1.9–2.7x** faster per step than a naive PyTorch GPT with the same learning quality (Sec.12).

---

## 1. Mandatory rules (for humans and AI alike)

1. **Right interpreter:** `<PROJECT>\venv-rocm10\Scripts\python.exe`
   The system `python` command has **no GPU support**; never use it for training. Easiest: `GPU_PYTHON.cmd script.py ...`
   in the repo root — it supplies the right python **and** the required TunableOp env vars together.
2. The `torch.cuda.*` namespace is normal on Radeon; this install is **ROCm/HIP**, not CUDA. There is no `nvidia-smi`.
3. If `torch.cuda.is_available()` is False, do not silently fall back to CPU — fix the environment (Sec.9).
4. **Do not delete, upgrade, or `pip install` into** the working venv (`venv-rocm10`). Create a separate venv for experiments.
5. Radeon 680M = **`gfx1035`**. Do not install packages for other GFX targets; do not set `HSA_OVERRIDE_GFX_VERSION`.
6. In training, **follow the speed recipe** (Sec.5): no DataLoader, no Conv1d, batch 16384–32768, pure fp16,
   no `loss.item()` inside the batch loop, an `isfinite` check. (Tabular recipe; language-model rules are in Sec.12.)
7. `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` is **required** — the hipBLASLt path crashes on gfx1035 because
   its kernel files are missing (Sec.5.3).
8. Every new training run writes to a separate output file; never delete an old model/checkpoint without the user asking.
9. Run only **one** GPU job at a time. The iGPU shares system RAM; a second process ruins both speed and measurements.
10. On NaN/Inf, training stops, the last good checkpoint is kept, and you lower lr / switch to fp32 master (Sec.6).

## 2. Hardware facts (read on this machine, 20 Sep 2026)

| Component | Value |
|---|---|
| Machine | Acer Aspire **AG15-41P** (laptop) |
| CPU | AMD **Ryzen 7 7735HS** — 8 cores / 16 threads, 3.2 GHz base |
| GPU | AMD Radeon 680M (`AMD Radeon(TM) Graphics`), **gfx1035**, RDNA2, **12 CU** (torch `multi_processor_count=6` = WGP count) |
| Memory visible to GPU | **7.14 GB** (shared system RAM; Windows `AdapterRAM=2 GB` is misleading) |
| RAM | **2x 16 GB = 32 GB DDR5-5600 SODIMM** — Hynix `HMCG78AGBSA092N` (`P0 CHANNEL A`) + Micron `MTC8C1084S1SC56BD1` (`P0 CHANNEL B`), running at **4800 MT/s**. **Second SODIMM installed on 27 Sep 2026** |
| Free RAM slots | **NONE — both slots are populated** (Channel A + B), motherboard max **64 GB** |
| Memory channels | **DUAL CHANNEL ACTIVE (27 Sep 2026)** — the old record was "single channel ≈ 38 GB/s theoretical, main bottleneck"; bandwidth is expected ~2x. The effect on training speed is to be confirmed in the first bench7 run |
| Driver | AMD `32.0.21045.9003` (27 Aug 2026) — updated from the old record 21043.19003 |
| OS | Windows 11 Pro build 26200 |
| Power plan | **"Ultimate Performance"** active (verified). The laptop must be plugged in. |
| Total RAM | **29.75 GB physical / ~27.7 GB visible** (27 Sep 2026; the old record was 14.7/13.7 GB) |

**Hardware recommendation — DONE (27 Sep 2026):** the second SODIMM (Hynix DDR5-5600 16 GB) was
installed in the free slot → **dual channel active**, bandwidth expected ~2x. iGPU training (especially
the gather/unfold and dW steps) was bandwidth-bound; this was a jump software alone could not deliver. Earlier
measurements (including the ~1.40M samples/s champion in Sec.5) were taken on a SINGLE CHANNEL — the dual-channel
gain should be re-measured with bench7 and noted (Sec.5 gets updated when the measurement arrives).

## 3. Software environment

### 3.1 Primary environment — `venv-rocm10` (in this folder, verified)

| Component | Value |
|---|---|
| Python | 3.12.10 64-bit |
| torch | `2.15.0a0+rocm10.1.0a20260822` (+ `amd-torch-device-gfx1035`) |
| HIP | `7.16.26332` |
| ROCm SDK | `10.1.0a20260822` (core / device-gfx1035 / libraries) |
| Extras | numpy 2.4.4, pandas 3.0.6, scikit-learn 1.9.1, lightgbm 4.7.0, onnx 1.23.0, onnxruntime 1.30.0 |
| GPU | `is_available()=True`, `gcnArchName=gfx1035` |
| Offline backup | `ml\wheels-rocm10\` (34 wheels, 1.2 GB) — offline reinstall in `ml\README.txt` (local copy only, not in the repo) |

### 3.2 Backup environment — `<OLD_VENV_THEROCK>`

torch `2.12.0+rocm7.14.0a20260519`, HIP 7.13. It benchmarked 3–7% faster than venv-rocm10
(fp32 455k vs 424k); the user still chose the newer build as primary. You can go
back if problems appear. Do not touch it.

### 3.3 Backup — ROCm 10.0.0 STABLE (`ml\wheels-rocm10-stable\`, venv not installed)

Installed into a separate venv on 20 Sep 2026 and tested (record: `ml\bench\stable_rocm10_test.txt`); after the decision
the venv was deleted; **the wheel archive stays** — if needed it installs offline in 2 minutes (`ml\README.txt`). venv-rocm10 was not touched.

| Component | Value |
|---|---|
| torch | `2.13.0+rocm10.0.0` (+ `amd-torch-device-gfx1035 2.13.0+rocm10.0.0`) |
| HIP | `7.15.26333` |
| ROCm SDK | `10.0.0` (core / device-gfx1035 / libraries), rocm-bootstrap 0.2.0, numpy 2.5.3 |
| GPU | `True`, `gfx1035`, 7.14 GB — `GPU_SMOKE_TEST` passed, hiprtc harness passed, `selftest` passed |
| DLL names | `amdhip64_7.dll`, **`hiprtc0715.dll`** (`hiprtc0716.dll` on the 10.1 nightly) → `hip_kernel.py` now finds them via glob |
| Offline archive | `ml\wheels-rocm10-stable\` (17 wheels, 1.1 GB) |

Same-session A/B (`bench7_fused_e2e.py`, 3 runs):

| Measurement | venv-rocm10 (torch 2.15 / HIP 7.16) | venv-rocm10-stable (torch 2.13 / ROCm 10.0.0) |
|---|---|---|
| **fused champion** (HIP L1) | **1,442,474 / 1,430,773 / 1,421,515** | 1,342,144 / 1,355,358 / 1,366,183 (**−5%**) |
| pure PyTorch baseline (`BNCH_FUSED=0`, fp16+TunableOp) | 1,304,097 / 1,301,029 / 1,308,710 | 1,305,473 / 1,309,739 / 1,311,824 (equal) |
| hipBLASLt on (`HIPBLASLT_ENABLED=1`) | crashes | **still crashes** (`TensileLibrary_lazy_gfx1035.dat` missing → `HIPBLAS_STATUS_INVALID_VALUE`) |
| selftest fwd err | 0.000000 | 0.001953 (fp16 rounding; grad err 0) |

**Decision:** **venv-rocm10 stays primary** (5% ahead on the fused path, all packages installed). The stable wheel archive
is a pinned / reproducible **backup**; if the nightly ever breaks, install from it. Since the kernel is identical,
the difference must come from the rocBLAS `mm` in backward.

**Install pitfall (important):** `pip install` → `WinError 206` (path too long). The stable wheel's `licenses/`
tree is ~230 characters; `LongPathsEnabled=0` and no admin rights. Fix: `subst S: "<PROJECT>"`
→ `S:\venv-rocm10-stable\Scripts\python.exe -m pip install ...` → `subst S: /d`. (Or, with admin,
`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1`.)

### 3.4 Package sources — **URL layout changed (detected 20 Sep 2026)**

AMD moved the TheRock package indexes under `repo.amd.com`. The old address is now **legacy**:

| Channel | Address | gfx1035 Windows status |
|---|---|---|
| **Stable** | `https://stable.repo.amd.com/rocm/whl-next/` | **ROCm 10.0.0** — torch 2.11.0 / 2.12.0 / 2.13.0 (+torchvision), cp310–cp314 win_amd64. The **first stable release** for gfx1035. |
| Nightly | `https://nightly.repo.amd.com/rocm/whl-next/` | ROCm 10.2.0a nightly; torch 2.12/2.13/2.14/2.15a0; latest 20260920 |
| Legacy nightly (the old one) | `https://rocm.nightlies.amd.com/whl-multi-arch/` | **Frozen** — last wheel for gfx1035 is 20260822 (same as the installed one). No new releases. |

Notes:
- `SUPPORTED_GPUS.md` (TheRock): gfx1035 Windows = Build ✅ / Sanity Tested ✅ / **Release Ready ✅**.
- **triton** is **linux_x86_64** only on both indexes → still no `torch.compile(inductor)` on Windows.
- Install command (new venv, stable):
  ```bat
  py -3.12 -m venv venv-X
  venv-X\Scripts\python.exe -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ --extra-index-url https://pypi.org/simple "torch[device-gfx1035]==2.13.0+rocm10.0.0" numpy pandas scikit-learn lightgbm
  ```
  For nightly use `--pre` and `nightly.repo.amd.com`. After installing, **always** run `GPU_SMOKE_TEST.py` + `ml\fxgpu\selftest.py`.

Official sources: `https://github.com/ROCm/TheRock/blob/main/RELEASES.md`,
`https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md`,
`https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md`.

## 4. At the start of every session

```bat
cd <PROJECT>
GPU_PYTHON.cmd -c "import torch;print(torch.__version__,torch.version.hip,torch.cuda.is_available(),torch.cuda.get_device_properties(0).gcnArchName,torch.cuda.tunable.is_enabled())"
```
Expected: `... True gfx1035 True`. If in doubt: `GPU_PYTHON.cmd GPU_SMOKE_TEST.py` (fp32 CNN smoke) and
`GPU_PYTHON.cmd ml\fxgpu\selftest.py` (HIP fused kernel correctness). Full pipeline health:
`GPU_PYTHON.cmd ml\bench\bench7_fused_e2e.py` → 3 runs **~1.39–1.44M samples/s**, `err=0.0000`.
Language-model package: `GPU_PYTHON.cmd ml\fxlm\selftest.py` → `SELFTEST OK` (~6 s).

`GPU_PYTHON.cmd` sets: `PYTORCH_TUNABLEOP_ENABLED=1`, `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`,
`PYTORCH_TUNABLEOP_MAX_TUNING_DURATION_MS=200`, `HIP_VISIBLE_DEVICES=0`; then it calls the `venv-rocm10` python.

## 5. SPEED RECIPE — the measured ladder (500k x 40 features, window W=30, bs=32768)

| Step | samples/s | Gain | Note |
|---|---|---|---|
| FP32 eager, DataLoader-free GPU-index recipe | ~455k | 1.0x | old baseline (already 10–50x faster than typical AI scripts) |
| + pure fp16 (`model.half()`, `X.half()`, loss `.float()`) | ~900–930k | 2.0x | RDNA2 packed fp16 |
| + TunableOp (rocBLAS path, Lt off) | ~1.24–1.28M | 2.7x | +38% on fp16 GEMM |
| + remove dropout, NN layout (pure PyTorch ceiling) | ~1.24M (early Sep) → **~1.30M** (20 Sep same session, `BNCH_FUSED=0`) | 2.9x | dropout costs ~10% |
| **+ hand-written HIP fused gather+GEMM L1** (`FusedGatherLinear`) | **~1.40–1.44M** | **3.1x** | +8–10% over pure PyTorch; grads are bit-for-bit equal |
| + fp32 master weights (for stability, Sec.6) | ~1.41M | 3.1x | costs ~1% |
| 2M-parameter model (1200→1024→768→1), same path | ~268k | — | ~1.84 s/epoch (500k rows) |

20 Sep 2026 re-measurements (`bench7_fused_e2e.py`, 3 runs the same day): 1,400,028 / 1,408,611 / 1,395,568 →
1,407,791 / 1,398,414 / 1,393,630 → 1,442,474 / 1,430,773 / 1,421,515 samples/s; `fwd/dW/db err = 0.0000` in all of them.
+-2% run-to-run variance is normal (TunableOp tuning + thermals).

### 5.1 The recipe's rules (each one measured on this hardware)

1. **NO DataLoader.** All X/y tensors live on the GPU from the start; batch = `X[rows]` GPU indexing. On Windows
   DataLoader+workers is slow/hangs.
2. **Window = `X.unfold(0, W, 1)` view** (no copy) or better the **fused kernel** (the window is
   never materialized).
3. **Conv1d is FORBIDDEN** (gfx1035 MIOpen generic → ~50x slower). A windowed model = `wflat`: `(B,W,C)→(B,W·C)→MLP`.
4. **Big batch:** 16384–32768. 65536 measured slower. A small batch (2048) is 4.6x slower.
5. **Pure fp16**, not autocast/AMP. Loss in fp32: `lossf(out.float(), y)`. `clip_grad_norm_(5.0)`.
6. **No syncs:** no `loss.item()`/`float()` inside the batch loop; `loss_sum += loss.detach()` accumulates on the GPU,
   `synchronize()` only at epoch end. The `isfinite` check is sparse (e.g. once every 50 batches).
7. `torch.set_float32_matmul_precision("high")`, `AdamW(fused=True)`, `randperm`/`randint` on device.
8. Eval with `torch.inference_mode()` + bs 32768; AUC is computed on the GPU via ranks (`train_template.py`).
9. **weight_decay instead of dropout** (dropout costs ~10% speed).

### 5.2 TunableOp — correct usage

- The env vars must be set **before** `import torch` → `GPU_PYTHON.cmd` or `fxgpu.setup_env()`.
- `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` is required: the default hipBLASLt path crashes on gfx1035 with
  `TensileLibrary_lazy_gfx1035.dat / Kernels.so-000-gfx1035.hsaco file not found` (the package has no Lt
  kernel files for this arch). The rocBLAS path works and gives ~38% on fp16 GEMM.
- Tuning happens in the first iterations → leave a warm-up loop (15 steps is enough). `MAX_TUNING_DURATION_MS=150–200` is stable.
- **Do not reuse the CSV** (`TUNING=0` + a saved csv dropped to ~933k; solution ids map to a different
  kernel). Fresh tuning every run.
- Each run writes `tunableop_results0.csv` to the working directory (TunableOp default). It is harmless and
  can be deleted; with tuning on it reuses results for the same shapes (confirmed in bench7 at 1.40M). If speed
  drops inexplicably, delete this file first.

### 5.3 Fused HIP kernel (`ml\fxgpu\gather_gemm.hip`)

`out[b,n] = bias[n] + Σ_{w,f} X16[idx[b]+w, f] · W1[n, w·F+f]` — gather + bias + GEMM in one kernel;
it also writes an `Xg` (B,K) fp16 copy so that `dW = gᵀ·Xg` in backward is fast with rocBLAS (a fused dW
was tried: correct but 14.4 ms vs rocBLAS 8.5 ms → not used). It is **compiled at runtime** with hiprtc;
**no host compiler** (hipcc/MSVC) is needed. Technique: k-pair-packed column-major LDS + interleaved lane
mapping (bank-conflict-free), `v_dot2` (`__builtin_amdgcn_fdot2`) fp32 accumulation, 128x128x32 tile,
8x8 thread tile, double buffering. Isolated GEMM 3.5 TF/s (rocBLAS 3.19).

Constraints: `F % 8 == 0` (16 B aligned float4 loads; pad the feature count to a multiple of 8 with zero columns —
`train_template.py` does this automatically), `idx` is int32 and `idx + W - 1 < R`, all tensors
contiguous fp16, single GPU.

## 6. Stability: the fp16 risk and the fp32 master-weight recipe

- fp16 AdamW diverged in ~10–15 steps on random labels (|w| 0.09 → 9.4 in the first step). The kernel is clean
  (GEMM rel err 3.4e-4). On real data: use **lr warmup + fp32 master weights**.
- **fp32 master** = parameters in fp32 (AdamW in fp32), fp16 compute via a `.half()` cast each step;
  autograd flows the grads back into the fp32 master. `WindowMLP(master=True)` does this.

**Measurement (20 Sep 2026, `ml\bench\bench9_fp32master.py`, output `ml\bench\bench9_out.txt`):** same model/data,
learnable label (`y = X[t,0] + 0.5·X[t-1,1] > 0`), lr 1e-3, 300 steps:

| Mode | samples/s | vs fp16 | After 300 steps |
|---|---|---|---|
| pure fp16 (AdamW fp16) | 1,422,089 | 100% | **NaN — diverged before step 50** |
| **fp32 master** (AdamW fp32, `.half()` cast) | 1,406,516 | **98.9%** | loss 0.246 → 0.0012, finite, \|w\|max 0.37 |
| fp32 master + static loss scale 256 | 1,393,445 | 98.0% | loss 0.246 → 0.0010 |

**Conclusion: fp32 master weights cost ~1% speed and end the divergence → this is the default** (`WindowMLP(master=True)`,
the `train_template.py` default). Pure fp16 AdamW is not used in real training. A static loss scale costs another
~1%; use it only when grad underflow is suspected (`--loss-scale 256`). `train_template.py --no-master` also diverged on
synthetic data (loss 0.71 → 1.88) — consistent with the fp16 warning above.

Rule: watch `isfinite` in the first real training; on NaN, in order: (1) lower lr / extend warmup,
(2) `--loss-scale 256` (static), (3) go back to fp32 (not `WindowMLP(master=False)` — the fully fp32 path,
half the speed of Sec.5).

## 7. The `ml\fxgpu\` package — starting point for tabular/window MLPs

```text
ml\fxgpu\
  __init__.py        FusedGatherLinear, fused_gather_linear, WindowMLP, setup_env, require_gpu
  env.py             setup_env() -> TunableOp env vars (call BEFORE import torch); require_gpu()
  hip_kernel.py      hiprtc + ctypes harness (finds the DLLs via glob: amdhip64*.dll, hiprtc*.dll)
  gather_gemm.hip    fused kernel source (single copy — the kernel lives here)
  fused_linear.py    FusedGatherLinear autograd.Function — K/F/W derived from shape, F%8 check,
                     does not write Xg in inference_mode (15% faster eval)
  model.py           WindowMLP(n_feat, window, hidden, hidden2, master=True): L1 fused, L2/L3 Fn.linear
  train_template.py  NPZ -> normalize -> fp16 GPU -> fused training -> rank-AUC eval -> model.pt
  selftest.py        fused vs reference (unfold + Fn.linear) correctness test
ml\bench\
  bench7_fused_e2e.py   champion end-to-end benchmark = health check (BNCH_FUSED=0 → pure PyTorch baseline)
  bench8_2m.py          2M-parameter model speed test
  bench9_fp32master.py  pure fp16 vs fp32 master speed + stability
  bench10_lm.py         GPT speed + convergence benchmark (fxlm, Sec.12) + bench10*_out.txt records
  bench9_out.txt, stable_rocm10_test.txt   measurement records
```
bench7–9 import `fxgpu` — if the package breaks, bench7 breaks too (intended: single source of truth).

Usage:

```python
from fxgpu import setup_env; setup_env()      # BEFORE import torch
import torch
from fxgpu import require_gpu, WindowMLP
dev = require_gpu()
X16 = X_norm.half().to(dev)                    # (R, F) F%8==0
model = WindowMLP(n_feat=F, window=30, master=True).to(dev)
idx32 = torch.randint(0, R - 30 + 1, (32768,), device=dev, dtype=torch.int32)
out = model(X16, idx32)                        # (B,1) fp16 ; label = y[idx32 + 29]
```

Ready-made script:

```bat
GPU_PYTHON.cmd ml\fxgpu\train_template.py --npz DATA.npz --window 30 --epochs 5 --batch 32768
```
NPZ contract: `X (n,f) float`, `y (n,) 0/1` (y[t] must be the **future** target for the window ending at row t —
leakage is the data preparer's responsibility), optional `ts`, `cols`. The last 20% is the test set
(time-ordered), normalization uses train statistics only. Defaults: fp32 master on, warmup
automatic (`min(200, max(10, total_steps/10))`), weight_decay 1e-4, dropout 0, `--check-every 50`.

**Reference run (20 Sep 2026, synthetic 500k x 40, learnable label):**

```text
bs=32768 --epochs 5 : warmup=10 (auto) | 13 steps/epoch, 65 total steps
  ep1 auc=0.7016 [1.5s]  ep2 0.8551 [0.3s 1,333,698/s]  ep3 0.9095  ep4 0.9428  ep5 0.9601 [0.3s 1,328,026/s]
bs=8192  --epochs 3 : warmup=14 (auto) | 49 steps/epoch, 147 total steps
  ep1 auc=0.9188 [1.6s]  ep2 0.9720 [0.5s 852,437/s]  ep3 0.9824 [0.5s 867,305/s]
```

**Lesson — samples/s is not the only metric:** bs=32768 gives the highest throughput but makes only 13 updates
per epoch on 400k rows; bs=8192 streams 35% slower but with 4x the updates it reaches a **higher AUC in less
wall-clock time**. If the data is < ~2M rows, start with **bs 8192–16384**; move up to 32768 as the data grows.
The first epoch is ~1.5 s longer in every run (hiprtc compile + TunableOp tuning) — normal.

## 8. Tried and REJECTED (don't retry — waste of time)

| Attempt | Result |
|---|---|
| `torch.compile` backend=`cudagraphs` | ~352k (Dynamo overhead) |
| `torch.compile` inductor | does not work — no triton on Windows (not in stable/nightly indexes either) |
| CUDA/HIP Graph (static buffer) | +5% in fp32, no gain in fp16 (~925k), full-step graph ~859k |
| bs=65536 | ~1.15M (slower than 32768) |
| Materialized lagged matrix `Xf` | ~881k (slower) |
| `fp16_accum=1` | no difference |
| Triton fused gather+GEMM | `tl.dot` falls back to the FMA path on gfx1035 — slow |
| Reusing the TunableOp CSV | ~933k (broken) |
| Fused dW HIP kernel | correct but 14.4 ms vs rocBLAS 8.5 ms |
| Conv1d | ~50x slower |
| DataLoader + num_workers | slow / hangs on Windows |
| LightGBM GPU (OpenCL) | not faster than 16-thread CPU → GBM stays on CPU |
| hipBLASLt (TunableOp default) | crashes on gfx1035 |
| hipcc host compile toolchain (amdclang/msvc-sdk/xwin, 2.4 GB) | unnecessary — hiprtc runtime compile is enough; deleted 20 Sep 2026 |
| SDPA flash / mem-efficient backend (gfx1035) | shows "available" but falls back to math with "No available kernel" — the hand-written flash kernel is 4.5–5.8x faster (fxlm) |
| Fused RMSNorm HIP kernel (fxlm) | not faster than F.rms_norm (0.84–0.96x, 8192x768 fwd+bwd 3.5 vs 3.4 ms) → default `norm_impl="torch"` |

## 9. Common problems

**`torch.cuda.is_available() == False`** → in order: is it the right python (`venv-rocm10`)? Is the AMD driver
enabled in Device Manager? Is `pip show amd-torch-device-gfx1035` present? Is `torch.version.hip` empty?
Reboot. Still nothing: reinstall the environment from `ml\wheels-rocm10` (`ml\README.txt`) or from the online indexes (Sec.3.4).

**TunableOp `...hsaco file not found` / `TensileLibrary_lazy_gfx1035.dat`** → `HIPBLASLT_ENABLED=0` was forgotten.

**`hiprtcCompileProgram failed`** → read the `[hiprtc log]` line; usually the include path
(`_rocm_sdk_core\include`) or `--offload-arch=gfx1035`. If the DLL name changed, `fxgpu\hip_kernel.py`
finds it via glob; if not found it lists the bin folder.

**Out of memory** → halve the batch; close browser/games; one training job; shard the data (500k x 40 fp16 = 40 MB,
Xg buffer 32768x1200 fp16 = 78 MB — usually not a problem).

**Speed far below 1.4M** → (1) are the env vars set (`torch.cuda.tunable.is_enabled()`)? (2) is another
GPU process running? (3) is the laptop plugged in with "Ultimate Performance"? (4) was a warm-up done? (5) F%8 may have
fallen to the slow path (pad F).

**MIOpen `CK grouped conv library not found`** → a warning, conv still works (there is no conv in our path).

**PC froze / screen hung (long GPU job)** → since the iGPU also drives the display, long and unbounded GPU
experiments (especially the first TunableOp tuning) can lock up the system; one freeze happened on
24 Sep 2026 and needed a reset. Run experiments first with small shapes, a timeout, and output written to a file;
keep a single kernel call under ~0.5 s (Windows TDR ~2 s).

**Image/CNN models (game/image CNN experience, gfx1035 Windows)** → `BatchNorm2d` hung on some
setups → prefer GroupNorm; first MIOpen kernel preparation can take 30–60 s (not a freeze);
keep FP32 default for CNNs (autocast/pure fp16 not verified on image models); `channels_last` off;
`num_workers=0`; start with batch 4–8.

## 10. Method notes for tabular / window models

- **Tabular features + GBM (LightGBM, 16-thread CPU)** is better than or equal to NNs on most tabular data and
  takes seconds — **build it first as the baseline**. Use the GPU NN for window/sequence models.
- On time-ordered data a **time-ordered split** is mandatory (last 20% test); a random split leaks. Normalize with train statistics only.
- Metric: AUC + top-10% confidence accuracy (`train_template.py` writes both); accuracy alone is misleading.
- Normalize features for fp16 (z-score); raw values overflow in fp16.
- Small model + weight_decay > big model + dropout (better for both speed and generalization in this data regime).

## 11. Decision tree and checklist for tabular training

```text
GPU_PYTHON.cmd shows True + gfx1035 + tunable True?     No -> Sec.9
selftest.py passed (fused == reference)?                No -> environment/kernel; no training
Data: time-ordered split, no leakage?                   No -> fix the data
Baseline LightGBM AUC obtained?                         No -> get it first
NN: train_template --epochs 1 speed ~1M+/s and loss finite?  No -> Sec.9 speed / Sec.6 NaN
Long training -> test AUC beats the baseline?           No -> model/features; stop fiddling with speed
```

- [ ] I use `GPU_PYTHON.cmd` (right python + env vars).
- [ ] I did not touch `venv-rocm10`.
- [ ] Single GPU process; laptop plugged in; "Ultimate Performance".
- [ ] Pure fp16 + fp32 master (`master=True`), warmup on, sparse `isfinite` check.
- [ ] No DataLoader/Conv1d/dropout; bs 16384–32768.
- [ ] F % 8 == 0 (padded), `idx + W - 1 < R`.
- [ ] Time-ordered split, train-statistics normalization.
- [ ] Separate output file; old model not deleted.

## 12. Language model / transformer training — `ml\fxlm\` (24 Sep 2026)

A small GPT-style language-model training package; the kernels are hand-written HIP compiled at runtime
with hiprtc (no host compiler). **1.6–2.7x faster** than a naive PyTorch GPT baseline (table below).

**Contents** (`ml\fxlm\`):

| File | Job |
|---|---|
| `kernels\attention.hip` | flash-attention fwd + bwd (prep/dkv/dq), fp16 io, fp32 softmax, fdot2 |
| `kernels\norm.hip` | fused RMSNorm fwd/bwd (not faster than F.rms_norm — torch is the default) |
| `kernels\ce.hip` | per-row online logsumexp CE + fp16 grad write |
| `kernels\optim.hip` | FlatAdamW: sumsq+nonfinite, final, flat update |
| `attention.py` | `flash_attention(q,k,v)` and `flash_attention_qkv(qkv)` — (B,T,H,Dh) fp16, strided views without copies |
| `norm.py` | `rms_norm(x, w)` |
| `cross_entropy.py` | `fused_cross_entropy`, `linear_cross_entropy` (the full (N,V) logits are never written; grad_scale is a float or a 0-dim CUDA tensor) |
| `optim.py` | `FlatAdamW` — fp32 master + fp16 flat param + dynamic loss scale + global clip, no host sync |
| `model.py` | `GPTConfig` + `GPT` (pre-norm, no bias, tied embedding by default) |
| `train_lm.py` | training CLI — token sequence on the GPU, device randint batches, cosine LR, atomic ckpt, --resume |
| `prepare_bytes.py` | text/glob/directory → byte-level uint16 .bin + .json (--ext directory filter, binaries containing NUL are skipped) |
| `sample.py` | ckpt + prompt → text generation (temperature/top-k) |
| `selftest.py` | PyTorch parity test of all kernels (~6 s) |
| `bench_kernels.py` | kernel microbenchmarks → `bench_kernels_out.txt` |
| `kernel_regs.py` | per-kernel vgpr/scratch/LDS (ELF .kd parse) — spill diagnosis |

**fxlm-specific rules:**

- TunableOp is **off by default** (`setup_lm_env` overrides GPU_PYTHON.cmd's =1);
  turn it on with `train_lm.py --tunable` — the first tuning takes minutes and the screen may hang.
- `batch x seq_len ≤ 16384` per micro-step; use `--accum` beyond that.
- Parameters are pure fp16 (flat buffer views), master fp32 + dynamic loss scale;
  no autocast, no per-step cast.
- Every training run uses a new `--out` directory; a directory that already holds a ckpt is refused without `--resume`.
- Single GPU process (the iGPU also drives the display).

**Speed** (`ml\bench\bench10_out.txt`, TunableOp off, ms/step):

| Config | baseline (SDPA+LN+autocast) | fxlm (all fused) | fxlm_torchnorm (recommended) |
|---|---|---|---|
| T256 B16 | 788.5 ms — 5,195 tok/s | 492.4 ms — 8,318 tok/s — 1.60x | 420.2 ms — 9,748 tok/s — 1.88x |
| T512 B8 | 981.1 ms — 4,175 tok/s | 530.1 ms — 7,726 tok/s — 1.85x | 450.8 ms — 9,086 tok/s — 2.18x |
| T1024 B4 | 1406.3 ms — 2,913 tok/s | 513.7 ms — 7,974 tok/s — 2.74x | 513.8 ms — 7,973 tok/s — 2.74x |

Batch scaling (`bench10_bigbatch_out.txt`): at T256B32 fxlm_torchnorm reaches
796.6 ms — **10,284 tok/s** (baseline 5,405); at T512B16 859.0 ms —
**9,536 tok/s** (baseline 4,299). fxlm scales with batch, the baseline does not.

Convergence (`bench10_converge_out.txt`, byte-level 24 MB site-packages, 600 steps):
baseline 201.6 s → val 1.6323; **fxlm 97.6 s → val 1.6448** (2.07x, same learning).

End-to-end trial (24 Sep 2026, via `GPU_PYTHON.cmd`): 10.8M params (D=384 L=6 H=6, T256 B16,
byte-level ~30 MB of .py text), ~27,800 tok/s, 0 skips; val 2.4556@100 → 2.3536@200 → 2.2159@300;
`--resume` continued 300→400 seamlessly, val 2.1444@400. With so little training the generated text
is weak — normal.

Kernel microbenchmarks (`bench_kernels_out.txt`):

| Kernel | fused | PyTorch equivalent | speed |
|---|---|---|---|
| flash attn fwd+bwd B16 T256 Dh64 | 7.5 ms | SDPA math 34.1 ms | 4.5x |
| flash attn fwd+bwd B4 T1024 Dh64 | 19.0 ms | 107.1 ms | 5.7x |
| CE 4096x8192 fwd+bwd | 10.0 ms | fp32 cast 39.7 ms | 3.9x |
| CE 4096x16384 | 19.7 ms | 86.7 ms | 4.4x |
| linearCE 4096x512x8192 | 58.4 ms | linear+CE 82.6 ms | 1.4x |
| FlatAdamW.step 29.5M | 35.4 ms | scaler+clip+fused AdamW 47.2 ms | 1.3x |
| rms_norm 8192x768 fwd+bwd | 3.5 ms | F.rms_norm 3.4 ms | 0.96x (no gain) |

**Lessons:**

a) On gfx1035 SDPA's flash/mem-efficient backend falls back to math with a
   "No available kernel" error even when it reports "available" — the math backend is slow
   (B16 T256 fwd+bwd ~34 ms). The hand-written flash kernel is 4.5–5.8x faster at Dh64
   and ~2.4x faster at Dh128.
b) FULL `#pragma unroll` in the inner GEMM loops → VGPR ceiling (248) + ~1.3 KB of per-thread
   scratch spill → backward became 10–30x slower (dkv was 81 ms, dq 24.6 ms; after the fix fwd+bwd
   6.7 ms). `#pragma unroll 4` fixed it. Spill diagnosis:
   `kernel_regs.py` (hipFuncGetAttributes does not work on module handles —
   the ELF .kd is parsed instead).
c) The hand-written fused RMSNorm is not faster than F.rms_norm → default
   `norm_impl="torch"`; the fused path remains behind `--norm fused`.
d) The remaining step time is mostly rocBLAS matmuls. Next levers:
   a bigger token batch (bigbatch table above), TunableOp
   (risky / slow first run), dual-channel RAM — **done on 27 Sep 2026**, the gain is still to be confirmed with a first measurement.
