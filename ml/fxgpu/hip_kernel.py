"""hip_kernel: minimal hiprtc+ctypes harness for raw HIP kernels on gfx1035.

fxgpu kopyasi — hip\\hip_kernel.py ile ayni API (compile_kernel/launch).
Fark: DLL'ler sabit isim yerine glob ile bulunur (amdhip64*.dll /
hiprtc*.dll) ve _rocm_sdk_core dizini/DLL'ler yoksa IndexError yerine
acik mesajli RuntimeError firlatilir.
"""
import ctypes
import glob
import os
import sys

import torch


def _sdk_subdir(kind):
    """site-packages altinda _rocm_sdk_core*/<kind> dizinini bul."""
    sp = os.path.abspath(os.path.join(os.path.dirname(torch.__file__), ".."))
    hits = sorted(glob.glob(os.path.join(sp, "_rocm_sdk_core*", kind)))
    if not hits:
        raise RuntimeError(
            f"ROCm SDK '{kind}' dizini bulunamadi: {sp} altinda "
            f"_rocm_sdk_core*/{kind} yok. Dizin icerigi: "
            f"{sorted(os.listdir(sp))}")
    return os.path.abspath(hits[0])


_BIN = _sdk_subdir("bin")
_INC = _sdk_subdir("include")


def _load_first(pattern):
    hits = sorted(h for h in glob.glob(os.path.join(_BIN, pattern))
                  if "builtins" not in os.path.basename(h))
    if not hits:
        raise RuntimeError(
            f"{pattern} bulunamadi: {_BIN} icerik={sorted(os.listdir(_BIN))}")
    return ctypes.CDLL(hits[0])


amdhip = _load_first("amdhip64*.dll")
hiprtc = _load_first("hiprtc*.dll")


def _chk(err, what):
    if err != 0:
        raise RuntimeError(f"{what} failed: hipError={err}")


def compile_kernel(src: str, name: str, arch: str = "gfx1035",
                   extra_opts=None):
    """Compile src with hiprtc, return (module, function) ctypes handles."""
    prog = ctypes.c_void_p()
    _chk(hiprtc.hiprtcCreateProgram(
        ctypes.byref(prog), src.encode(), b"k.hip", 0, None, None),
        "hiprtcCreateProgram")
    opts = [f"--offload-arch={arch}", f"-I{_INC}", "-O3"]
    if extra_opts:
        opts += extra_opts
    opt_arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    err = hiprtc.hiprtcCompileProgram(prog, len(opts), opt_arr)
    # grab log regardless
    logsize = ctypes.c_size_t()
    hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(logsize))
    if logsize.value > 1:
        log = ctypes.create_string_buffer(logsize.value)
        hiprtc.hiprtcGetProgramLog(prog, log)
        print("[hiprtc log]", log.value.decode(errors="replace"))
    if err != 0:
        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"hiprtcCompileProgram failed: {err}")

    codesize = ctypes.c_size_t()
    _chk(hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(codesize)), "GetCodeSize")
    code = ctypes.create_string_buffer(codesize.value)
    _chk(hiprtc.hiprtcGetCode(prog, code), "GetCode")
    hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))

    module = ctypes.c_void_p()
    _chk(amdhip.hipModuleLoadData(ctypes.byref(module), code),
         "hipModuleLoadData")
    func = ctypes.c_void_p()
    _chk(amdhip.hipModuleGetFunction(
        ctypes.byref(func), module, name.encode()), "hipModuleGetFunction")
    return module, func


def compile_module(src: str, names, arch: str = "gfx1035",
                   extra_opts=None):
    """Tek hiprtc derlemesinde birden cok kernel: {isim: func} sozlugu.

    Bir .hip modulunun tum __global__ fonksiyonlarini tek seferde derleyip
    dondurur (fxlm gibi cok-kernelli paketler icin). Geriye donuk:
    compile_kernel davranisina dokunmaz.
    """
    prog = ctypes.c_void_p()
    _chk(hiprtc.hiprtcCreateProgram(
        ctypes.byref(prog), src.encode(), b"k.hip", 0, None, None),
        "hiprtcCreateProgram")
    opts = [f"--offload-arch={arch}", f"-I{_INC}", "-O3"]
    if extra_opts:
        opts += extra_opts
    opt_arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    err = hiprtc.hiprtcCompileProgram(prog, len(opts), opt_arr)
    logsize = ctypes.c_size_t()
    hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(logsize))
    if logsize.value > 1:
        log = ctypes.create_string_buffer(logsize.value)
        hiprtc.hiprtcGetProgramLog(prog, log)
        print("[hiprtc log]", log.value.decode(errors="replace"))
    if err != 0:
        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"hiprtcCompileProgram failed: {err}")

    codesize = ctypes.c_size_t()
    _chk(hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(codesize)), "GetCodeSize")
    code = ctypes.create_string_buffer(codesize.value)
    _chk(hiprtc.hiprtcGetCode(prog, code), "GetCode")
    hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))

    module = ctypes.c_void_p()
    _chk(amdhip.hipModuleLoadData(ctypes.byref(module), code),
         "hipModuleLoadData")
    out = {}
    for name in names:
        func = ctypes.c_void_p()
        _chk(amdhip.hipModuleGetFunction(
            ctypes.byref(func), module, name.encode()),
            f"hipModuleGetFunction({name})")
        out[name] = func
    return out


def launch(func, grid, block, args, shared=0, stream=None):
    """args: list of ctypes values passed by reference."""
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    params = (ctypes.c_void_p * len(args))()
    holders = [ctypes.byref(a) for a in args]
    for i, h in enumerate(holders):
        params[i] = ctypes.cast(h, ctypes.c_void_p)
    _chk(amdhip.hipModuleLaunchKernel(
        func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
        shared, ctypes.c_void_p(stream),
        params, None), "hipModuleLaunchKernel")


if __name__ == "__main__":
    # smoke: fill a tensor with 42 via raw HIP kernel
    src = r"""
extern "C" __global__ void fill42(float* p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 42.0f;
}
"""
    torch.cuda.init()
    x = torch.zeros(1000, device="cuda")
    mod, fn = compile_kernel(src, "fill42")
    launch(fn, (4, 1, 1), (256, 1, 1),
           [ctypes.c_void_p(x.data_ptr()), ctypes.c_int(x.numel())])
    torch.cuda.synchronize()
    assert (x == 42).all(), "fill42 failed"
    print("hiprtc harness OK — fill42 wrote 42s to torch tensor")
