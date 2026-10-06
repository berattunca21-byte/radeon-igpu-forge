"""fxlm.kernel_regs: her kernel icin vgpr/sgpr/scratch/LDS raporu.

Neden var: attention bwd kernel'lari ic dongulerde TAM `#pragma unroll`
kullaninca LLVM tum LDS yuklerini ayni anda canli tuttu; VGPR 248'e
(tavana) dayanip thread basina ~1.3KB scratch spill oldu ve geri
gecis 10-30x yavasladi (B16 T256'da dkv 81 ms, dq 24.6 ms; duzeltme
sonrasi fwd+bwd 6.7 ms). hipFuncGetAttributes modul-handle'li
kernel'larda calismaz (hipError 13) — bu yuzden hiprtc code object
(ELF64) icindeki <kernel>.kd descriptor'u parse edilir:
compute_pgm_rsrc1 bit[5:0]*8 = vgpr, bit[9:6]*16 = sgpr;
offset 4 = private_segment_fixed_size (spill/scratch byte);
offset 0 = LDS byte. scratch > 0 => spill var, kerneli incele.

Calistirma (KB kokunden):
    venv-rocm10\\Scripts\\python.exe ml\\fxlm\\kernel_regs.py
"""
import ctypes
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fxgpu.hip_kernel import hiprtc, _INC  # noqa: E402

_KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "kernels")


def compile_code(src, arch="gfx1035", extra_opts=None):
    """hiprtc ile derle, ham code object (ELF64) byte'larini dondur."""
    prog = ctypes.c_void_p()
    err = hiprtc.hiprtcCreateProgram(ctypes.byref(prog), src.encode(),
                                   b"k.hip", 0, None, None)
    if err != 0:
        raise RuntimeError(f"hiprtcCreateProgram: {err}")
    opts = [f"--offload-arch={arch}", f"-I{_INC}", "-O3"] + (extra_opts or [])
    arr = (ctypes.c_char_p * len(opts))(*[o.encode() for o in opts])
    err = hiprtc.hiprtcCompileProgram(prog, len(opts), arr)
    ls = ctypes.c_size_t()
    hiprtc.hiprtcGetProgramLogSize(prog, ctypes.byref(ls))
    if ls.value > 1:
        log = ctypes.create_string_buffer(ls.value)
        hiprtc.hiprtcGetProgramLog(prog, log)
        print("[hiprtc log]", log.value.decode(errors="replace"))
    if err != 0:
        hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
        raise RuntimeError(f"hiprtcCompileProgram: {err}")
    cs = ctypes.c_size_t()
    hiprtc.hiprtcGetCodeSize(prog, ctypes.byref(cs))
    code = ctypes.create_string_buffer(cs.value)
    hiprtc.hiprtcGetCode(prog, code)
    hiprtc.hiprtcDestroyProgram(ctypes.byref(prog))
    return code.raw


def kd_info(d):
    """ELF64 code object'ten {kernel: {vgpr, sgpr, scratch, lds}} cikar."""
    e_shoff = struct.unpack_from("<Q", d, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", d, 0x3A)
    sh = [struct.unpack_from("<IIQQQQIIQQ", d, e_shoff + i * e_shentsize)
          for i in range(e_shnum)]
    so = sh[e_shstrndx][4]

    def sname(n):
        return d[so + n:d.index(b"\0", so + n)].decode()

    sym = str_ = None
    for s in sh:
        nm = sname(s[0])
        if nm == ".symtab":
            sym = s
        elif nm == ".strtab":
            str_ = s
    out = {}
    if not (sym and str_):
        return out
    for i in range(sym[5] // 24):
        o = sym[4] + i * 24
        st_name, _info, _other, shndx, val, _size = struct.unpack_from(
            "<IBBHQQ", d, o)
        nm = d[str_[4] + st_name:d.index(b"\0", str_[4] + st_name)].decode()
        if nm.endswith(".kd") and shndx < len(sh):
            sec = sh[shndx]
            foff = sec[4] + (val - sec[3])
            kd = d[foff:foff + 64]
            rsrc1 = struct.unpack_from("<I", kd, 48)[0]
            out[nm[:-3]] = dict(
                lds=struct.unpack_from("<I", kd, 0)[0],
                scratch=struct.unpack_from("<I", kd, 4)[0],
                vgpr=(rsrc1 & 0x3F) * 8,
                sgpr=((rsrc1 >> 6) & 0xF) * 16)
    return out


def main():
    jobs = []
    # attention.hip: wrapper'in kullandigi define'larla iki varyant
    for dh, brbc in ((64, (64, 64)), (128, (32, 32))):
        jobs.append((f"attention.hip HEAD_DIM={dh} BR={brbc[0]} "
                     f"BC={brbc[1]}",
                     "attention.hip",
                     [f"-DHEAD_DIM={dh}", f"-DBR={brbc[0]}",
                      f"-DBC={brbc[1]}"]))
    for f in ("norm.hip", "ce.hip", "optim.hip"):
        jobs.append((f, f, []))

    any_spill = False
    for label, fname, opts in jobs:
        with open(os.path.join(_KERNELS_DIR, fname),
                  encoding="utf-8") as fh:
            src = fh.read()
        info = kd_info(compile_code(src, extra_opts=opts))
        print(f"--- {label} ---")
        for k in sorted(info):
            v = info[k]
            flag = "  <-- SPILL!" if v["scratch"] else ""
            any_spill |= bool(v["scratch"])
            print(f"  {k:20s} vgpr={v['vgpr']:3d} sgpr={v['sgpr']:3d} "
                  f"scratch={v['scratch']:5d}B lds={v['lds']:5d}B{flag}")
    print("\n" + ("UYARI: spill'li kernel var!" if any_spill
                  else "spill yok."))


if __name__ == "__main__":
    main()
