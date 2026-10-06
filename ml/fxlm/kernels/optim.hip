// fxlm FlatAdamW kernels: an apex multi_tensor_apply style by-value
// tensor list + flat master/m/v/p16 buffers. No host sync at all.
//
// adamw_sumsq  : sumsq + nonfinite detection over the fp16 grads
//                (atomicAdd -> sumsq, atomicOr -> flags)
// adamw_final  : single block; finalizes found_inf/norm/clip/step,
//                resets sumsq and flags for the next step
// adamw_update : no-op if found_inf; otherwise flat AdamW + p16 write
#include <hip/hip_fp16.h>

#define NTH 256
#define MAX_TENSORS 64
#define ELEMS_PER_BLOCK 2048          // 256 thread x 8 half

struct TensorList {
    const __half* g[MAX_TENSORS];   // fp16 grad pointers
    long long off[MAX_TENSORS];          // flat buffer offset (elements)
    int n[MAX_TENSORS];             // numel
    int wd[MAX_TENSORS];            // apply weight decay? (ndim>=2)
    int count;
};

// scratch layout (fp32): [0]=sumsq accum, [1]=clip_coef
// flags: int accum (atomicOr), step_ctr: int, found_inf: bool(uchar)
extern "C" __global__ void __launch_bounds__(NTH)
adamw_sumsq(TensorList L, float* __restrict__ sumsq,
            int* __restrict__ flags) {
    const int t = blockIdx.y;
    if (t >= L.count) return;
    const __half* g = L.g[t];
    const int n = L.n[t];
    const int start = blockIdx.x * ELEMS_PER_BLOCK;
    if (start >= n) return;
    float ss = 0.f;
    int bad = 0;
    const int i0 = start + threadIdx.x * 8;
    if (i0 + 8 <= n) {
        float4 v4 = *(const float4*)(g + i0);
        const __half2* h = (const __half2*)&v4;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            float2 f = __half22float2(h[u]);
            ss += f.x * f.x + f.y * f.y;
            bad |= !(isfinite(f.x) && isfinite(f.y));
        }
    } else {
        for (int i = i0; i < n && i < start + ELEMS_PER_BLOCK; i++) {
            float f = __half2float(g[i]);
            ss += f * f;
            bad |= !isfinite(f);
        }
    }
    // block reduce
    for (int off = 16; off; off >>= 1) {
        ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off, 32);
        bad |= __shfl_xor_sync(0xffffffffffffffffULL, bad, off, 32);
    }
    __shared__ float sred[8];
    __shared__ int bred[8];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { sred[wid] = ss; bred[wid] = bad; }
    __syncthreads();
    if (wid == 0) {
        ss = (lane < 8) ? sred[lane] : 0.f;
        bad = (lane < 8) ? bred[lane] : 0;
        for (int off = 4; off; off >>= 1) {
            ss += __shfl_xor_sync(0xffffffffffffffffULL, ss, off, 8);
            bad |= __shfl_xor_sync(0xffffffffffffffffULL, bad, off, 8);
        }
        if (lane == 0) {
            if (ss != 0.f) atomicAdd(sumsq, ss);
            if (bad) atomicOr(flags, 1);
        }
    }
}

extern "C" __global__ void adamw_final(float* __restrict__ scratch,
                                       int* __restrict__ flags,
                                       int* __restrict__ step_ctr,
                                       float* __restrict__ found_inf,
                                       float* __restrict__ norm_out,
                                       const float* __restrict__ scale,
                                       float max_norm) {
    if (threadIdx.x != 0) return;
    const int bad = *flags || !isfinite(scratch[0]);
    *found_inf = bad ? 1.f : 0.f;
    const float norm = sqrtf(scratch[0]) / (*scale);
    *norm_out = norm;
    scratch[1] = bad ? 0.f : fminf(1.f, max_norm / (norm + 1e-6f));
    if (!bad) *step_ctr += 1;
    *flags = 0;
    scratch[0] = 0.f;
}

extern "C" __global__ void __launch_bounds__(NTH)
adamw_update(TensorList L, float* __restrict__ master,
             float* __restrict__ m, float* __restrict__ v,
             __half* __restrict__ p16,
             const float* __restrict__ scratch,
             const float* __restrict__ found_inf,
             const float* __restrict__ scale_p,
             const int* __restrict__ step_p,
             float lr, float b1, float b2, float eps, float wd) {
    if (*found_inf != 0.f) return;
    const int t = blockIdx.y;
    if (t >= L.count) return;
    const int n = L.n[t];
    const int start = blockIdx.x * ELEMS_PER_BLOCK;
    if (start >= n) return;
    const float inv = scratch[1] / (*scale_p);     // clip_coef / scale
    const int step = *step_p;
    const float bc1 = 1.f - powf(b1, (float)step);
    const float bc2 = 1.f - powf(b2, (float)step);
    const float b1c = 1.f - b1, b2c = 1.f - b2;
    const int wd_on = L.wd[t];
    const __half* g = L.g[t];
    const long long base = L.off[t];

    const int i0 = start + threadIdx.x * 8;
    const int iend = min(n, start + ELEMS_PER_BLOCK);
    if (i0 + 8 <= n) {
        // 8-wide vectorized path
        float4 gv4 = *(const float4*)(g + i0);
        const __half2* gh = (const __half2*)&gv4;
        float gv[8];
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            float2 f = __half22float2(gh[u]);
            gv[2 * u] = f.x; gv[2 * u + 1] = f.y;
        }
        __half out[8];
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            const long long gi = base + i0 + u;
            float gg = gv[u] * inv;
            float mw = master[gi];
            if (wd_on) mw -= lr * wd * mw;
            float mv = m[gi] * b1 + b1c * gg;
            float vv = v[gi] * b2 + b2c * gg * gg;
            m[gi] = mv;
            v[gi] = vv;
            mw -= lr * (mv / bc1) / (sqrtf(vv / bc2) + eps);
            master[gi] = mw;
            out[u] = __float2half(mw);
        }
        *(float4*)(p16 + base + i0) = *(const float4*)out;
    } else {
        for (int i = i0; i < iend; i++) {
            const long long gi = base + i;
            float gg = __half2float(g[i]) * inv;
            float mw = master[gi];
            if (wd_on) mw -= lr * wd * mw;
            float mv = m[gi] * b1 + b1c * gg;
            float vv = v[gi] * b2 + b2c * gg * gg;
            m[gi] = mv;
            v[gi] = vv;
            mw -= lr * (mv / bc1) / (sqrtf(vv / bc2) + eps);
            master[gi] = mw;
            p16[gi] = __float2half(mw);
        }
    }
}
