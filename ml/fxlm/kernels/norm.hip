// fxlm norm kernel'lari: fused RMSNorm fwd+bwd (fp16 io, fp32 math).
// D % 8 == 0 ve D <= 8192 sart (host tarafi dogrular).
#include <hip/hip_fp16.h>

#define NTH 256

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32);
    return v;
}

// tum thread'ler cagirmali (uniform control flow)
__device__ float block_sum(float v, float* wred /* >= 8 */) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) wred[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = (lane < (int)((blockDim.x + 31) >> 5)) ? wred[lane] : 0.f;
        t = warp_sum(t);
        if (lane == 0) wred[0] = t;
    }
    __syncthreads();
    float r = wred[0];
    __syncthreads();
    return r;
}

// bir blok bir satir: y = x * rstd * w ; rstd = rsqrt(mean(x^2)+eps)
// extern shared: D half (x satir cache'i)
extern "C" __global__ void __launch_bounds__(NTH)
rms_norm_fwd(const __half* __restrict__ x, const __half* __restrict__ w,
             __half* __restrict__ y, float* __restrict__ rstd,
             int D, float eps) {
    extern __shared__ __half xs[];
    __shared__ float wred[8];
    __shared__ float s_rstd;
    const long long row = blockIdx.x;
    const __half* xr = x + row * (long long)D;

    float ss = 0.f;
    for (int d = threadIdx.x; d < D; d += NTH) {
        __half xv = xr[d];
        xs[d] = xv;
        float f = __half2float(xv);
        ss += f * f;
    }
    ss = block_sum(ss, wred);
    if (threadIdx.x == 0) {
        s_rstd = rsqrtf(ss / (float)D + eps);
        rstd[row] = s_rstd;
    }
    __syncthreads();
    const float r = s_rstd;
    __half* yr = y + row * (long long)D;
    for (int d = threadIdx.x; d < D; d += NTH)
        yr[d] = __float2half(__half2float(xs[d]) * r * __half2float(w[d]));
}

// bir blok bir satir: dx = rstd*w*dy - x * (rstd^3 * t / D), t = sum(x*w*dy)
// extern shared: 2*D half (x + dy cache)
extern "C" __global__ void __launch_bounds__(NTH)
rms_norm_bwd_dx(const __half* __restrict__ x, const __half* __restrict__ dy,
                const __half* __restrict__ w, const float* __restrict__ rstd,
                __half* __restrict__ dx, int D) {
    extern __shared__ __half xs[];          // xs[0:D] = x, xs[D:2D] = dy
    __shared__ float wred[8];
    __shared__ float s_t;
    const long long row = blockIdx.x;
    const __half* xr = x + row * (long long)D;
    const __half* dr = dy + row * (long long)D;
    for (int d = threadIdx.x; d < D; d += NTH) {
        xs[d] = xr[d];
        xs[D + d] = dr[d];
    }
    __syncthreads();
    float t = 0.f;
    for (int d = threadIdx.x; d < D; d += NTH)
        t += __half2float(xs[d]) * __half2float(w[d])
             * __half2float(xs[D + d]);
    t = block_sum(t, wred);
    if (threadIdx.x == 0) {
        float r = rstd[row];
        s_t = r * r * r * t / (float)D;
    }
    __syncthreads();
    const float r = rstd[row];
    const float corr = s_t;
    __half* dxr = dx + row * (long long)D;
    for (int d = threadIdx.x; d < D; d += NTH)
        dxr[d] = __float2half(
            r * __half2float(w[d]) * __half2float(xs[D + d])
            - __half2float(xs[d]) * corr);
}

// dweight kismi toplami: part[strip][d] = sum_{r in strip} x[r][d]*rstd[r]*dy[r][d]
// grid.x = strip sayisi; her blok RPB satiri isler. Host tarafi fp32 part'i
// sum(0) ile indirger.
#define RPB 64
extern "C" __global__ void __launch_bounds__(NTH)
rms_norm_bwd_dw(const __half* __restrict__ x, const __half* __restrict__ dy,
                const float* __restrict__ rstd, float* __restrict__ part,
                int N, int D, int nrows_total) {
    const int r0 = blockIdx.x * RPB;
    const int r1 = min(r0 + RPB, nrows_total);
    // thread d-subset'i: d = tid + k*NTH (k compile-time unroll — register'da kalir)
    float acc[32];                          // D <= 8192 -> kmax <= 32
    #pragma unroll
    for (int k = 0; k < 32; k++) acc[k] = 0.f;
    for (int r = r0; r < r1; r++) {
        const float rs = rstd[r];
        const __half* xr = x + (long long)r * D;
        const __half* dr = dy + (long long)r * D;
        #pragma unroll
        for (int k = 0; k < 32; k++) {
            int d = threadIdx.x + k * NTH;
            if (d < D)
                acc[k] += __half2float(xr[d]) * rs * __half2float(dr[d]);
        }
    }
    #pragma unroll
    for (int k = 0; k < 32; k++) {
        int d = threadIdx.x + k * NTH;
        if (d < D) part[(long long)blockIdx.x * D + d] = acc[k];
    }
}
