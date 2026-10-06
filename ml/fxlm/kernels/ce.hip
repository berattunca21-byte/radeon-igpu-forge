// fxlm CE kernel: satir basina online logsumexp (fp32) + softmax grad yazimi.
// Bir blok bir satir isler; V*2 <= 60KB ise satir LDS'e cache'lenir.
// grad = (softmax - onehot) * (*mul)   (mul = scale / n_valid, device scalar)
// loss += atomicAdd( lse - x_target )   (sadece gecerli satirlar)
#include <hip/hip_fp16.h>
#define F_INF __int_as_float(0x7f800000)

#define NTH 256

struct CEArgs {
    const __half* __restrict__ logits;   // (N, V) fp16 contiguous
    const long long* __restrict__ target;     // (N,) int64
    __half* __restrict__ grad;           // (N, V) fp16 veya nullptr
    float* __restrict__ loss;            // () fp32 accum (host'ta sifirlanir)
    const float* __restrict__ mul;       // () fp32: grad carpani (scale/n_valid)
    int N, V;
    long long ignore_index;
    int use_lds;                         // satir LDS'e sigarsa 1
};

__device__ __forceinline__ float warp_max(float v) {
    for (int off = 16; off; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32));
    return v;
}
__device__ __forceinline__ float warp_sum2(float v) {
    for (int off = 16; off; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 32);
    return v;
}

extern "C" __global__ void __launch_bounds__(NTH)
ce_fwd_grad(CEArgs a) {
    extern __shared__ __half rowbuf[];    // use_lds ise V half
    __shared__ float wred[8];
    __shared__ float mred[8];
    __shared__ float s_m, s_z;
    const int row = blockIdx.x;
    if (row >= a.N) return;
    const __half* lr = a.logits + (long long)row * a.V;
    const int V = a.V;
    const long long t = a.target[row];
    const bool valid = (t != a.ignore_index) && (t >= 0) && (t < (long long)V);

    const __half* src = lr;
    if (a.use_lds) {
        for (int d = threadIdx.x; d < V; d += NTH) rowbuf[d] = lr[d];
        __syncthreads();
        src = rowbuf;
    }

    // pass 1: online logsumexp — thread-local (m, l), sonra blok merge
    float mt = (-F_INF), lt = 0.f;
    for (int d = threadIdx.x; d < V; d += NTH) {
        float f = __half2float(src[d]);
        if (f > mt) { lt = lt * __expf(mt - f) + 1.f; mt = f; }
        else        { lt += __expf(f - mt); }
    }
    // warp merge (m, l)
    for (int off = 16; off; off >>= 1) {
        float m2 = __shfl_xor_sync(0xffffffffffffffffULL, mt, off, 32);
        float l2 = __shfl_xor_sync(0xffffffffffffffffULL, lt, off, 32);
        float mn = fmaxf(mt, m2);
        float ea = (mt > (-F_INF)) ? __expf(mt - mn) : 0.f;
        float eb = (m2 > (-F_INF)) ? __expf(m2 - mn) : 0.f;
        lt = lt * ea + l2 * eb;
        mt = mn;
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { mred[wid] = mt; wred[wid] = lt; }
    __syncthreads();
    if (wid == 0) {
        mt = (lane < 8) ? mred[lane] : (-F_INF);
        lt = (lane < 8) ? wred[lane] : 0.f;
        for (int off = 4; off; off >>= 1) {
            float m2 = __shfl_xor_sync(0xffffffffffffffffULL, mt, off, 8);
            float l2 = __shfl_xor_sync(0xffffffffffffffffULL, lt, off, 8);
            float mn = fmaxf(mt, m2);
            float ea = (mt > (-F_INF)) ? __expf(mt - mn) : 0.f;
            float eb = (m2 > (-F_INF)) ? __expf(m2 - mn) : 0.f;
            lt = lt * ea + l2 * eb;
            mt = mn;
        }
        if (lane == 0) { s_m = mt; s_z = lt; }
    }
    __syncthreads();
    const float m = s_m, Z = s_z;

    if (valid && threadIdx.x == 0) {
        float xt = __half2float(src[t]);
        atomicAdd(a.loss, m + __logf(Z) - xt);
    }
    if (a.grad == nullptr) return;
    const float gmul = valid ? *a.mul : 0.f;
    const float invz = 1.f / Z;
    __half* gr = a.grad + (long long)row * a.V;
    if ((V & 1) == 0) {
        // half2 yazim: p(d), p(d+1)
        for (int d = threadIdx.x * 2; d < V; d += NTH * 2) {
            float p0 = __expf(__half2float(src[d]) - m) * invz;
            float p1 = __expf(__half2float(src[d + 1]) - m) * invz;
            if (valid) {
                if (d == t) p0 -= 1.f;
                if (d + 1 == t) p1 -= 1.f;
            } else { p0 = 0.f; p1 = 0.f; }
            __half2 h = __floats2half2_rn(p0 * gmul, p1 * gmul);
            *(__half2*)(gr + d) = h;
        }
    } else {
        for (int d = threadIdx.x; d < V; d += NTH) {
            float p = valid ? __expf(__half2float(src[d]) - m) * invz
                            - (d == t ? 1.f : 0.f) : 0.f;
            gr[d] = __float2half(p * gmul);
        }
    }
}
