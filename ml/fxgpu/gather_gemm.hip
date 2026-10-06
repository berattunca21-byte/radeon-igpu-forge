#include <hip/hip_fp16.h>

#define BM 128
#define BN 128
#define BK 32
#define TM 8
#define TN 8
#define PAD 4
#define TX (BN / TN)
#define TY (BM / TM)
#define NTHREADS 256
#define KP (BK / 2)

// X: (R, F) fp16; idx: (B,) int32 row indices (already = r + lo_offset applied)
// Wt: (N, K) fp16 ;  C: (B, N) fp16  ; K = WF
extern "C" __global__ void __launch_bounds__(NTHREADS)
gather_gemm(const __half* __restrict__ X, const int* __restrict__ idx,
            const __half* __restrict__ Wt, const __half* __restrict__ bias,
            __half* __restrict__ C, __half* __restrict__ Xg,
            int Bsz, int N, int K, int F_) {
    __shared__ __half2 As[2][KP][BM + PAD];
    __shared__ __half2 Bs[2][KP][BN + PAD];
    __shared__ int sidx[BM];          // resolved X row bases per tile

    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;
    const int tx = threadIdx.x % TX;
    const int ty = threadIdx.x / TX;
    const int tid = threadIdx.x;

    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.f;

    const int nkt = K / BK;

    // resolve idx[] for this block's rows once per tile is wasteful; idx is
    // fixed per block -> resolve once into registers/smem at start.
    // Each thread resolves the rows it will load: a_m0 = tid>>2, +64
    const int a_m0 = tid >> 2;
    const int a_m1 = a_m0 + 64;
    const int a_kc0 = (tid & 3) * 8;
    const long row0 = (bm + a_m0 < Bsz) ? (long)idx[bm + a_m0] : -1;
    const long row1 = (bm + a_m1 < Bsz) ? (long)idx[bm + a_m1] : -1;

    // B tile loader: Bs[kp][n] = half2(Wt[n][2kp], Wt[n][2kp+1])
    auto bload = [&](int kt, int buf) {
        const int b_n0 = tid >> 2, b_n1 = b_n0 + 64;
        const int b_kc = (tid & 3) * 8;
        int gk = kt * BK + b_kc;
        for (int rr = 0; rr < 2; rr++) {
            int n = rr ? b_n1 : b_n0;
            int gn = bn + n;
            float4 v = {0,0,0,0};
            if (gn < N) {
                if (gk + 7 < K)
                    v = *reinterpret_cast<const float4*>(Wt + (long)gn * K + gk);
                else {
                    __half tmp[8] = {};
                    for (int t = 0; t < 8 && gk + t < K; t++)
                        tmp[t] = Wt[(long)gn * K + gk + t];
                    v = *reinterpret_cast<float4*>(tmp);
                }
            }
            const __half2* hv = (const __half2*)&v;
            #pragma unroll
            for (int u = 0; u < 4; u++)
                Bs[buf][b_kc / 2 + u][n] = hv[u];
        }
    };

    // A gather loader: per half2 u, k_abs = kt*BK + kc + 2u -> w=k_abs/F_, f=k_abs%F_
    auto aload = [&](int kt, int buf) {
        for (int rr = 0; rr < 2; rr++) {
            long row = rr ? row1 : row0;
            int m = rr ? a_m1 : a_m0;
            int kbase = kt * BK + a_kc0;          // first k of this thread's 8
            int w = kbase / F_, f = kbase - w * F_;
            if (row >= 0 && f + 7 < F_ && (f & 7) == 0 && kbase + 7 < K) {
                // whole 8-half chunk inside one X row, 16B aligned
                float4 v4 = *reinterpret_cast<const float4*>(X + (row + w) * F_ + f);
                const __half2* hv = (const __half2*)&v4;
                #pragma unroll
                for (int u = 0; u < 4; u++)
                    As[buf][a_kc0 / 2 + u][m] = hv[u];
                if (Xg && bm + m < Bsz)
                    *reinterpret_cast<float4*>(Xg + (long)(bm + m) * K + kbase) = v4;
            } else {
                #pragma unroll
                for (int u = 0; u < 4; u++) {
                    int kabs = kbase + 2 * u;
                    int wu = kabs / F_, fu = kabs - wu * F_;
                    __half2 v = __floats2half2_rn(0.f, 0.f);
                    if (row >= 0 && kabs + 1 < K) {
                        v = *reinterpret_cast<const __half2*>(X + (row + wu) * F_ + fu);
                    } else if (row >= 0 && kabs < K) {
                        v = __halves2half2(X[(row + wu) * F_ + fu], __half(0.f));
                    }
                    As[buf][a_kc0 / 2 + u][m] = v;
                    if (Xg && bm + m < Bsz && kabs + 1 < K)
                        *reinterpret_cast<__half2*>(Xg + (long)(bm + m) * K + kabs) = v;
                }
            }
        }
    };

    aload(0, 0);
    bload(0, 0);
    __syncthreads();

    for (int kt = 0; kt < nkt; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < nkt) {
            aload(kt + 1, buf ^ 1);
            bload(kt + 1, buf ^ 1);
        }
        #pragma unroll
        for (int kk = 0; kk < KP; kk++) {
            __half2 a2[TM], b2[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                a2[i] = As[buf][kk][ty + i * TY];
            #pragma unroll
            for (int j = 0; j < TN; j++)
                b2[j] = Bs[buf][kk][tx + j * TX];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], acc[i][j], false);
        }
        __syncthreads();
    }

    // tail
    for (int k0 = nkt * BK; k0 < K; k0 += 2) {
        __half2 a2[TM], b2[TN];
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            int m = ty + i * TY;
            long row = (bm + m < Bsz) ? (long)idx[bm + m] : -1;
            int w0 = k0 / F_, f0 = k0 - w0 * F_;
            int w1 = (k0 + 1) / F_, f1 = k0 + 1 - w1 * F_;
            __half lo = (row >= 0) ? X[(row + w0) * F_ + f0] : __half(0.f);
            __half hi = (row >= 0) ? X[(row + w1) * F_ + f1] : __half(0.f);
            a2[i] = __halves2half2(lo, hi);
            if (Xg && bm + m < Bsz && k0 + 1 < K)
                *reinterpret_cast<__half2*>(Xg + (long)(bm + m) * K + k0) = a2[i];
        }
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = bn + tx + j * TX;
            __half lo = (gn < N && k0 < K) ? Wt[(long)gn * K + k0] : __half(0.f);
            __half hi = (gn < N && k0 + 1 < K) ? Wt[(long)gn * K + k0 + 1] : __half(0.f);
            b2[j] = __halves2half2(lo, hi);
        }
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++)
                acc[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], acc[i][j], false);
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        int gm = bm + ty + i * TY;
        if (gm >= Bsz) break;
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int gn = bn + tx + j * TX;
            if (gn < N) {
                float v = acc[i][j];
                if (bias) v += __half2float(bias[gn]);
                C[(long)gm * N + gn] = __float2half(v);
            }
        }
    }
}
