// fxlm flash-attention kernel'lari (FA2 tarzi, gfx1035 wave32, fdot2).
//
// Layout: q,k,v (B,T,H,DH) fp16 — d stride=1, diger stride'lar serbest.
// o (B,T,H,DH) fp16 contiguous; lse (B,H,T) fp32.
//
// Tile semasi (16x16 = 256 thread): S icin m = TM*ty+i, n = TN*tx+j;
// O icin m = TM*ty+i, d = TD*tx+j. Softmax satir gruplari 16 ardisik lane
// (shfl_xor width=16). P/dS LDS'e n-cifti paketli yazilir (fdot2 icin).
//
// LDS plani:
//   fwd    : Qs[D2*BR] Ks[D2*BC] Vn[BC/2*DH] Ps[BC/2*BR]
//   bwd_dkv: Ks[D2*BC] Vs[D2*BC] Qm[BR/2*DH] dOm[BR/2*DH]
//            bufA[D2*BR] (Qs_d -> Ps^T)  bufB[D2*BR] (dOs_d -> dSs^T)
//   bwd_dq : Qs[D2*BR] dOs[D2*BR] Ks[D2*BC] Vs[D2*BC] Kn[BC/2*DH] dSs[BC/2*BR]
#include <hip/hip_fp16.h>
#define F_INF __int_as_float(0x7f800000)

#ifndef HEAD_DIM
#define HEAD_DIM 64
#endif
#ifndef BR
#define BR 64
#endif
#ifndef BC
#define BC 64
#endif

#define NT 16
#define NTH 256
#define D2 (HEAD_DIM / 2)
#define TM (BR / NT)
#define TN (BC / NT)
#define TD (HEAD_DIM / NT)

struct AttnArgs {
    const __half* q; const __half* k; const __half* v;
    const __half* o; const __half* dout;
    __half* o_out; __half* dq; __half* dk; __half* dv;
    const float* lse; float* lse_out;
    const float* delta; float* delta_out;
    // eleman stride'lari (b, t, h sirasiyla; d stride = 1 varsayilir)
    long long qsb, qst, qsh, ksb, kst, ksh, vsb, vst, vsh;
    long long dqb, dqt, dqh, dkb, dkt, dkh, dvb, dvt, dvh;
    int B, T, H;
    float scale;
    int causal;
};

// --- yardimcilar ------------------------------------------------------------

// R satirlik (R x HEAD_DIM) tile'i LDS'e "d-boyunca paketli" yukler:
// dst[kp*R + r] = half2(src[r][2kp], src[r][2kp+1])
__device__ void load_pd(const __half* g, long long rstride, int valid,
                        __half2* dst, int R) {
    const int chunks = R * HEAD_DIM / 8;
    const int cpr = HEAD_DIM / 8;                 // satir basi 16B chunk
    for (int c = threadIdx.x; c < chunks; c += NTH) {
        int r = c / cpr;
        int d0 = (c - r * cpr) * 8;
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (r < valid)
            v = *(const float4*)(g + (long long)r * rstride + d0);
        const __half2* hv = (const __half2*)&v;
        #pragma unroll
        for (int u = 0; u < 4; u++)
            dst[(d0 / 2 + u) * R + r] = hv[u];
    }
}

// R satirlik tile'i LDS'e "satir-cifti paketli" yukler:
// dst[rp*HEAD_DIM + d] = half2(src[2rp][d], src[2rp+1][d])
__device__ void load_pr(const __half* g, long long rstride, int valid,
                        __half2* dst, int R) {
    const int total = (R / 2) * HEAD_DIM;
    for (int e = threadIdx.x; e < total; e += NTH) {
        int rp = e / HEAD_DIM;
        int d = e - rp * HEAD_DIM;
        __half lo = __float2half(0.f), hi = __float2half(0.f);
        if (2 * rp < valid)
            lo = g[(long long)(2 * rp) * rstride + d];
        if (2 * rp + 1 < valid)
            hi = g[(long long)(2 * rp + 1) * rstride + d];
        dst[rp * HEAD_DIM + d] = __halves2half2(lo, hi);
    }
}

// 16-lane'lik satir-grubu icinde max/sum (ty ayni olan 16 thread)
__device__ __forceinline__ float grp_max(float v) {
    #pragma unroll
    for (int off = 8; off; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffffffffffULL, v, off, 16));
    return v;
}
__device__ __forceinline__ float grp_sum(float v) {
    #pragma unroll
    for (int off = 8; off; off >>= 1)
        v += __shfl_xor_sync(0xffffffffffffffffULL, v, off, 16);
    return v;
}

// --- forward ----------------------------------------------------------------

extern "C" __global__ void __launch_bounds__(NTH)
attn_fwd(AttnArgs a) {
    __shared__ __half2 Qs[D2 * BR];
    __shared__ __half2 Ks[D2 * BC];
    __shared__ __half2 Vn[(BC / 2) * HEAD_DIM];
    __shared__ __half2 Ps[(BC / 2) * BR];

    const int bh = blockIdx.y;
    const int b = bh / a.H, h = bh - b * a.H;
    const int bm = blockIdx.x * BR;
    const int tx = threadIdx.x & (NT - 1), ty = threadIdx.x / NT;

    load_pd(a.q + b * a.qsb + h * a.qsh + (long long)bm * a.qst,
            a.qst, a.T - bm, Qs, BR);
    __syncthreads();

    float m[TM], l[TM], acc[TM][TD];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        m[i] = (-F_INF); l[i] = 0.f;
        #pragma unroll
        for (int j = 0; j < TD; j++) acc[i][j] = 0.f;
    }

    const int jmax = a.causal ? min(a.T, bm + BR) : a.T;
    for (int jn0 = 0; jn0 < jmax; jn0 += BC) {
        load_pd(a.k + b * a.ksb + h * a.ksh + (long long)jn0 * a.kst,
                a.kst, a.T - jn0, Ks, BC);
        load_pr(a.v + b * a.vsb + h * a.vsh + (long long)jn0 * a.vst,
                a.vst, a.T - jn0, Vn, BC);
        __syncthreads();

        // S = Q K^T (TM x TN per thread)
        float s[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) s[i][j] = 0.f;
        #pragma unroll
        for (int kp = 0; kp < D2; kp++) {
            __half2 a2[TM], b2[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a2[i] = Qs[kp * BR + TM * ty + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b2[j] = Ks[kp * BC + TN * tx + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    s[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], s[i][j], false);
        }

        // online softmax + P'nin n-cifti paketli yazimi
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int ml = TM * ty + i;
            const int mg = bm + ml;
            float t[TN];
            float rmax = (-F_INF);
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const int ng = jn0 + TN * tx + j;
                const bool ok = (mg < a.T) && (ng < a.T)
                                && (!a.causal || ng <= mg);
                t[j] = ok ? s[i][j] * a.scale : (-F_INF);
                rmax = fmaxf(rmax, t[j]);
            }
            rmax = grp_max(rmax);
            const float mnew = fmaxf(m[i], rmax);
            const float mu = isinf(mnew) ? 0.f : mnew;
            const float alpha = __expf(m[i] - mu);
            float rs = 0.f;
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                t[j] = __expf(t[j] - mu);
                rs += t[j];
            }
            rs = grp_sum(rs);
            #pragma unroll
            for (int jj = 0; jj < TN / 2; jj++)
                Ps[(TN * tx / 2 + jj) * BR + ml] =
                    __floats2half2_rn(t[2 * jj], t[2 * jj + 1]);
            l[i] = l[i] * alpha + rs;
            #pragma unroll
            for (int j = 0; j < TD; j++) acc[i][j] *= alpha;
            m[i] = mnew;
        }
        __syncthreads();

        // acc += P V  (np = kv-cifti)
        #pragma unroll
        for (int np = 0; np < BC / 2; np++) {
            __half2 a2[TM], b2[TD];
            #pragma unroll
            for (int i = 0; i < TM; i++) a2[i] = Ps[np * BR + TM * ty + i];
            #pragma unroll
            for (int j = 0; j < TD; j++) b2[j] = Vn[np * HEAD_DIM + TD * tx + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TD; j++)
                    acc[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], acc[i][j], false);
        }
        __syncthreads();
    }

    // epilogue: O ve LSE
    const long long obase = (long long)b * (a.T * a.H * HEAD_DIM) + h * HEAD_DIM;
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int mg = bm + TM * ty + i;
        if (mg >= a.T) continue;
        const float inv = (l[i] > 0.f) ? 1.f / l[i] : 0.f;
        __half* orow = a.o_out + obase + (long long)mg * (a.H * HEAD_DIM);
        #pragma unroll
        for (int j = 0; j < TD; j++)
            orow[TD * tx + j] = __float2half(acc[i][j] * inv);
        if (tx == 0)
            a.lse_out[((long long)bh) * a.T + mg] =
                (l[i] > 0.f) ? m[i] + __logf(l[i]) : (-F_INF);
    }
}

// --- backward: Delta = rowsum(dO * O) ---------------------------------------

extern "C" __global__ void __launch_bounds__(NTH)
attn_bwd_prep(AttnArgs a) {
    // grid.x = B*H ; her warp bir satir (t = wid, wid+8, ...)
    // o ve dout contiguous (B,T,H,DH) varsayilir (wrapper garantiler)
    const int bh = blockIdx.x;
    const int b = bh / a.H, h = bh - b * a.H;
    const int wid = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const long long rowstride = (long long)a.H * HEAD_DIM;
    for (int t = wid; t < a.T; t += NTH / 32) {
        const __half* o2 = a.o + ((long long)b * a.T + t) * rowstride
                         + h * HEAD_DIM;
        const __half* g2 = a.dout + ((long long)b * a.T + t) * rowstride
                           + h * HEAD_DIM;
        float d = 0.f;
        for (int dd = lane * 2; dd < HEAD_DIM; dd += 64) {
            __half2 ov = *(const __half2*)(o2 + dd);
            __half2 gv = *(const __half2*)(g2 + dd);
            d = __builtin_amdgcn_fdot2(ov, gv, d, false);
        }
        for (int off = 16; off; off >>= 1)
            d += __shfl_xor_sync(0xffffffffffffffffULL, d, off, 32);
        if (lane == 0) a.delta_out[(long long)bh * a.T + t] = d;
    }
}

// --- backward: dK/dV (kv tile basina blok) ----------------------------------

extern "C" __global__ void __launch_bounds__(NTH)
attn_bwd_dkv(AttnArgs a) {
    __shared__ __half2 Ks[D2 * BC];
    __shared__ __half2 Vs[D2 * BC];
    __shared__ __half2 Qm[(BR / 2) * HEAD_DIM];
    __shared__ __half2 dOm[(BR / 2) * HEAD_DIM];
    __shared__ __half2 bufA[D2 * BR];   // once Qs_d, sonra Ps^T
    __shared__ __half2 bufB[D2 * BR];   // once dOs_d, sonra dSs^T

    const int bh = blockIdx.y;
    const int b = bh / a.H, h = bh - b * a.H;
    const int bn = blockIdx.x * BC;
    const int tx = threadIdx.x & (NT - 1), ty = threadIdx.x / NT;
    const int qtiles = (a.T + BR - 1) / BR;

    load_pd(a.k + b * a.ksb + h * a.ksh + (long long)bn * a.kst,
            a.kst, a.T - bn, Ks, BC);
    load_pd(a.v + b * a.vsb + h * a.vsh + (long long)bn * a.vst,
            a.vst, a.T - bn, Vs, BC);
    __syncthreads();

    float accV[TM][TD], accK[TM][TD];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TD; j++) { accV[i][j] = 0.f; accK[i][j] = 0.f; }

    const int bi0 = a.causal ? (bn / BR) : 0;
    for (int bi = bi0; bi < qtiles; bi++) {
        const int bm = bi * BR;
        const int vq = a.T - bm;
        load_pd(a.q + b * a.qsb + h * a.qsh + (long long)bm * a.qst,
                a.qst, vq, bufA, BR);
        load_pd(a.dout + (long long)b * (a.T * a.H * HEAD_DIM) + h * HEAD_DIM
                + (long long)bm * (a.H * HEAD_DIM),
                a.H * HEAD_DIM, vq, bufB, BR);
        load_pr(a.q + b * a.qsb + h * a.qsh + (long long)bm * a.qst,
                a.qst, vq, Qm, BR);
        load_pr(a.dout + (long long)b * (a.T * a.H * HEAD_DIM) + h * HEAD_DIM
                + (long long)bm * (a.H * HEAD_DIM),
                a.H * HEAD_DIM, vq, dOm, BR);
        __syncthreads();

        // S^T[n][m] = K[n] . Q[m]   (n = TM*ty+i, m = TN*tx+j)
        // (kismi unroll: tam unroll yukleri erkene cekip VGPR tasmasi yapardi)
        float st[TM][TN], dpt[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) { st[i][j] = 0.f; dpt[i][j] = 0.f; }
        #pragma unroll 4
        for (int kp = 0; kp < D2; kp++) {
            __half2 a2[TM], b2[TN], c2[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) a2[i] = Ks[kp * BC + TM * ty + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) b2[j] = bufA[kp * BR + TN * tx + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    st[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], st[i][j], false);
            // dP^T[n][m] = V[n] . dO[m]
            #pragma unroll
            for (int i = 0; i < TM; i++) a2[i] = Vs[kp * BC + TM * ty + i];
            #pragma unroll
            for (int j = 0; j < TN; j++) c2[j] = bufB[kp * BR + TN * tx + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++)
                    dpt[i][j] = __builtin_amdgcn_fdot2(a2[i], c2[j], dpt[i][j], false);
        }

        // P^T ve dS^T'yi m-cifti paketli yaz (bufA/bufB'yi yeniden kullan —
        // once tum thread'lerin st/dpt okumasi bitsin diye sync sart)
        // lse/delta sadece mg'ye (yani tx'e) bagli: i dongusu disinda yukle.
        __syncthreads();
        float lv[TN], dl[TN];
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const int mg = bm + TN * tx + j;
            lv[j] = (mg < a.T) ? a.lse[(long long)bh * a.T + mg] : 0.f;
            dl[j] = (mg < a.T) ? a.delta[(long long)bh * a.T + mg] : 0.f;
        }
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int nl = TM * ty + i;
            const int ng = bn + nl;
            float p[TN], ds[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const int mg = bm + TN * tx + j;
                const bool ok = (mg < a.T) && (ng < a.T)
                                && (!a.causal || ng <= mg);
                float pv = 0.f;
                if (ok) pv = __expf(st[i][j] * a.scale - lv[j]);
                float dv = 0.f;
                if (ok) dv = pv * (dpt[i][j] - dl[j]) * a.scale;
                p[j] = pv; ds[j] = dv;
            }
            #pragma unroll
            for (int jj = 0; jj < TN / 2; jj++) {
                const int mp0 = (TN * tx) / 2 + jj;
                bufA[mp0 * BC + nl] =
                    __floats2half2_rn(p[2 * jj], p[2 * jj + 1]);
                bufB[mp0 * BC + nl] =
                    __floats2half2_rn(ds[2 * jj], ds[2 * jj + 1]);
            }
        }
        __syncthreads();

        // accV[n][d] += P^T[n][m] . dO[m][d]  (mp = m-cifti)
        // accK[n][d] += dS^T[n][m] . Q[m][d]
        #pragma unroll 4
        for (int mp = 0; mp < BR / 2; mp++) {
            __half2 a2[TM], c2[TM], b2[TD], e2[TD];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a2[i] = bufA[mp * BC + TM * ty + i];
                c2[i] = bufB[mp * BC + TM * ty + i];
            }
            #pragma unroll
            for (int j = 0; j < TD; j++) {
                b2[j] = dOm[mp * HEAD_DIM + TD * tx + j];
                e2[j] = Qm[mp * HEAD_DIM + TD * tx + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TD; j++) {
                    accV[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], accV[i][j], false);
                    accK[i][j] = __builtin_amdgcn_fdot2(c2[i], e2[j], accK[i][j], false);
                }
        }
        __syncthreads();
    }

    // dk/dv yaz (strided ciktilar)
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int ng = bn + TM * ty + i;
        if (ng >= a.T) continue;
        __half* dkr = a.dk + b * a.dkb + (long long)ng * a.dkt + h * a.dkh;
        __half* dvr = a.dv + b * a.dvb + (long long)ng * a.dvt + h * a.dvh;
        #pragma unroll
        for (int j = 0; j < TD; j++) {
            dkr[TD * tx + j] = __float2half(accK[i][j]);
            dvr[TD * tx + j] = __float2half(accV[i][j]);
        }
    }
}

// --- backward: dQ (q tile basina blok) --------------------------------------

extern "C" __global__ void __launch_bounds__(NTH)
attn_bwd_dq(AttnArgs a) {
    __shared__ __half2 Qs[D2 * BR];
    __shared__ __half2 dOs[D2 * BR];
    __shared__ __half2 Ks[D2 * BC];
    __shared__ __half2 Vs[D2 * BC];
    __shared__ __half2 Kn[(BC / 2) * HEAD_DIM];
    __shared__ __half2 dSs[(BC / 2) * BR];

    const int bh = blockIdx.y;
    const int b = bh / a.H, h = bh - b * a.H;
    const int bm = blockIdx.x * BR;
    const int tx = threadIdx.x & (NT - 1), ty = threadIdx.x / NT;

    load_pd(a.q + b * a.qsb + h * a.qsh + (long long)bm * a.qst,
            a.qst, a.T - bm, Qs, BR);
    load_pd(a.dout + (long long)b * (a.T * a.H * HEAD_DIM) + h * HEAD_DIM
            + (long long)bm * (a.H * HEAD_DIM),
            a.H * HEAD_DIM, a.T - bm, dOs, BR);
    __syncthreads();

    // bu blok icin lse/delta register'lara
    float lse_r[TM], del_r[TM];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int mg = bm + TM * ty + i;
        lse_r[i] = (mg < a.T) ? a.lse[(long long)bh * a.T + mg] : 0.f;
        del_r[i] = (mg < a.T) ? a.delta[(long long)bh * a.T + mg] : 0.f;
    }

    float acc[TM][TD];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TD; j++) acc[i][j] = 0.f;

    const int jlast = a.causal
        ? min(bm + BR, a.T) : a.T;    // jn0 < jlast
    for (int jn0 = 0; jn0 < jlast; jn0 += BC) {
        const int vk = a.T - jn0;
        load_pd(a.k + b * a.ksb + h * a.ksh + (long long)jn0 * a.kst,
                a.kst, vk, Ks, BC);
        load_pd(a.v + b * a.vsb + h * a.vsh + (long long)jn0 * a.vst,
                a.vst, vk, Vs, BC);
        load_pr(a.k + b * a.ksb + h * a.ksh + (long long)jn0 * a.kst,
                a.kst, vk, Kn, BC);
        __syncthreads();

        // S[m][n] = Q[m] . K[n] ; dP[m][n] = dO[m] . V[n]
        float s[TM][TN], dp[TM][TN];
        #pragma unroll
        for (int i = 0; i < TM; i++)
            #pragma unroll
            for (int j = 0; j < TN; j++) { s[i][j] = 0.f; dp[i][j] = 0.f; }
        #pragma unroll 4
        for (int kp = 0; kp < D2; kp++) {
            __half2 a2[TM], g2[TM], b2[TN], c2[TN];
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                a2[i] = Qs[kp * BR + TM * ty + i];
                g2[i] = dOs[kp * BR + TM * ty + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                b2[j] = Ks[kp * BC + TN * tx + j];
                c2[j] = Vs[kp * BC + TN * tx + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    s[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], s[i][j], false);
                    dp[i][j] = __builtin_amdgcn_fdot2(g2[i], c2[j], dp[i][j], false);
                }
        }
        // dS = P o (dP - Delta) * scale -> n-cifti paketli yaz
        // (ds bir kez hesaplanir; jj dongusu sadece paketler)
        #pragma unroll
        for (int i = 0; i < TM; i++) {
            const int ml = TM * ty + i;
            const int mg = bm + ml;
            float ds[TN];
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const int ng = jn0 + TN * tx + j;
                const bool ok = (mg < a.T) && (ng < a.T)
                                && (!a.causal || ng <= mg);
                float pv = 0.f;
                if (ok) pv = __expf(s[i][j] * a.scale - lse_r[i]);
                ds[j] = pv * (dp[i][j] - del_r[i]) * a.scale;
            }
            #pragma unroll
            for (int jj = 0; jj < TN / 2; jj++)
                dSs[(TN * tx / 2 + jj) * BR + ml] =
                    __floats2half2_rn(ds[2 * jj], ds[2 * jj + 1]);
        }
        __syncthreads();

        // acc[m][d] += dS[m][n] . K[n][d]  (np = n-cifti)
        #pragma unroll 4
        for (int np = 0; np < BC / 2; np++) {
            __half2 a2[TM], b2[TD];
            #pragma unroll
            for (int i = 0; i < TM; i++) a2[i] = dSs[np * BR + TM * ty + i];
            #pragma unroll
            for (int j = 0; j < TD; j++) b2[j] = Kn[np * HEAD_DIM + TD * tx + j];
            #pragma unroll
            for (int i = 0; i < TM; i++)
                #pragma unroll
                for (int j = 0; j < TD; j++)
                    acc[i][j] = __builtin_amdgcn_fdot2(a2[i], b2[j], acc[i][j], false);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const int mg = bm + TM * ty + i;
        if (mg >= a.T) continue;
        __half* dqr = a.dq + b * a.dqb + (long long)mg * a.dqt + h * a.dqh;
        #pragma unroll
        for (int j = 0; j < TD; j++)
            dqr[TD * tx + j] = __float2half(acc[i][j]);
    }
}
