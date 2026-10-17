// tuning/micro/skinny_bench.cu - the hyper-connections' down product (BF16 X [T, 10240] . W [320, 10240]^T -> FP32)
// with a dedicated kernel against a host reference.  Standalone 4.3 ms (12 TFLOPS; 64-row blocks 4.8 ms) against
// ~8 ms for the widened SGEMM route - but inside the engine it was no faster (32-row) or slower (64-row), so the engine
// keeps the widened route (tuning/LOG.md, "hc down kernel").  Kept here for a later look.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#define CK(x) do { auto e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)
constexpr int kSkinnyN = 320, kSkM = 32, kSkK = 32, kSkThreads = 256;
__global__ void __launch_bounds__(kSkThreads) skinny_bf16_kernel(const uint16_t* __restrict__ X,
                                                                 const uint16_t* __restrict__ W, float* __restrict__ Y,
                                                                 int64_t T, int64_t K, int64_t ldy, float beta) {
    __shared__ float xs[kSkK][kSkM];         // 4 KB
    __shared__ float ws[kSkK][kSkinnyN];     // 40 KB
    const int t = threadIdx.x, lane = t & 31, tm = t >> 5;   // rows 4 tm .. 4 tm + 3, columns lane + 32 j
    const int64_t m0 = (int64_t) blockIdx.x * kSkM;
    constexpr int WP = kSkinnyN * kSkK / 8, WR = (WP + kSkThreads - 1) / kSkThreads;   // W's uint4 pieces per tile
    uint4 wr[WR], xr = make_uint4(0, 0, 0, 0);
    auto load = [&](int64_t k0) {
#pragma unroll
        for (int r = 0; r < WR; ++r) {
            const int i = t + r * kSkThreads, n = i >> 2, kq = i & 3;
            if (i < WP) wr[r] = *reinterpret_cast<const uint4*>(W + (int64_t) n * K + k0 + kq * 8);
        }
        if (t < kSkM * 4) {
            const int64_t m = m0 + (t >> 2);
            xr = m < T ? *reinterpret_cast<const uint4*>(X + m * K + k0 + (t & 3) * 8) : make_uint4(0, 0, 0, 0);
        }
    };
    float acc[4][10];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 10; ++j) acc[i][j] = 0.0f;
    load(0);
    for (int64_t k0 = 0; k0 < K; k0 += kSkK) {
        __syncthreads();   // the previous tile is consumed
#pragma unroll
        for (int r = 0; r < WR; ++r) {
            const int i = t + r * kSkThreads, n = i >> 2, kq = i & 3;
            if (i >= WP) break;
            const uint32_t v[4] = {wr[r].x, wr[r].y, wr[r].z, wr[r].w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                ws[kq * 8 + 2 * e][n] = __uint_as_float(v[e] << 16);
                ws[kq * 8 + 2 * e + 1][n] = __uint_as_float(v[e] & 0xffff0000u);
            }
        }
        if (t < kSkM * 4) {
            const int m = t >> 2, kq = t & 3;
            const uint32_t v[4] = {xr.x, xr.y, xr.z, xr.w};
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                xs[kq * 8 + 2 * e][m] = __uint_as_float(v[e] << 16);
                xs[kq * 8 + 2 * e + 1][m] = __uint_as_float(v[e] & 0xffff0000u);
            }
        }
        __syncthreads();
        if (k0 + kSkK < K) load(k0 + kSkK);
#pragma unroll 4
        for (int k = 0; k < kSkK; ++k) {
            const float4 xv = *reinterpret_cast<const float4*>(&xs[k][4 * tm]);
#pragma unroll
            for (int j = 0; j < 10; ++j) {
                const float w = ws[k][lane + 32 * j];
                acc[0][j] = fmaf(xv.x, w, acc[0][j]);
                acc[1][j] = fmaf(xv.y, w, acc[1][j]);
                acc[2][j] = fmaf(xv.z, w, acc[2][j]);
                acc[3][j] = fmaf(xv.w, w, acc[3][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int64_t m = m0 + 4 * tm + i;
        if (m >= T) break;
#pragma unroll
        for (int j = 0; j < 10; ++j) {
            float* y = Y + m * ldy + lane + 32 * j;
            *y = beta == 0.0f ? acc[i][j] : fmaf(beta, *y, acc[i][j]);
        }
    }
}


int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 7952, K = 10240, N = kSkinnyN;
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    for (int d = 0; d < nd; ++d) {
        hipDeviceProp_t pr{};
        CK(hipGetDeviceProperties(&pr, d));
        if (std::strncmp(pr.gcnArchName, "gfx1030", 7) == 0) { CK(hipSetDevice(d)); break; }
    }
    std::vector<uint16_t> x(T * K), w(N * K);
    uint32_t h = 1;
    auto rnd = [&] { h = h * 1664525u + 1013904223u; float v = ((h >> 8) / 16777216.0f - 0.5f) * 0.2f; uint32_t b; std::memcpy(&b, &v, 4); return (uint16_t) (b >> 16); };
    for (auto& v : x) v = rnd();
    for (auto& v : w) v = rnd();
    uint16_t *dx, *dw; float* dy;
    CK(hipMalloc(&dx, x.size() * 2)); CK(hipMalloc(&dw, w.size() * 2)); CK(hipMalloc(&dy, T * N * 4));
    CK(hipMemcpy(dx, x.data(), x.size() * 2, hipMemcpyHostToDevice));
    CK(hipMemcpy(dw, w.data(), w.size() * 2, hipMemcpyHostToDevice));
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    auto go = [&] { skinny_bf16_kernel<<<(unsigned) ((T + kSkM - 1) / kSkM), kSkThreads>>>(dx, dw, dy, T, K, N, 0.0f); };
    go(); CK(hipGetLastError()); CK(hipDeviceSynchronize());
    CK(hipEventRecord(a));
    for (int i = 0; i < 10; ++i) go();
    CK(hipEventRecord(b)); CK(hipEventSynchronize(b));
    float ms = 0;
    CK(hipEventElapsedTime(&ms, a, b));
    ms /= 10;
    std::vector<float> y(T * N);
    CK(hipMemcpy(y.data(), dy, y.size() * 4, hipMemcpyDeviceToHost));
    double worst = 0;
    auto f = [](uint16_t v) { uint32_t b = (uint32_t) v << 16; float r; std::memcpy(&r, &b, 4); return r; };
    for (int64_t m = 0; m < T; m += 997)
        for (int64_t n = 0; n < N; n += 7) {
            double s = 0;
            for (int64_t k = 0; k < K; ++k) s += (double) f(x[m * K + k]) * f(w[n * K + k]);
            worst = std::max(worst, std::fabs(s - y[m * N + n]) / (std::fabs(s) + 1e-3));
        }
    std::printf("T %lld: %.3f ms, %.1f TFLOPS; worst relative error vs FP64 %.2g\n", (long long) T, ms,
                2.0 * T * N * K / ms / 1e9, worst);
    return 0;
}
