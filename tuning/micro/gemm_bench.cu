// tuning/micro/gemm_bench.cu - the prompt path's BLAS shapes on this GPU: Y[T, N] = X[T, K] . W[N, K]^T, FP32
// compute.  rocBLAS ships tuned gfx1030 kernels only for HH (FP16 in and out) and SS (FP32); the engine's BF16->FP32
// (BS) and FP16->FP32 (HS) calls get the generic fallback kernels.  Columns: BS, HS (the engine's calls), SS (BF16
// widened to FP32 - exact), HH (FP16 out).  Prints ms and TFLOPS.
//   gemm_bench.exe [T]
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(x) do { auto e_ = (x); if (e_ != hipSuccess) { std::printf("%s failed: %d\n", #x, (int) e_); std::exit(1); } } while (0)
#define CB(x) do { auto e_ = (x); if (e_ != HIPBLAS_STATUS_SUCCESS) { std::printf("%s failed: %d\n", #x, (int) e_); std::exit(1); } } while (0)

__global__ void widen(const uint16_t* a, float* b, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __uint_as_float((uint32_t) a[i] << 16);
}
__global__ void fill(uint16_t* p, size_t n, uint32_t seed, bool bf) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t h = (uint32_t) i * 2654435761u ^ seed;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    float v = ((h & 0xffff) / 65536.0f - 0.5f) * 0.1f;
    if (bf) { uint32_t b = __float_as_uint(v); p[i] = (uint16_t) (b >> 16); }
    else { __half hv = __float2half(v); p[i] = *(uint16_t*) &hv; }
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? std::atoi(argv[1]) : 8192;
    struct Shape { const char* name; int N, K, ldy; };
    const Shape shapes[] = {
        {"hc down (10240->320)", 320, 10240, 320},   {"hc up (320->10240)", 10240, 320, 10240},
        {"router (2560->512)", 512, 2560, 512},      {"ssm alpha (2560->48)", 48, 2560, 96},
        {"indexer q (2560->512)", 512, 2560, 512},   {"ple key (2560->10240)", 10240, 2560, 10240},
        {"gdn qkv (2560->10240)", 10240, 2560, 10240}, {"gdn gate (2560->6144)", 6144, 2560, 6144},
        {"ssm out (6144->2560)", 2560, 6144, 2560},  {"attn q (2560->12288)", 12288, 2560, 12288},
    };
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    for (int d = 0; d < nd; ++d) {
        hipDeviceProp_t pr{};
        CK(hipGetDeviceProperties(&pr, d));
        if (std::strncmp(pr.gcnArchName, "gfx1030", 7) == 0) { CK(hipSetDevice(d)); std::printf("device %d: %s\n", d, pr.name); break; }
    }
    hipblasHandle_t h;
    CB(hipblasCreate(&h));
    size_t maxX = 0, maxW = 0, maxY = 0;
    for (auto& s : shapes) {
        maxX = std::max(maxX, (size_t) T * s.K); maxW = std::max(maxW, (size_t) s.N * s.K);
        maxY = std::max(maxY, (size_t) T * s.ldy);
    }
    uint16_t *X, *W;
    float *Y, *X32, *W32;
    CK(hipMalloc(&X, maxX * 2)); CK(hipMalloc(&W, maxW * 2)); CK(hipMalloc(&Y, maxY * 4));
    CK(hipMalloc(&X32, maxX * 4)); CK(hipMalloc(&W32, maxW * 4));
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const float alpha = 1.0f, beta = 0.0f;
    std::printf("T = %d, FP32 compute\n%-24s %16s %16s %16s %16s\n", T, "shape", "BS ms / TFLOPS", "HS ms / TFLOPS",
                "SS ms / TFLOPS", "HH ms / TFLOPS");
    auto timed = [&](auto&& go) {
        for (int i = 0; i < 3; ++i) go();
        CK(hipDeviceSynchronize());
        const int reps = 10;
        CK(hipEventRecord(a));
        for (int i = 0; i < reps; ++i) go();
        CK(hipEventRecord(b));
        CK(hipEventSynchronize(b));
        float t = 0;
        CK(hipEventElapsedTime(&t, a, b));
        return (double) t / reps;
    };
    for (auto& s : shapes) {
        const size_t nx = (size_t) T * s.K, nw = (size_t) s.N * s.K;
        double ms[4];
        for (int v = 0; v < 4; ++v) {   // 0 BS, 1 HS, 2 SS, 3 HH
            const bool bf = v == 0 || v == 2;
            fill<<<(unsigned) ((nx + 255) / 256), 256>>>(X, nx, 1, bf);
            fill<<<(unsigned) ((nw + 255) / 256), 256>>>(W, nw, 2, bf);
            if (v == 2) {
                widen<<<(unsigned) ((nx + 255) / 256), 256>>>(X, X32, nx);
                widen<<<(unsigned) ((nw + 255) / 256), 256>>>(W, W32, nw);
            }
            const hipDataType ti = v == 0 ? HIP_R_16BF : v == 2 ? HIP_R_32F : HIP_R_16F;
            const hipDataType to = v == 3 ? HIP_R_16F : HIP_R_32F;
            const void* pw = v == 2 ? (const void*) W32 : (const void*) W;
            const void* px = v == 2 ? (const void*) X32 : (const void*) X;
            ms[v] = timed([&] {
                CB(hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, s.N, T, s.K, &alpha, pw, ti, s.K, px, ti, s.K, &beta, Y,
                                 to, s.ldy, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
            });
        }
        const double fl = 2.0 * T * s.N * s.K;
        std::printf("%-24s", s.name);
        for (int v = 0; v < 4; ++v) std::printf(" %8.3f / %5.1f", ms[v], fl / ms[v] / 1e9);
        std::printf("\n");
    }
    return 0;
}
