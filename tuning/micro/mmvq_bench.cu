// tuning/micro/mmvq_bench.cu - the decode GEMVs (native_mmvq) on their real shapes, timed back to back.
//
//     tuning\micro\build.bat && tuning\micro\mmvq_bench.exe [reps]
//
// Weights and activations are random bytes: the kernels have no data-dependent control flow, so the timing is the
// real one; the numbers they compute are meaningless.  Prints us per call and the weight bytes / time rate.
#include "strata/kernels/native_mmvq.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

struct Shape { int type; int n_in, n_out; const char* name; };

static const Shape kShapes[] = {
    {23, 2560, 10240, "GDN attn_qkv IQ4_XS"},
    {23, 2560, 6144, "GDN attn_gate IQ4_XS"},
    {21, 2560, 6144, "GDN attn_gate IQ3_S"},
    {23, 6144, 2560, "GDN ssm_out IQ4_XS"},
    {23, 2560, 12288, "QSA attn_q IQ4_XS"},
    {23, 2560, 640, "shexp gate/up IQ4_XS"},
    {20, 640, 2560, "shexp down IQ4_NL"},
    {23, 2560, 248320, "LM head IQ4_XS"},
    // the MTP draft layer (mtp/rt dense.bin: Q8_0) and its draft-vocabulary head
    {8, 2560, 12288, "MTP q_proj Q8_0"},
    {8, 6144, 2560, "MTP o_proj Q8_0"},
    {8, 2560, 2560, "MTP fc_embedding Q8_0"},
    {8, 2560, 640, "MTP shexp gate Q8_0"},
    {8, 640, 2560, "MTP shexp down Q8_0"},
    {23, 2560, 106299, "MTP draft head IQ4_XS"},
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::atoi(argv[1]) : 50;
    const int only = argc > 2 ? std::atoi(argv[2]) : -1;
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    std::mt19937 rng(42);
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    std::printf("%-24s %5s %10s %10s %9s\n", "shape", "ncols", "MB", "us/call", "GB/s");
    for (int si = 0; si < (int) (sizeof kShapes / sizeof kShapes[0]); ++si) {
        if (only >= 0 && si != only) continue;
        const Shape& sh = kShapes[si];
        const size_t wb = strata::kernels::native_mmvq_weight_bytes(sh.type, sh.n_in, sh.n_out);
        const size_t xb = strata::kernels::native_q8_1_bytes(sh.n_in, 8);
        std::vector<uint8_t> h(wb > xb ? wb : xb);
        for (auto& v : h) v = (uint8_t) rng();
        // tame the fp16 scales a little: every other byte's top bits cleared keeps most halves finite
        for (size_t i = 1; i < h.size(); i += 2) h[i] &= 0x3b;
        void *w, *x;
        float* y;
        CK(cudaMalloc(&w, wb));
        CK(cudaMalloc(&x, xb));
        CK(cudaMalloc((void**) &y, (size_t) sh.n_out * 8 * sizeof(float)));
        CK(cudaMemcpy(w, h.data(), wb, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(x, h.data(), xb, cudaMemcpyHostToDevice));
        for (int nc = 1; nc <= 6; ++nc) {
            for (int i = 0; i < 5; ++i) strata::kernels::native_mmvq(sh.type, w, x, y, sh.n_in, sh.n_out, nc, s);
            CK(cudaStreamSynchronize(s));
            CK(cudaEventRecord(a, s));
            for (int i = 0; i < reps; ++i) strata::kernels::native_mmvq(sh.type, w, x, y, sh.n_in, sh.n_out, nc, s);
            CK(cudaEventRecord(b, s));
            CK(cudaEventSynchronize(b));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, a, b));
            const double us = 1000.0 * ms / reps;
            // FNV-1a of the output bits: two builds of the kernels must print the same value (bit-exactness)
            std::vector<uint32_t> hy((size_t) sh.n_out * nc);
            CK(cudaMemcpy(hy.data(), y, hy.size() * 4, cudaMemcpyDeviceToHost));
            uint64_t fnv = 1469598103934665603ull;
            for (uint32_t v : hy) { fnv ^= v; fnv *= 1099511628211ull; }
            std::printf("%-24s %5d %10.2f %10.1f %9.1f  %016llx\n", sh.name, nc, wb / 1e6, us, wb / (us * 1e3),
                        (unsigned long long) fnv);
        }
        CK(cudaFree(w));
        CK(cudaFree(x));
        CK(cudaFree(y));
    }
    return 0;
}
