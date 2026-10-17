// tuning/micro/attn_bench.cu - the prompt path's QSA attention on this GPU: qsa_prompt_attn_batch (on HIP the RDNA
// FP32 kernel in src/kernels/cuda/qsa_prompt_attn.cu) against the decode kernel 32 queries at a time (what the prompt
// path ran before), on a synthetic 8K prompt: int8 KV in shuffled pages (1% not resident), each query selecting
// min(position + 1, 2048) cells.  Prints the output difference and both times.
//   attn_bench.exe [n_queries]
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "strata/kernels/kv_q8.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;
#define CK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { std::printf("%s: %s\n", #x, cudaGetErrorString(e_)); std::exit(1); } } while (0)

template <typename T> T* up(const std::vector<T>& h) {
    T* d = nullptr;
    CK(cudaMalloc(&d, h.size() * sizeof(T)));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    const int NQ = argc > 1 ? std::atoi(argv[1]) : 7952;
    int nd = 0;
    CK(cudaGetDeviceCount(&nd));
    for (int d = 0; d < nd; ++d) {
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, d));
        if (std::strncmp(pr.gcnArchName, "gfx1030", 7) == 0) { CK(cudaSetDevice(d)); break; }
    }
    {
        int cur = -1;
        CK(cudaGetDevice(&cur));
        cudaDeviceProp pr{};
        CK(cudaGetDeviceProperties(&pr, cur));
        std::printf("%d devices, on %d: %s (%s)\n", nd, cur, pr.name, pr.gcnArchName);
    }
    k::QsaShapes s = k::qsa_real_shapes();
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size, NH = (int) s.n_head;
    const int cells = (NQ + P - 1) / P * P, pages = cells / P, cap = 2048;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd01(0.0f, 1.0f);
    std::uniform_int_distribution<int> code(-127, 127);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (int32_t) (((int64_t) i * 7919 + 13) % pages);
    for (int i = 0; i < pages; i += 100) table[i] = -1;   // not resident: masked
    const size_t rows = (size_t) pages * H * P;
    std::vector<int8_t> kq(rows * D), vq(rows * D);
    for (auto& x : kq) x = (int8_t) code(rng);
    for (auto& x : vq) x = (int8_t) code(rng);
    std::vector<uint16_t> ks(rows * (D / k::KV_Q8_GROUP)), vs(rows * (D / k::KV_Q8_GROUP));
    for (auto& x : ks) { __half h = __float2half(sc(rng)); std::memcpy(&x, &h, 2); }
    for (auto& x : vs) { __half h = __float2half(sc(rng)); std::memcpy(&x, &h, 2); }
    std::vector<float> q((size_t) NQ * NH * D);
    for (auto& x : q) x = nd01(rng) * 2.0f;
    std::vector<int32_t> ids((size_t) NQ * cap, 0), steps((size_t) NQ * k::kStepCount, 0);
    std::vector<int32_t> pool;
    for (int i = 0; i < NQ; ++i) {
        const int n = std::min(i + 1, cap);
        if (i + 1 <= cap) {
            for (int c = 0; c < n; ++c) ids[(size_t) i * cap + c] = c;
        } else {
            pool.resize(i + 1);
            for (int c = 0; c <= i; ++c) pool[c] = c;
            for (int c = 0; c < n; ++c) std::swap(pool[c], pool[c + rng() % (i + 1 - c)]);
            std::sort(pool.begin(), pool.begin() + n);
            std::copy(pool.begin(), pool.begin() + n, ids.begin() + (size_t) i * cap);
        }
        steps[(size_t) i * k::kStepCount + k::kStepWidth] = n;
    }
    k::QsaAttnPools pools;
    pools.k_q = up(kq); pools.v_q = up(vq); pools.k_scale = up(ks); pools.v_scale = up(vs);
    pools.page_table = up(table);
    const float* dq = up(q);
    const int32_t* dids = up(ids);
    const int32_t* dsteps = up(steps);
    float *ref = nullptr, *out = nullptr, *scratch = nullptr;
    CK(cudaMalloc(&ref, q.size() * 4)); CK(cudaMalloc(&out, q.size() * 4));
    const int B = 32;
    CK(cudaMalloc(&scratch, (size_t) B * k::qsa_decode_attn_scratch_floats(cap, s) * 4));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    auto old_path = [&] {
        for (int q0 = 0; q0 < NQ; q0 += B) {
            const int nb = std::min(B, NQ - q0);
            k::qsa_decode_attn_batch(dq + (size_t) q0 * NH * D, pools, dids + (size_t) q0 * cap,
                                     dsteps + (size_t) q0 * k::kStepCount, cap, s, scratch, ref + (size_t) q0 * NH * D,
                                     nb, nullptr);
        }
    };
    bool ok = true;
    auto new_path = [&] { ok = k::qsa_prompt_attn_batch(dq, pools, dids, dsteps, cap, s, out, NQ, nullptr); };
    float ms_old = 0, ms_new = 0;
    for (int rep = 0; rep < 3; ++rep) {
        float t = 0;
        CK(cudaEventRecord(a)); old_path(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&t, a, b)); if (rep) ms_old += t / 2;
        CK(cudaEventRecord(a)); new_path(); CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&t, a, b)); if (rep) ms_new += t / 2;
    }
    if (!ok) { std::printf("qsa_prompt_attn_batch declined (no kernel for this build/device)\n"); return 1; }
    std::vector<float> hr(q.size()), ho(q.size());
    CK(cudaMemcpy(hr.data(), ref, hr.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ho.data(), out, ho.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0, sum_rel = 0, maxabs = 0;
    int bad = 0;
    for (int i = 0; i < NQ; ++i) {
        double e2 = 0, r2 = 0;
        for (size_t j = (size_t) i * NH * D; j < (size_t) (i + 1) * NH * D; ++j) {
            if (!std::isfinite(ho[j])) ++bad;
            e2 += (ho[j] - hr[j]) * (double) (ho[j] - hr[j]);
            r2 += (double) hr[j] * hr[j];
            maxabs = std::max(maxabs, (double) std::fabs(ho[j] - hr[j]));
        }
        const double rel = r2 > 0 ? std::sqrt(e2 / r2) : std::sqrt(e2);
        worst = std::max(worst, rel);
        sum_rel += rel;
    }
    std::printf("%d queries, cap %d: old (decode kernel, 32 at a time) %.2f ms, new %.2f ms (%.2fx)\n", NQ, cap, ms_old,
                ms_new, ms_old / ms_new);
    std::printf("output vs old: relative L2 per query mean %.3g, worst %.3g; max abs diff %.3g; non-finite %d\n",
                sum_rel / NQ, worst, maxabs, bad);
    return 0;
}
