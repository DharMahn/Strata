// src/prefill/gemm.cu - see include/strata/prefill/gemm.hpp.
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
// The HIP compatibility shim maps CUDA shuffle spellings to Strata helpers.
// hipBLASLt's public headers declare native HIP shuffle functions, so keep
// those declarations from being macro-expanded in this translation unit.
#undef __shfl_xor_sync
#undef __shfl_down_sync
#undef __shfl_up_sync
#undef __shfl_sync
#undef __ballot_sync
#endif

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <memory>

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
#include "hipblaslt_tuning.hpp"
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <map>
#include <set>
#include <tuple>
#endif

namespace strata::prefill {
namespace {

void ck(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: %s: cuBLAS status %d\n", what, (int) s);
        std::exit(1);
    }
}

// dst[r * cols + c] = src[r * ld + c] as FP32, from BF16 or FP16 bits (exact either way)
__global__ void widen_kernel(const uint16_t* __restrict__ src, int64_t ld, float* __restrict__ dst, int64_t rows,
                             int64_t cols, bool bf) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    const int64_t r = i / cols, c = i - r * cols;
    const uint16_t v = src[r * ld + c];
    dst[i] = bf ? __uint_as_float((uint32_t) v << 16) : __half2float(__ushort_as_half(v));
}
void widen(const uint16_t* src, int64_t ld, float* dst, int64_t rows, int64_t cols, bool bf, cudaStream_t s) {
    const int64_t n = rows * cols;
    if (n > 0) widen_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(src, ld, dst, rows, cols, bf);
}

// RDNA2: rocBLAS ships tuned gfx1030 kernels only for FP16->FP16 and FP32 GEMMs.  The BF16->FP32 and FP16->FP32
// calls below get its generic fallback kernels: ~6 TFLOPS on the RX 6950 XT against ~22 for the tuned FP32 ones
// (tuning/micro/gemm_bench.cu; tuning/LOG.md, "rocBLAS on gfx1030").  There the inputs are widened to FP32 - exact
// for both - and multiplied by the tuned SGEMM.  STRATA_GEMM_WIDEN=0/1 overrides the architecture check.
// A product with a handful of outputs (the hyper-connections' inject, N 4; the shared expert's gate, N 1) is a read
// of X, not a GEMM: one wave per row, each lane 8 values per step, the N dot products summed over the wave.  On the
// widened route the SGEMM spent 4 ms per inject (X widened, then N = 4 tiles); this reads X once (~0.4 ms).
constexpr int kSmallN = 8;
__global__ void small_n_kernel(const uint16_t* __restrict__ X, const uint16_t* __restrict__ W, float* __restrict__ Y,
                               int64_t T, int N, int64_t K, int64_t ldy, float beta) {
    const int64_t row = (int64_t) blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= T) return;
    const uint16_t* x = X + row * K;
    float acc[kSmallN] = {};
    for (int64_t k = (int64_t) lane * 8; k < K; k += 256) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x + k);
        const uint32_t xs[4] = {xv.x, xv.y, xv.z, xv.w};
        float xf[8];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            xf[2 * i] = __uint_as_float(xs[i] << 16);
            xf[2 * i + 1] = __uint_as_float(xs[i] & 0xffff0000u);
        }
#pragma unroll
        for (int n = 0; n < kSmallN; ++n) {
            if (n >= N) break;
            const uint4 wv = *reinterpret_cast<const uint4*>(W + (int64_t) n * K + k);
            const uint32_t ws[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[n] = fmaf(xf[2 * i], __uint_as_float(ws[i] << 16), acc[n]);
                acc[n] = fmaf(xf[2 * i + 1], __uint_as_float(ws[i] & 0xffff0000u), acc[n]);
            }
        }
    }
#pragma unroll
    for (int n = 0; n < kSmallN; ++n) {
        if (n >= N) break;
        float v = acc[n];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) Y[row * ldy + n] = beta == 0.0f ? v : fmaf(beta, Y[row * ldy + n], v);
    }
}

bool widen_route() {
    static const bool on = [] {
        if (const char* v = std::getenv("STRATA_GEMM_WIDEN")) return std::atoi(v) != 0;
#if defined(__HIPCC__)
        int dev = 0;
        cudaDeviceProp p{};
        if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return std::strncmp(p.gcnArchName, "gfx103", 6) == 0;
#else
        return false;
#endif
    }();
    return on;
}

#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
struct HipLtCallKey {
    strata::prefill::hipblaslt::InputType type;
    int t;
    int n;
    int k;
    int ldy;
    uint32_t beta_bits;

    bool operator<(const HipLtCallKey& other) const {
        return std::tie(type, n, k, ldy, t, beta_bits) <
               std::tie(other.type, other.n, other.k, other.ldy, other.t, other.beta_bits);
    }
};

struct HipLtCachedAlgo {
    bool supported = false;
    hipblasLtMatmulAlgo_t algo{};
    size_t workspace_bytes = 0;
};

struct HipLtState {
    hipblasLtHandle_t handle = nullptr;
    void* workspace = nullptr;
    size_t workspace_bytes = 0;
    strata::prefill::hipblaslt::TuningTable table;
    std::map<HipLtCallKey, HipLtCachedAlgo> cache;
    uint64_t lt_launches = 0;
    uint64_t fallbacks = 0;
    std::set<std::tuple<strata::prefill::hipblaslt::InputType, int, int, int, int>> fallback_shapes;

    ~HipLtState() {
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: hipBLASLt summary launches=%llu fallbacks=%llu unique_fallback_shapes=%zu\n",
                         (unsigned long long) lt_launches, (unsigned long long) fallbacks, fallback_shapes.size());
            for (const auto& shape : fallback_shapes) {
                const auto type = std::get<0>(shape);
                std::fprintf(stderr, "prefill gemm: fallback shape dtype=%s T=%d N=%d K=%d ldy=%d\n",
                             type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16",
                             std::get<1>(shape), std::get<2>(shape), std::get<3>(shape), std::get<4>(shape));
            }
        }
        if (handle) hipblasLtDestroy(handle);
    }
};

struct HipLtDescriptors {
    hipblasLtMatmulDesc_t op = nullptr;
    hipblasLtMatrixLayout_t a = nullptr;
    hipblasLtMatrixLayout_t b = nullptr;
    hipblasLtMatrixLayout_t c = nullptr;

    ~HipLtDescriptors() {
        if (op) hipblasLtMatmulDescDestroy(op);
        if (a) hipblasLtMatrixLayoutDestroy(a);
        if (b) hipblasLtMatrixLayoutDestroy(b);
        if (c) hipblasLtMatrixLayoutDestroy(c);
    }

    bool init(hipDataType type, int t, int n, int k, int ldy) {
        const hipblasOperation_t trans_a = HIPBLAS_OP_T;
        const hipblasOperation_t trans_b = HIPBLAS_OP_N;
        if (hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)) !=
                HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&a, type, k, n, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&b, type, k, t, k) != HIPBLAS_STATUS_SUCCESS ||
            hipblasLtMatrixLayoutCreate(&c, HIP_R_32F, n, t, ldy) != HIPBLAS_STATUS_SUCCESS) {
            return false;
        }
        return true;
    }
};

std::unique_ptr<HipLtState> create_hipblaslt_state(void* workspace, size_t workspace_bytes) {
    const char* path = std::getenv("STRATA_HIPBLASLT_TUNING");
    if (!path || !*path) return nullptr;

    auto state = std::make_unique<HipLtState>();
    state->workspace = workspace;
    state->workspace_bytes = workspace_bytes;
    if (hipblasLtCreate(&state->handle) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt handle creation failed; using hipBLASEx\n");
        return nullptr;
    }

    int version = 0;
    if (hipblasLtGetVersion(state->handle, &version) != HIPBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "prefill gemm: hipBLASLt version query failed; using hipBLASEx\n");
        return nullptr;
    }
    int device = 0;
    hipDeviceProp_t properties{};
    if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&properties, device) != hipSuccess) {
        std::fprintf(stderr, "prefill gemm: HIP device query failed; using hipBLASEx\n");
        return nullptr;
    }
    std::string arch(properties.gcnArchName);
    const auto suffix = arch.find(':');
    if (suffix != std::string::npos) arch.resize(suffix);

    std::string error;
    if (!state->table.load(path, arch, version, error)) {
        std::fprintf(stderr, "prefill gemm: %s; using hipBLASEx\n", error.c_str());
        return nullptr;
    }
    std::fprintf(stderr, "prefill gemm: hipBLASLt tuning enabled (%zu rows, %s, version %d)\n",
                 state->table.rows().size(), arch.c_str(), version);
    return state;
}

HipLtCachedAlgo resolve_hipblaslt_algo(HipLtState& state, strata::prefill::hipblaslt::InputType type, int t,
                                       int n, int k, int ldy, float beta) {
    uint32_t beta_bits = 0;
    static_assert(sizeof(beta_bits) == sizeof(beta));
    std::memcpy(&beta_bits, &beta, sizeof(beta));
    const HipLtCallKey key{type, t, n, k, ldy, beta_bits};
    const auto cached = state.cache.find(key);
    if (cached != state.cache.end()) return cached->second;

    HipLtCachedAlgo resolved;
    const bool verbose = std::getenv("STRATA_HIPBLASLT_VERBOSE") != nullptr;
    const auto* row = state.table.closest(type, n, k, ldy, t);
    if (!row) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; no calibration for dtype=%s T=%d N=%d K=%d ldy=%d\n",
                         type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, t, n, k, ldy)) {
        return state.cache.emplace(key, resolved).first->second;
    }

    std::vector<int> solution_ids{row->solution_id};
    std::vector<hipblasLtMatmulHeuristicResult_t> candidates;
    if (hipblaslt_ext::getAlgosFromIndex(state.handle, solution_ids, candidates) != HIPBLAS_STATUS_SUCCESS ||
        candidates.empty() || candidates.front().state != HIPBLAS_STATUS_SUCCESS ||
        hipblaslt_ext::getIndexFromAlgo(candidates.front().algo) != row->solution_id) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d unavailable for T=%d N=%d K=%d ldy=%d\n",
                         row->solution_id, t, n, k, ldy);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    const float alpha = 1.0f;
    size_t required_workspace = 0;
    auto algo = candidates.front().algo;
    if (hipblaslt_ext::matmulIsAlgoSupported(state.handle, desc.op, &alpha, desc.a, desc.b, &beta, desc.c, desc.c,
                                             algo, required_workspace) != HIPBLAS_STATUS_SUCCESS) {
        if (verbose) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution %d rejects actual T=%d N=%d K=%d ldy=%d beta=%.9g\n",
                         row->solution_id, t, n, k, ldy, beta);
        }
        return state.cache.emplace(key, resolved).first->second;
    }

    resolved.supported = true;
    resolved.algo = algo;
    resolved.workspace_bytes = required_workspace;
    if (verbose) {
        std::fprintf(stderr,
                     "prefill gemm: Lt solution=%d dtype=%s T=%d N=%d K=%d ldy=%d beta=%.9g workspace=%zu\n",
                     row->solution_id, type == strata::prefill::hipblaslt::InputType::bf16 ? "bf16" : "f16", t, n,
                     k, ldy, beta, required_workspace);
    }
    return state.cache.emplace(key, resolved).first->second;
}

bool try_hipblaslt(void* opaque_state, strata::prefill::hipblaslt::InputType type, const uint16_t* x,
                   const uint16_t* w, float* y, int64_t t, int64_t n, int64_t k, int64_t ldy, float beta,
                   void* stream) {
    auto* state = static_cast<HipLtState*>(opaque_state);
    if (!state || t <= 0 || n <= 0 || k <= 0 || t > INT_MAX || n > INT_MAX || k > INT_MAX || ldy > INT_MAX ||
        ldy < n) {
        return false;
    }
    const auto resolved = resolve_hipblaslt_algo(*state, type, (int) t, (int) n, (int) k, (int) ldy, beta);
    if (!resolved.supported) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        return false;
    }
    if (resolved.workspace_bytes > state->workspace_bytes) {
        ++state->fallbacks;
        state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
        if (std::getenv("STRATA_HIPBLASLT_VERBOSE")) {
            std::fprintf(stderr, "prefill gemm: Lt fallback; solution needs %zu workspace bytes, have %zu\n",
                         resolved.workspace_bytes, state->workspace_bytes);
        }
        return false;
    }

    HipLtDescriptors desc;
    const hipDataType input_type = type == strata::prefill::hipblaslt::InputType::bf16 ? HIP_R_16BF : HIP_R_16F;
    if (!desc.init(input_type, (int) t, (int) n, (int) k, (int) ldy)) return false;
    const float alpha = 1.0f;
    const hipblasStatus_t status = hipblasLtMatmul(state->handle, desc.op, &alpha, w, desc.a, x, desc.b, &beta, y,
                                                   desc.c, y, desc.c, &resolved.algo, state->workspace,
                                                   state->workspace_bytes, (hipStream_t) stream);
    if (status == HIPBLAS_STATUS_SUCCESS) {
        ++state->lt_launches;
        return true;
    }

    std::fprintf(stderr, "prefill gemm: hipBLASLt launch failed with status %d\n", (int) status);
    if (beta != 0.0f) {
        std::fprintf(stderr, "prefill gemm: refusing a fallback after hipBLASLt failed with nonzero beta\n");
        std::exit(1);
    }
    auto* mutable_state = static_cast<HipLtState*>(opaque_state);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &beta, sizeof(beta_bits));
    auto cached = mutable_state->cache.find(HipLtCallKey{type, (int) t, (int) n, (int) k, (int) ldy, beta_bits});
    if (cached != mutable_state->cache.end()) cached->second.supported = false;
    ++mutable_state->fallbacks;
    mutable_state->fallback_shapes.emplace(type, (int) t, (int) n, (int) k, (int) ldy);
    return false;
}
#endif

}  // namespace

Gemm::~Gemm() {
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    delete static_cast<HipLtState*>(hipblaslt_state_);
#endif
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (!external_) {
        if (scratch_) cudaFree(scratch_);
        if (workspace_) cudaFree(workspace_);
    }
}

bool Gemm::init_external(void* stream, uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes,
                         std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    external_ = true;
    cublasSetStream(h, (cudaStream_t) stream);
    workspace_ = workspace;
    cublasSetWorkspace(h, workspace_, ws_bytes);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws_bytes).release();
#endif
    return true;
}

void Gemm::rebind(uint16_t* scratch, int64_t scratch_elems, void* workspace, size_t ws_bytes) {
    scratch_ = scratch;
    scratch_elems_ = scratch_elems;
    workspace_ = workspace;
    cublasSetWorkspace((cublasHandle_t) handle_, workspace_, ws_bytes);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (hipblaslt_state_) {
        auto* state = static_cast<HipLtState*>(hipblaslt_state_);
        state->workspace = workspace_;
        state->workspace_bytes = ws_bytes;
    }
#endif
}

bool Gemm::init(void* stream, int64_t scratch_elems, std::string& err) {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) { err = "prefill gemm: cublasCreate failed"; return false; }
    handle_ = h;
    stream_ = stream;
    cublasSetStream(h, (cudaStream_t) stream);
    // A fixed workspace so the handle never allocates on the way (and graphs could capture it later).
    const size_t ws = 32u << 20;
    if (cudaMalloc(&workspace_, ws) != cudaSuccess) { err = "prefill gemm: workspace"; return false; }
    cublasSetWorkspace(h, workspace_, ws);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    hipblaslt_state_ = create_hipblaslt_state(workspace_, ws).release();
#endif
    if (scratch_elems > 0 && cudaMalloc((void**) &scratch_, (size_t) scratch_elems * 2) != cudaSuccess) {
        err = "prefill gemm: dequant scratch of " + std::to_string(scratch_elems * 2 >> 20) + " MiB";
        return false;
    }
    scratch_elems_ = scratch_elems;
    return true;
}

// STRATA_GEMM_SMALL_N=0: the small-N kernel off (those products take the widened route) - the A/B
bool small_n_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_GEMM_SMALL_N"); return v == nullptr || std::atoi(v) != 0; }();
    return on;
}

// The widened product in tiles of the scratch `ws` (`ws_floats`): W tiles of [nc, kc] and X tiles of [rc, kc] in FP32,
// the K tiles summed into Y (beta 1 after the first).  Whole rows of W and X when they fit; K is split when a tile
// of whole X rows would be shorter than 2048 rows (the tuned SGEMM's rate drops below that: hc down, K 10240).
// false: not even a small tile fits (the caller takes the BLAS path).
bool Gemm::widened(const uint16_t* X, const uint16_t* W, bool bf, float* Y, int64_t T, int64_t N, int64_t K,
                   int64_t ldy, float beta, float* ws, int64_t ws_floats) {
    if (ws == nullptr || ws_floats <= 0) return false;
    const int64_t want = std::min<int64_t>(T, 2048);
    int64_t kc = K, nc = N, rc = 0;
    for (;;) {
        nc = std::min<int64_t>(N, std::max<int64_t>(64, (ws_floats / 2) / kc / 64 * 64));
        rc = std::min<int64_t>(T, (ws_floats - nc * kc) / kc);
        if (rc >= want || kc <= 512) break;
        kc = std::max<int64_t>(512, (kc / 2 + 255) / 256 * 256);
    }
    if (rc < std::min<int64_t>(T, 256) || nc * kc + rc * kc > ws_floats) return false;
    const cudaStream_t s = (cudaStream_t) stream_;
    float* wf = ws;
    float* xf = ws + nc * kc;
    const float alpha = 1.0f;
    for (int64_t n0 = 0; n0 < N; n0 += nc) {
        const int64_t nn = std::min(nc, N - n0);
        for (int64_t k0 = 0; k0 < K; k0 += kc) {
            const int64_t kk = std::min(kc, K - k0);
            const float b = k0 == 0 ? beta : 1.0f;
            widen(W + n0 * K + k0, K, wf, nn, kk, bf, s);
            for (int64_t r0 = 0; r0 < T; r0 += rc) {
                const int64_t rr = std::min(rc, T - r0);
                widen(X + r0 * K + k0, K, xf, rr, kk, bf, s);
                ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) nn, (int) rr, (int) kk, &alpha,
                                wf, CUDA_R_32F, (int) kk, xf, CUDA_R_32F, (int) kk, &b, Y + r0 * ldy + n0, CUDA_R_32F,
                                (int) ldy, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                   "cublasGemmEx widened");
            }
        }
    }
    return true;
}

void Gemm::bf16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
                float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (widen_route() && N <= kSmallN && K % 8 == 0 && ((uintptr_t) X | (uintptr_t) W) % 16 == 0 && small_n_on()) {
        small_n_kernel<<<(unsigned) ((T + 7) / 8), 256, 0, (cudaStream_t) stream_>>>(X, W, Y, T, (int) N, K, ldy, beta);
        return;
    }
    if (widen_route() && widened(X, W, true, Y, T, N, K, ldy, beta, (float*) scratch_, scratch_elems_ / 2)) return;
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::bf16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
    // Column-major view: Y^T[N, T] = W[N, K] (stored K x N col-major, transposed) . X^T[K, T].
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16BF, (int) K, X, CUDA_R_16BF, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx");
}

void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N, int64_t K, int64_t ldy,
               float beta) {
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    if (widen_route()) {
        // a W that `native` dequantized into the scratch: the tiles go after it
        const bool in_scratch = W >= scratch_ && W < scratch_ + scratch_elems_;
        const int64_t used = in_scratch ? ((W - scratch_) + N * K + 1) / 2 * 2 : 0;   // FP16 elements, kept even
        if (widened(X, W, false, Y, T, N, K, ldy, beta, (float*) (scratch_ + used), (scratch_elems_ - used) / 2)) return;
    }
    const float alpha = 1.0f;
#if defined(__HIPCC__) && defined(STRATA_HIPBLASLT_AVAILABLE)
    if (try_hipblaslt(hipblaslt_state_, strata::prefill::hipblaslt::InputType::f16, X, W, Y, T, N, K, ldy,
                      beta, stream_)) {
        return;
    }
#endif
    ck(cublasGemmEx((cublasHandle_t) handle_, CUBLAS_OP_T, CUBLAS_OP_N, (int) N, (int) T, (int) K, &alpha, W,
                    CUDA_R_16F, (int) K, X, CUDA_R_16F, (int) K, &beta, Y, CUDA_R_32F, (int) ldy,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
       "cublasGemmEx f16");
}

void Gemm::native(const uint16_t* X, int ggml_type, const void* W_blocks, float* Y, int64_t T, int64_t N, int64_t K,
                  int64_t ldy, float beta) {
    if (N * K > scratch_elems_) {
        // Too large for the scratch at once: in row slices.
        const int64_t rows = scratch_elems_ / K;
        if (rows <= 0) { std::fprintf(stderr, "prefill gemm: scratch too small for K=%lld\n", (long long) K); std::exit(1); }
        if (ldy <= 0) ldy = N;
        for (int64_t r0 = 0; r0 < N; r0 += rows) {
            const int64_t n = (N - r0 < rows) ? N - r0 : rows;
            strata::kernels::dequant_f16(ggml_type, W_blocks, r0, n, K, scratch_, stream_);
            f16(X, scratch_, Y + r0, T, n, K, ldy, beta);
        }
        return;
    }
    strata::kernels::dequant_f16(ggml_type, W_blocks, 0, N, K, scratch_, stream_);
    f16(X, scratch_, Y, T, N, K, ldy, beta);
}

}  // namespace strata::prefill
