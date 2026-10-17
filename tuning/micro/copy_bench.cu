// tuning/micro/copy_bench.cu - host->device copy rate from pinned memory by copy size and batching, on one stream:
// the prompt path streams ~23,000 expert blobs of ~1.7 MB per 8K prompt as one cudaMemcpyAsync each.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#define CK(x) do { auto e_ = (x); if (e_ != hipSuccess) { std::printf("%s failed: %d\n", #x, (int) e_); std::exit(1); } } while (0)
int main() {
    int nd = 0;
    CK(hipGetDeviceCount(&nd));
    for (int d = 0; d < nd; ++d) {
        hipDeviceProp_t pr{};
        CK(hipGetDeviceProperties(&pr, d));
        if (std::strncmp(pr.gcnArchName, "gfx1030", 7) == 0) { CK(hipSetDevice(d)); break; }
    }
    const size_t total = (size_t) 2 << 30;   // 2 GiB per measurement
    void *h, *dv;
    CK(hipHostMalloc(&h, total, 0));
    CK(hipMalloc(&dv, total));
    std::memset(h, 1, total);
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    hipEvent_t a, b;
    CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const size_t sizes[] = {256u << 10, 1u << 20, 1740800, 4u << 20, 16u << 20, 64u << 20, 256u << 20};
    std::printf("%12s %8s %10s %12s\n", "copy bytes", "copies", "GB/s", "us/copy");
    for (size_t sz : sizes) {
        const size_t n = total / sz;
        for (int rep = 0; rep < 2; ++rep) {
            CK(hipStreamSynchronize(s));
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < n; ++i)
                CK(hipMemcpyAsync((char*) dv + i * sz, (char*) h + i * sz, sz, hipMemcpyHostToDevice, s));
            CK(hipStreamSynchronize(s));
            const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (rep == 1) std::printf("%12zu %8zu %10.2f %12.1f\n", sz, n, n * sz / ms / 1e6, ms * 1000.0 / n);
        }
    }
    // the same 1.7 MB copies with an event record after every k-th (what the prompt path does per expert: k = 1),
    // and with a second stream waiting on each event (the compute stream's wait for its expert)
    hipStream_t s2;
    CK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    for (int wait = 0; wait < 2; ++wait)
        for (int k : {1, 2, 4, 8, 16}) {
            const size_t sz = 1740800, n = total / sz;
            hipEvent_t ev[64];
            for (auto& e : ev) CK(hipEventCreateWithFlags(&e, hipEventDisableTiming));
            CK(hipStreamSynchronize(s));
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t i = 0; i < n; ++i) {
                CK(hipMemcpyAsync((char*) dv + i * sz, (char*) h + i * sz, sz, hipMemcpyHostToDevice, s));
                if ((i + 1) % k == 0) {
                    CK(hipEventRecord(ev[(i / k) % 64], s));
                    if (wait) CK(hipStreamWaitEvent(s2, ev[(i / k) % 64], 0));
                }
            }
            CK(hipStreamSynchronize(s));
            CK(hipStreamSynchronize(s2));
            const float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
            std::printf("%12zu %8zu %10.2f %12.1f  (an event every %d copies%s)\n", sz, n, n * sz / ms / 1e6, ms * 1000.0 / n,
                        k, wait ? ", a stream waiting on each" : "");
            for (auto& e : ev) CK(hipEventDestroy(e));
        }
    return 0;
}
