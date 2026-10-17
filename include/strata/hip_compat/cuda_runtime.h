#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA headers.
#include <hip/hip_runtime.h>
// Do not let HIP's legacy macro corrupt libstdc++ attribute names.
#ifdef __noinline__
#undef __noinline__
#endif
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaDevAttrMultiProcessorCount hipDeviceAttributeMultiprocessorCount
#define cudaDevAttrClockRate hipDeviceAttributeClockRate
#define cudaDevAttrComputeCapabilityMajor hipDeviceAttributeComputeCapabilityMajor
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize ::strata::hip_compat::device_sync
#define cudaDriverGetVersion hipDriverGetVersion
#define cudaErrorNotReady hipErrorNotReady
#define cudaErrorStreamCaptureUnsupported hipErrorStreamCaptureUnsupported
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree ::strata::hip_compat::free_dev
#define cudaFreeHost ::strata::hip_compat::free_host
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaGraphExec_t hipGraphExec_t
#define cudaGraphGetNodes hipGraphGetNodes
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphUpload hipGraphUpload
#define cudaGraph_t hipGraph_t
#define cudaHostAlloc(...) (::strata::hip_compat::host_alloc(__VA_ARGS__))
#define cudaHostAllocDefault hipHostAllocDefault
#define cudaHostAllocMapped hipHostAllocMapped
#define cudaHostAllocPortable hipHostAllocPortable
#define cudaHostGetDevicePointer hipHostGetDevicePointer
#define cudaHostRegister ::strata::hip_compat::host_register
#define cudaHostRegisterMapped hipHostRegisterMapped
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostUnregister ::strata::hip_compat::host_unregister
#define cudaLaunchHostFunc hipLaunchHostFunc
#define cudaMalloc(...) (::strata::hip_compat::malloc_dev(__VA_ARGS__))
#define cudaMallocHost(...) (::strata::hip_compat::malloc_host(__VA_ARGS__))
#define cudaMemcpy hipMemcpy
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpyDefault hipMemcpyDefault
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaPeekAtLastError hipPeekAtLastError
#define cudaRuntimeGetVersion hipRuntimeGetVersion
#define cudaStreamBeginCapture ::strata::hip_compat::begin_capture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaStreamCaptureStatus hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamDestroy ::strata::hip_compat::stream_destroy
#define cudaStreamEndCapture ::strata::hip_compat::end_capture
#define cudaStreamIsCapturing hipStreamIsCapturing
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamQuery hipStreamQuery
#define cudaStreamSynchronize ::strata::hip_compat::stream_sync
#define cudaStreamWaitEvent hipStreamWaitEvent
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

#include <cstdio>
#include <cstdlib>
#include <cstring>
namespace strata::platform {
int guard_address_range(const void* owner, const void* p, unsigned long long n);
void unguard_address_range(const void* owner);
}
namespace strata::hip_compat {
// THE ALIAS GUARD (the "!!!" bug, tuning/LOG.md 2026-10-03).  On Windows the HIP runtime's DEVICE addresses - a
// hipMalloc's, and the device alias of mapped host memory (hipHostRegister / hipHostMalloc) - are plain numbers that
// nothing reserves in the process's own address space.  The OS could therefore give the same numbers to an ordinary
// allocation, and did: a 112 MiB conversation snapshot buffer landed inside the device alias of the expert arena's
// third registered slice.  HIP classifies every hipMemcpy pointer by looking it up among its memory objects, whatever
// `kind` says, so the snapshot's "device to host" copy was done as a device-to-device copy INTO THE ARENA: 63 experts
// of layer 2 were overwritten in RAM (the CPU pool then computed NaN from them: "!!!"), and the snapshot buffer got
// nothing.  The same collision made read-backs come out as zeros and the start-up slot check fail.  About one start
// in four with a small expert cache; which buffer and which slice depended on where the heap happened to be.
// So every device range the engine learns of is reserved in the host address space (no memory is committed) for as
// long as the object lives.  A range that already holds committed host memory is a collision that exists: a mapped
// host allocation is then made again, a registration is refused (the caller keeps that memory unregistered).
// STRATA_ALIAS_GUARD=0 turns it off.
inline bool alias_guard_on() {
#if defined(_WIN32)
    static const bool v = [] {
        const char* e = std::getenv("STRATA_ALIAS_GUARD");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
#else
    return false;
#endif
}
// the device alias of mapped host memory `h`: 1 guarded (or no separate alias), -1 it collides with host memory
inline int guard_host_alias(void* h, size_t bytes) {
    void* d = nullptr;
    if (hipHostGetDevicePointer(&d, h, 0) != hipSuccess) { (void) hipGetLastError(); return 1; }
    if (d == nullptr || d == h) return 1;
    return ::strata::platform::guard_address_range(h, d, bytes);
}
inline hipError_t free_dev(void* p) {
    const hipError_t e = hipFree(p);
    if (p != nullptr && alias_guard_on()) ::strata::platform::unguard_address_range(p);
    return e;
}
inline hipError_t free_host(void* p) {
    const hipError_t e = hipFreeHost(p);
    if (p != nullptr && alias_guard_on()) ::strata::platform::unguard_address_range(p);
    return e;
}
inline hipError_t host_register(void* p, size_t bytes, unsigned int flags) {
    const hipError_t e = hipHostRegister(p, bytes, flags);
    if (e != hipSuccess || !alias_guard_on() || bytes == 0) return e;
    if (guard_host_alias(p, bytes) >= 0) return e;
    std::fprintf(stderr, "strata hip: a host registration of %zu MiB is refused: host memory already lives at its device "
                         "alias (see the alias guard in hip_compat/cuda_runtime.h)\n", bytes >> 20);
    ::strata::platform::unguard_address_range(p);
    (void) hipHostUnregister(p);
    (void) hipGetLastError();
    return hipErrorOutOfMemory;
}
inline hipError_t host_unregister(void* p) {
    const hipError_t e = hipHostUnregister(p);
    if (p != nullptr && alias_guard_on()) ::strata::platform::unguard_address_range(p);
    return e;
}
// debug (STRATA_DBG_POISON=1, the "!!!" hunt): every new device and pinned host buffer starts as 0xff bytes - NaN as
// floats - so a read of a buffer before anything wrote it fails every time instead of on whatever it held before
inline bool dbg_poison() {
    static const bool v = std::getenv("STRATA_DBG_POISON") != nullptr;
    return v;
}
template <typename T>
inline hipError_t malloc_dev(T** pointer, size_t bytes) {
    const hipError_t e = hipMalloc(reinterpret_cast<void**>(pointer), bytes);
    if (e == hipSuccess && bytes > 0 && alias_guard_on()) {   // (committed memory there already: nothing to be done)
        void* const p = *reinterpret_cast<void**>(pointer);
        (void) ::strata::platform::guard_address_range(p, p, bytes);
    }
    if (e == hipSuccess && bytes > 0 && dbg_poison()) {
        (void) hipMemset(*reinterpret_cast<void**>(pointer), 0xff, bytes);
        (void) hipDeviceSynchronize();
    }
    return e;
}
template <typename T>
inline hipError_t host_alloc(T** pointer, size_t bytes, unsigned int flags) {
    void** const pp = reinterpret_cast<void**>(pointer);
    const hipError_t e = hipHostMalloc(pp, bytes, flags);
    if (e == hipSuccess && bytes > 0 && alias_guard_on()) {
        // an alias that collides with host memory: allocate again (the colliding ones are held until one is clear)
        void* bad[8];
        int nbad = 0;
        while (guard_host_alias(*pp, bytes) < 0) {
            ::strata::platform::unguard_address_range(*pp);
            void* again = nullptr;
            if (nbad == 8 || hipHostMalloc(&again, bytes, flags) != hipSuccess) {
                (void) hipGetLastError();
                std::fprintf(stderr, "strata hip: a mapped host buffer of %zu KiB keeps a device alias that host memory "
                                     "already lives at\n", bytes >> 10);
                break;
            }
            bad[nbad++] = *pp;
            *pp = again;
        }
        for (int i = 0; i < nbad; ++i) (void) hipFreeHost(bad[i]);
    }
    if (e == hipSuccess && bytes > 0 && dbg_poison()) std::memset(*pp, 0xff, bytes);
    return e;
}
template <typename T>
inline hipError_t malloc_host(T** pointer, size_t bytes) {
    return host_alloc(pointer, bytes, hipHostMallocDefault);
}
}  // namespace strata::hip_compat

// The GPUs the engine can run on, numbered from 0 - what HIP_VISIBLE_DEVICES does on Linux.  A PC with a CPU-
// integrated Radeon lists that as HIP device 0 (Ryzen 7000/9000 desktops), before the real card, and on Windows
// HIP_VISIBLE_DEVICES cannot hide it (any index other than 0 makes the runtime list no device).  The engine's own
// code means "the main GPU" by device 0 (cudaSetDevice(0), OnDevice(0), a stream created under the default
// device), so every device-ordinal API below speaks in these LOGICAL numbers: logical 0 is the first GPU whose
// architecture the binary has code for, and the others are not visible at all.  STRATA_GPU=<n> (a HIP ordinal)
// picks the one GPU by hand.  A PC where every GPU is usable keeps HIP's own numbering.
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
namespace strata::hip_compat {
inline bool arch_compiled(const char* gcn_arch_name) {
#if defined(STRATA_HIP_ARCHS)
    const std::string arch(gcn_arch_name, std::strcspn(gcn_arch_name, ":"));
    const std::string list = std::string(",") + STRATA_HIP_ARCHS + ",";
    return list.find("," + arch + ",") != std::string::npos;
#else
    (void) gcn_arch_name;
    return true;
#endif
}
struct VisibleGpus {
    std::vector<int> phys;    // logical -> HIP ordinal
    bool active = false;      // false: HIP's numbering is used as it is
};
inline const VisibleGpus& visible_gpus() {
    static const VisibleGpus v = [] {
        VisibleGpus r;
#if defined(STRATA_HIP_ARCHS)
        int count = 0;
        if (hipGetDeviceCount(&count) != hipSuccess || count <= 0) { (void) hipGetLastError(); return r; }
        if (const char* forced = std::getenv("STRATA_GPU"); forced != nullptr && forced[0] != '\0') {
            const int n = std::atoi(forced);
            if (n >= 0 && n < count) r.phys.push_back(n);
        } else {
            for (int i = 0; i < count; ++i) {
                hipDeviceProp_t p{};
                if (hipGetDeviceProperties(&p, i) == hipSuccess && arch_compiled(p.gcnArchName) && p.warpSize == 32)
                    r.phys.push_back(i);
            }
        }
        r.active = !r.phys.empty() && (int) r.phys.size() != count;
#endif
        return r;
    }();
    return v;
}
inline int to_phys(int logical) {
    const VisibleGpus& v = visible_gpus();
    return v.active && logical >= 0 && logical < (int) v.phys.size() ? v.phys[(size_t) logical] : logical;
}
inline int to_logical(int phys) {
    const VisibleGpus& v = visible_gpus();
    if (!v.active) return phys;
    for (size_t i = 0; i < v.phys.size(); ++i) if (v.phys[i] == phys) return (int) i;
    return -1;
}
inline hipError_t set_device(int logical) {
    const VisibleGpus& v = visible_gpus();
    if (v.active && (logical < 0 || logical >= (int) v.phys.size())) return hipErrorInvalidDevice;
    return hipSetDevice(to_phys(logical));
}
// A thread that never chose a device starts on HIP's device 0 - the integrated Radeon, here - so the first time it
// asks, it is moved to logical 0 (CUDA's own default) and told so.
inline hipError_t get_device(int* logical) {
    const hipError_t e = hipGetDevice(logical);
    if (e != hipSuccess || !visible_gpus().active) return e;
    const int l = to_logical(*logical);
    if (l >= 0) { *logical = l; return hipSuccess; }
    *logical = 0;
    return hipSetDevice(visible_gpus().phys[0]);
}
inline hipError_t get_device_count(int* n) {
    const hipError_t e = hipGetDeviceCount(n);
    if (e == hipSuccess && visible_gpus().active) *n = (int) visible_gpus().phys.size();
    return e;
}
inline hipError_t get_device_properties(hipDeviceProp_t* p, int logical) { return hipGetDeviceProperties(p, to_phys(logical)); }
inline hipError_t device_get_attribute(int* v, hipDeviceAttribute_t a, int logical) {
    return hipDeviceGetAttribute(v, a, to_phys(logical));
}
// every program that includes this shim starts on logical GPU 0 (before main, on the main thread)
struct DefaultDevice {
    DefaultDevice() {
        if (visible_gpus().active) (void) hipSetDevice(visible_gpus().phys[0]);
        (void) hipGetLastError();
    }
};
#if !defined(__HIP_DEVICE_COMPILE__)
inline DefaultDevice g_default_device;
#endif
}  // namespace strata::hip_compat
// Graph capture beside other threads.  A capture in thread-local mode was still broken on HIP (Windows) by another
// thread's host->device copy from pageable memory - the adaptive tier's refills, which HIP stages synchronously -
// "operation failed due to a previous error during capture" (2026-10-02).  So every capture holds this lock from its
// begin to its end, and a thread that copies beside the decode loop takes it around each copy (capture_mutex()).
// Per thread and counted: an end without its begin releases nothing.
#include <mutex>
namespace strata::hip_compat {
inline std::mutex& capture_mutex() {
    static std::mutex m;
    return m;
}
inline int& capture_depth() {
    static thread_local int d = 0;
    return d;
}
inline hipError_t begin_capture(hipStream_t s, hipStreamCaptureMode mode) {
    if (capture_depth() == 0) capture_mutex().lock();
    const hipError_t e = hipStreamBeginCapture(s, mode);
    if (e == hipSuccess) ++capture_depth();
    else if (capture_depth() == 0) capture_mutex().unlock();
    return e;
}
inline hipError_t end_capture(hipStream_t s, hipGraph_t* g) {
    const hipError_t e = hipStreamEndCapture(s, g);
    if (capture_depth() > 0 && --capture_depth() == 0) capture_mutex().unlock();
    return e;
}
}  // namespace strata::hip_compat
#define cudaGetDevice ::strata::hip_compat::get_device
#define cudaSetDevice ::strata::hip_compat::set_device
#define cudaGetDeviceCount ::strata::hip_compat::get_device_count
#define cudaGetDeviceProperties ::strata::hip_compat::get_device_properties
#define cudaDeviceGetAttribute ::strata::hip_compat::device_get_attribute

// Free device memory, as the WDDM budget sees it.  On Windows an allocation past the process's local-memory budget
// does not fail: the driver backs it with system RAM, which the GPU then reads at the PCIe rate (~28 GB/s instead of
// ~565 on the RX 6950 XT: the tail of the expert cache and the MTP draft head ended up there, tuning/LOG.md).
// hipMemGetInfo does not count what the rest of the system holds (the desktop compositor alone kept 1.4 GiB), so the
// engine sized its expert cache past the budget.  cudaMemGetInfo therefore reports the smaller of HIP's free figure
// and the budget DXGI leaves this process (QueryVideoMemoryInfo on the adapter with this device's LUID), which the
// engine's own sizing and its post-allocation check then respect.  STRATA_WDDM_BUDGET=0 reports HIP's figure alone.
namespace strata::platform {
bool wddm_local_memory(const char* luid8, unsigned long long* budget, unsigned long long* usage);
bool wddm_others_local_usage(const char* luid8, unsigned long long* others);
}
namespace strata::hip_compat {
inline hipError_t mem_get_info(size_t* free_b, size_t* total_b) {
    const hipError_t e = hipMemGetInfo(free_b, total_b);
#if defined(_WIN32)
    static const bool on = [] {
        const char* v = std::getenv("STRATA_WDDM_BUDGET");
        return v == nullptr || std::atoi(v) != 0;
    }();
    int dev = 0;
    hipDeviceProp_t p{};
    if (e == hipSuccess && on && free_b != nullptr && hipGetDevice(&dev) == hipSuccess &&
        hipGetDeviceProperties(&p, dev) == hipSuccess) {
        unsigned long long budget = 0, usage = 0;
        if (strata::platform::wddm_local_memory(p.luid, &budget, &usage)) {
            const unsigned long long room = budget > usage ? budget - usage : 0;
            if (room < (unsigned long long) *free_b) *free_b = (size_t) room;
            // The budget only shrinks for other programs' VRAM once this process presses on it (RX 6950 XT, desktop and
            // apps holding 1.95 GiB: 15.20 GiB idle, 13.74 GiB after the cache was allocated) - so the cache, sized
            // before that, used to overrun and the last allocations (the draft head) landed in system RAM.  Predict it:
            // the limit is min(budget now, the largest budget seen - what the other programs hold).  Not "budget minus
            // others": once the budget has shrunk that counts them twice.  STRATA_WDDM_OTHERS=0: the budget alone.
            static const bool others_on = [] {
                const char* v = std::getenv("STRATA_WDDM_OTHERS");
                return v == nullptr || std::atoi(v) != 0;
            }();
            static std::atomic<unsigned long long> max_budget{0};
            unsigned long long seen = max_budget.load();
            while (budget > seen && !max_budget.compare_exchange_weak(seen, budget)) {}
            seen = std::max(seen, budget);
            unsigned long long others = 0;
            if (others_on && strata::platform::wddm_others_local_usage(p.luid, &others)) {
                const unsigned long long limit = std::min(budget, seen > others ? seen - others : 0ull);
                const unsigned long long room2 = limit > usage ? limit - usage : 0;
                static const bool dbg = std::getenv("STRATA_WDDM_DEBUG") != nullptr;
                if (dbg)
                    std::fprintf(stderr, "wddm: hip free %.2f, budget %.2f, usage %.2f, others %.2f -> room %.2f / %.2f GiB\n",
                                 *free_b / 1073741824.0, budget / 1073741824.0, usage / 1073741824.0, others / 1073741824.0,
                                 room / 1073741824.0, room2 / 1073741824.0);
                if (room2 < (unsigned long long) *free_b) *free_b = (size_t) room2;
            }
        }
    }
    (void) hipGetLastError();
#endif
    return e;
}
}  // namespace strata::hip_compat
#define cudaMemGetInfo ::strata::hip_compat::mem_get_info

// Scratch (private memory) warm-up.  Windows' HIP runtime keeps one scratch buffer per stream and GROWS it when a
// kernel needs more per-thread scratch than the buffer was sized for - which it does by waiting for every command
// already queued on that stream.  The engine launches graphs that begin with kernels spinning on a host flag and
// whose later kernels need more scratch: hipGraphLaunch then waits for the spin to end, which only the launching
// thread can cause (a deadlock; the GPU spins until the driver resets it).  Every stream therefore starts with one
// trivial kernel that needs STRATA_SCRATCH_WARM_BYTES (default 512: the largest of the engine's kernels is 424)
// bytes of scratch per thread, so the buffer is at its final size before any spin kernel exists.  It costs 42.6 MiB
// of VRAM per stream at 512 B; 0 turns it off.  (The HIP-language TUs install the launcher below at start-up.)
#include <atomic>
#include <chrono>
#include <thread>
namespace strata::hip_compat {
inline int scratch_warm_bytes() {
    static const int v = [] {
        const char* e = std::getenv("STRATA_SCRATCH_WARM_BYTES");
        return e != nullptr && e[0] != '\0' ? std::atoi(e) : 512;
    }();
    return v;
}
using WarmStreamFn = void (*)(hipStream_t, int);
inline std::atomic<WarmStreamFn>& warm_stream_fn() {
    static std::atomic<WarmStreamFn> f{nullptr};
    return f;
}
inline void warm_stream(hipStream_t s) {
    const int bytes = scratch_warm_bytes();
    if (bytes <= 0) return;
    if (const WarmStreamFn f = warm_stream_fn().load()) f(s, bytes);
}
// the streams the engine created (a device sync checks each of them, see stream_sync below)
inline std::mutex& streams_mutex() {
    static std::mutex m;
    return m;
}
inline std::vector<hipStream_t>& streams() {
    static std::vector<hipStream_t> v;
    return v;
}
inline hipError_t stream_create(hipStream_t* s) {
    const hipError_t e = hipStreamCreate(s);
    if (e == hipSuccess) {
        warm_stream(*s);
        std::lock_guard<std::mutex> lk(streams_mutex());
        streams().push_back(*s);
    }
    return e;
}
inline hipError_t stream_create_with_flags(hipStream_t* s, unsigned int flags) {
    const hipError_t e = hipStreamCreateWithFlags(s, flags);
    if (e == hipSuccess) {
        warm_stream(*s);
        std::lock_guard<std::mutex> lk(streams_mutex());
        streams().push_back(*s);
    }
    return e;
}
inline hipError_t stream_destroy(hipStream_t s) {
    {
        std::lock_guard<std::mutex> lk(streams_mutex());
        auto& v = streams();
        v.erase(std::remove(v.begin(), v.end(), s), v.end());
    }
    return hipStreamDestroy(s);
}

// A SYNC THAT IS CHECKED AGAINST THE DEVICE (STRATA_SYNC_CHECK; the "!!!" hunt, tuning/LOG.md 2026-10-03).
// On Windows HIP (ROCm 7.13, RX 6950 XT) hipStreamSynchronize was seen to return while work queued on the stream
// still ran - the per-request reset's memsets; whatever the next stream then wrote was wiped, or read too early.
// The check: a one-thread kernel queued on the stream right before the sync writes a serial number into host memory
// the device can address (hipHostMalloc, mapped); when the sync returns the host looks for that number.  It is there
// only if everything queued before it has run - no runtime bookkeeping is involved in that answer.
//   STRATA_SYNC_CHECK=1  report a sync that returned early (stderr, the first 20 and then every 100th)
//   STRATA_SYNC_CHECK=2  report it and wait for the stamp (up to 20 s) before returning
// Unset or 0: the plain HIP calls.  One GPU only (the stamp's device address belongs to the device that mapped it).
using StampFn = void (*)(hipStream_t, unsigned long long*, unsigned long long);
inline std::atomic<StampFn>& stamp_fn() {
    static std::atomic<StampFn> f{nullptr};
    return f;
}
struct SyncCheck {
    static constexpr unsigned long long kSlots = 4096;
    int mode = 0;
    volatile unsigned long long* host = nullptr;
    unsigned long long* dev = nullptr;
    std::atomic<unsigned long long> seq{0}, early{0}, checked{0};
};
inline SyncCheck& sync_check() {
    static SyncCheck c;
    static const bool once = [] {
        const char* v = std::getenv("STRATA_SYNC_CHECK");
        const int mode = v != nullptr ? std::atoi(v) : 0;
        int count = 0;
        if (mode <= 0 || hipGetDeviceCount(&count) != hipSuccess) { (void) hipGetLastError(); return true; }
        if (visible_gpus().active ? visible_gpus().phys.size() != 1 : count != 1) return true;
        void *h = nullptr, *d = nullptr;
        if (hipHostMalloc(&h, SyncCheck::kSlots * sizeof(unsigned long long), hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer(&d, h, 0) != hipSuccess) {
            (void) hipGetLastError();
            return true;
        }
        std::memset(h, 0, SyncCheck::kSlots * sizeof(unsigned long long));
        c.host = (volatile unsigned long long*) h;
        c.dev = (unsigned long long*) d;
        c.mode = mode;
        std::fprintf(stderr, "strata hip: stream syncs are checked against the device (STRATA_SYNC_CHECK=%d)\n", mode);
        return true;
    }();
    (void) once;
    return c;
}
inline hipError_t stream_sync(hipStream_t s) {
    SyncCheck& c = sync_check();
    const StampFn f = c.mode > 0 ? stamp_fn().load() : nullptr;
    // not inside this thread's graph capture, and not beside another thread's (the stamp is a kernel launch)
    if (f == nullptr || capture_depth() > 0 || !capture_mutex().try_lock()) return hipStreamSynchronize(s);
    const unsigned long long v = c.seq.fetch_add(1) + 1;
    const unsigned long long slot = v % SyncCheck::kSlots;
    f(s, c.dev + slot, v);
    capture_mutex().unlock();
    const hipError_t e = hipStreamSynchronize(s);
    if (e != hipSuccess) return e;
    c.checked.fetch_add(1);
    if (c.host[slot] == v) return e;
    const unsigned long long n = c.early.fetch_add(1) + 1;
    const auto t0 = std::chrono::steady_clock::now();
    double waited_ms = -1.0;
    if (c.mode >= 2) {
        while (c.host[slot] != v && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20))
            std::this_thread::yield();
        waited_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    if (n <= 20 || n % 100 == 0)
        std::fprintf(stderr, "strata hip: hipStreamSynchronize(%p) RETURNED EARLY: the work queued before it had not run "
                             "(%llu of %llu checked syncs)%s%.2f%s\n", (void*) s, n, c.checked.load(),
                     waited_ms >= 0 ? "; waited " : "; not waited ", waited_ms >= 0 ? waited_ms : 0.0,
                     waited_ms >= 0 ? (c.host[slot] == v ? " ms for it" : " ms - IT NEVER RAN") : "");
    return e;
}
inline hipError_t device_sync() {
    if (sync_check().mode > 0) {
        std::vector<hipStream_t> v;
        {
            std::lock_guard<std::mutex> lk(streams_mutex());
            v = streams();
        }
        for (hipStream_t s : v) (void) stream_sync(s);
        (void) stream_sync(nullptr);
        (void) hipGetLastError();
    }
    return hipDeviceSynchronize();
}
}  // namespace strata::hip_compat
#define cudaStreamCreate ::strata::hip_compat::stream_create
#define cudaStreamCreateWithFlags ::strata::hip_compat::stream_create_with_flags
#if defined(__HIP__)
namespace strata::hip_compat {
template <int N>
__global__ void __launch_bounds__(256) scratch_warm_kernel(int* sink, int n) {
    volatile int buf[N];                       // a dynamically indexed volatile array stays in scratch memory
    for (int i = 0; i < N; ++i) buf[i] = i + n;
    const int v = buf[(n + (int) threadIdx.x) % N];
    if (n == 123456789) sink[0] = v;           // never taken: only keeps the array alive
}
inline void warm_stream_impl(hipStream_t s, int bytes) {
    if (bytes <= 128) scratch_warm_kernel<32><<<64, 256, 0, s>>>(nullptr, 0);
    else if (bytes <= 512) scratch_warm_kernel<128><<<64, 256, 0, s>>>(nullptr, 0);
    else scratch_warm_kernel<512><<<64, 256, 0, s>>>(nullptr, 0);
    (void) hipGetLastError();
}
#if !defined(__HIP_DEVICE_COMPILE__)
inline const bool g_warm_installed = (warm_stream_fn().store(&warm_stream_impl), true);
#endif
template <int N>
__global__ void sync_stamp_kernel(unsigned long long* p, unsigned long long v) {
    if (blockIdx.x == 0 && threadIdx.x == 0) *p = v;
}
inline void sync_stamp_impl(hipStream_t s, unsigned long long* p, unsigned long long v) {
    sync_stamp_kernel<0><<<1, 1, 0, s>>>(p, v);
    (void) hipGetLastError();
}
#if !defined(__HIP_DEVICE_COMPILE__)
inline const bool g_stamp_installed = (stamp_fn().store(&sync_stamp_impl), true);
#endif
}  // namespace strata::hip_compat
#endif

template <typename Kernel>
inline hipError_t cudaFuncSetAttribute(Kernel kernel, hipFuncAttribute attribute, int value) {
    return hipFuncSetAttribute(reinterpret_cast<const void*>(kernel), attribute, value);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph, unsigned long long flags) {
    return hipGraphInstantiateWithFlags(exec, graph, flags);
}
inline hipError_t cudaGraphInstantiate(hipGraphExec_t* exec, hipGraph_t graph,
                                     hipGraphNode_t* error, char* log, size_t size) {
    return hipGraphInstantiate(exec, graph, error, log, size);
}
#define __trap() __builtin_trap()   // the compiled-out sm_80 paths (never selected on AMD)
#define cudaMemcpyToSymbol(symbol, ...) hipMemcpyToSymbol(HIP_SYMBOL(symbol), __VA_ARGS__)

#include "intrinsics.hpp"
