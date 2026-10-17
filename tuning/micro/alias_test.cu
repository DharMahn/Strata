// The "!!!" bug in 80 lines (Windows, ROCm 7.13, RX 6950 XT; tuning/LOG.md 2026-10-03).
// hipHostRegister(..., Mapped) gives the registered memory a DEVICE alias: an address the GPU reads it at.  On Windows
// that address is a plain number - nothing reserves it in the process's own address space - so an ordinary host
// allocation can later be given the very same numbers.  hipMemcpy classifies its pointers by looking them up among
// the runtime's memory objects (whatever `kind` says), so a copy "device -> that host buffer" is then done as a copy
// INTO THE REGISTERED MEMORY, and the host buffer gets nothing.
// This program makes the collision on purpose: it commits host memory AT the alias, copies a device pattern "to the
// host", and reports where the bytes went.  With the engine's fix (the alias range reserved right after the
// registration: pass "guard") the allocation at the alias is refused, so the collision cannot happen.
//   alias_test [MiB]            the collision, shown (the registration's size, default 1024)
//   alias_test [MiB] guard      the alias range's free parts reserved first: no host buffer can land there
// What the runtime reserves of the alias range depends on the size: a 64 MiB registration's alias is reserved whole,
// the engine's 0.6-0.9 GiB slices' were not.
#include <hip/hip_runtime.h>
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { std::printf("%s: %s\n", #x, hipGetErrorString(e_)); std::exit(1); } } while (0)

int main(int argc, char** argv) {
    const bool guard = argc > 1 && std::strcmp(argv[argc - 1], "guard") == 0;
    const size_t reg_mib = argc > 1 && std::atoi(argv[1]) > 0 ? (size_t) std::atoi(argv[1]) : 1024;
    int n = 0, dev = -1;
    CK(hipGetDeviceCount(&n));
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p;
        CK(hipGetDeviceProperties(&p, i));
        if (dev < 0 && std::strstr(p.gcnArchName, "gfx1030")) dev = i;
    }
    if (dev < 0) { std::printf("no gfx1030\n"); return 1; }
    CK(hipSetDevice(dev));
    const size_t reg_bytes = reg_mib << 20, n_copy = 4u << 20;
    // the "arena": registered, mapped host memory holding 0x11
    unsigned char* arena = (unsigned char*) VirtualAlloc(nullptr, reg_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    std::memset(arena, 0x11, reg_bytes);
    CK(hipHostRegister(arena, reg_bytes, hipHostRegisterPortable | hipHostRegisterMapped));
    void* alias = nullptr;
    CK(hipHostGetDevicePointer(&alias, arena, 0));
    std::printf("registered host memory %p (%zu MiB), its device alias %p\n", (void*) arena, reg_mib, alias);
    if (alias == (void*) arena) { std::printf("the alias is the host address itself: nothing to collide with\n"); return 0; }
    // what this process's address space holds at the alias's numbers; the first free part is where a host buffer could land
    auto walk = [&](bool reserve) -> unsigned char* {
        unsigned char* free_at = nullptr;
        for (unsigned char* a = (unsigned char*) alias; a < (unsigned char*) alias + reg_bytes;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery(a, &mbi, sizeof mbi) == 0) break;
            unsigned char* end = (unsigned char*) mbi.BaseAddress + mbi.RegionSize;
            if (end > (unsigned char*) alias + reg_bytes) end = (unsigned char*) alias + reg_bytes;
            std::printf("  alias + %5zu MiB .. + %5zu MiB: %s\n", (size_t) (a - (unsigned char*) alias) >> 20,
                        (size_t) (end - (unsigned char*) alias) >> 20,
                        mbi.State == MEM_FREE ? "FREE" : mbi.State == MEM_RESERVE ? "reserved" : "committed");
            if (mbi.State == MEM_FREE) {
                unsigned char* s = (unsigned char*) (((uintptr_t) a + 65535) & ~(uintptr_t) 65535);
                if (reserve && s < end) VirtualAlloc(s, (size_t) (end - s), MEM_RESERVE, PAGE_NOACCESS);
                if (free_at == nullptr && s + n_copy <= end) free_at = s;
            }
            a = end;
        }
        return free_at;
    };
    unsigned char* free_at = walk(guard);
    if (guard) { std::printf("guard: the free parts are reserved; now:\n"); free_at = walk(false); }
    if (free_at == nullptr) {
        std::printf("no free part in the alias range: a host allocation cannot land there%s\n", guard ? " - the guard works" : "");
        return 0;
    }
    // an ordinary host buffer that gets the alias's numbers (here: asked for; in the engine: by chance)
    const size_t off = (size_t) (free_at - (unsigned char*) alias);
    unsigned char* host = (unsigned char*) VirtualAlloc(free_at, n_copy, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    if (host == nullptr) {
        std::printf("a host allocation at the alias is refused (error %lu)\n", GetLastError());
        return 1;
    }
    std::memset(host, 0x22, n_copy);
    // device memory holding 0x33, copied "device -> host"
    unsigned char* d = nullptr;
    CK(hipMalloc(&d, n_copy));
    CK(hipMemset(d, 0x33, n_copy));
    const hipError_t ce = hipMemcpy(host, d, n_copy, hipMemcpyDeviceToHost);
    CK(hipDeviceSynchronize());
    size_t host_got = 0, arena_hit = 0;
    for (size_t i = 0; i < n_copy; ++i) host_got += host[i] == 0x33;
    for (size_t i = 0; i < reg_bytes; ++i) arena_hit += arena[i] != 0x11;
    size_t first = 0;
    while (first < reg_bytes && arena[first] == 0x11) ++first;
    std::printf("hipMemcpy(host buffer %p at alias + %zu MiB, device, %zu MiB, DeviceToHost): %s\n", (void*) host, off >> 20,
                n_copy >> 20, hipGetErrorString(ce));
    std::printf("  the host buffer received %zu of %zu bytes\n", host_got, n_copy);
    std::printf("  the REGISTERED memory changed in %zu bytes", arena_hit);
    if (arena_hit) std::printf(", from offset %zu MiB (now 0x%02x)", first >> 20, arena[first]);
    std::printf("\n%s\n", arena_hit && host_got == 0 ? "=> the copy went into the registered memory: the bug"
                                                      : host_got == n_copy && !arena_hit ? "=> the copy went where it was told"
                                                                                         : "=> something else");
    return 0;
}
