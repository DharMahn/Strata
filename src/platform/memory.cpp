// src/platform/memory.cpp - see include/strata/platform/memory.hpp.
#include "strata/platform/memory.hpp"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi1_4.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cwchar>
#include <map>
#include <mutex>
#include <vector>
#else
#include <sys/mman.h>
#endif

namespace strata::platform {

#if defined(_WIN32)
// The local (VRAM) budget the OS gives this process on the adapter with this LUID, and what the process uses of it.
// dxgi.dll is loaded on first use (no link dependency); the adapter is looked up once and kept.
bool wddm_local_memory(const char* luid8, unsigned long long* budget, unsigned long long* usage) {
    static std::mutex mu;
    static IDXGIAdapter3* adapter = nullptr;
    static LUID have{};
    static bool tried = false;
    std::lock_guard<std::mutex> lk(mu);
    LUID want;
    std::memcpy(&want, luid8, sizeof want);
    if (adapter == nullptr || have.LowPart != want.LowPart || have.HighPart != want.HighPart) {
        if (adapter != nullptr) { adapter->Release(); adapter = nullptr; }
        if (tried && have.LowPart == want.LowPart && have.HighPart == want.HighPart) return false;
        tried = true;
        have = want;
        HMODULE dxgi = LoadLibraryA("dxgi.dll");
        using Create = HRESULT(WINAPI*)(REFIID, void**);
        Create create = dxgi ? (Create) (void*) GetProcAddress(dxgi, "CreateDXGIFactory1") : nullptr;
        IDXGIFactory4* factory = nullptr;
        if (create == nullptr || FAILED(create(__uuidof(IDXGIFactory4), (void**) &factory)) || factory == nullptr)
            return false;
        const HRESULT hr = factory->EnumAdapterByLuid(want, __uuidof(IDXGIAdapter3), (void**) &adapter);
        factory->Release();
        if (FAILED(hr) || adapter == nullptr) { adapter = nullptr; return false; }
    }
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (FAILED(adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) return false;
    *budget = info.Budget;
    *usage = info.CurrentUsage;
    return true;
}

// One PDH query with two wildcard counters, opened on first use (pdh.dll loaded at run time, no link dependency).
// Instance names: "luid_0x<High>_0x<Low>_phys_<n>" (adapter) and "pid_<pid>_luid_0x<High>_0x<Low>_phys_<n>" (process),
// hex in lower case on Windows 11 - matched without regard to case.
bool wddm_others_local_usage(const char* luid8, unsigned long long* others) {
    using OpenQ = PDH_STATUS(WINAPI*)(LPCWSTR, DWORD_PTR, PDH_HQUERY*);
    using AddC = PDH_STATUS(WINAPI*)(PDH_HQUERY, LPCWSTR, DWORD_PTR, PDH_HCOUNTER*);
    using Collect = PDH_STATUS(WINAPI*)(PDH_HQUERY);
    using GetArr = PDH_STATUS(WINAPI*)(PDH_HCOUNTER, DWORD, LPDWORD, LPDWORD, PPDH_FMT_COUNTERVALUE_ITEM_W);
    static std::mutex mu;
    static bool tried = false, ok = false;
    static Collect collect = nullptr;
    static GetArr get_arr = nullptr;
    static PDH_HQUERY query = nullptr;
    static PDH_HCOUNTER adapter_c = nullptr, process_c = nullptr;
    static std::chrono::steady_clock::time_point last{};
    static unsigned long long last_others = 0;
    static char last_luid[8] = {};
    std::lock_guard<std::mutex> lk(mu);
    if (!tried) {
        tried = true;
        HMODULE pdh = LoadLibraryA("pdh.dll");
        OpenQ open_q = pdh ? (OpenQ) (void*) GetProcAddress(pdh, "PdhOpenQueryW") : nullptr;
        AddC add_c = pdh ? (AddC) (void*) GetProcAddress(pdh, "PdhAddEnglishCounterW") : nullptr;
        collect = pdh ? (Collect) (void*) GetProcAddress(pdh, "PdhCollectQueryData") : nullptr;
        get_arr = pdh ? (GetArr) (void*) GetProcAddress(pdh, "PdhGetFormattedCounterArrayW") : nullptr;
        ok = open_q && add_c && collect && get_arr && open_q(nullptr, 0, &query) == ERROR_SUCCESS &&
             add_c(query, L"\\GPU Adapter Memory(*)\\Dedicated Usage", 0, &adapter_c) == ERROR_SUCCESS &&
             add_c(query, L"\\GPU Process Memory(*)\\Dedicated Usage", 0, &process_c) == ERROR_SUCCESS;
    }
    if (!ok) return false;
    const auto now = std::chrono::steady_clock::now();
    if (last.time_since_epoch().count() != 0 && now - last < std::chrono::milliseconds(500) &&
        std::memcmp(last_luid, luid8, 8) == 0) {
        *others = last_others;
        return true;
    }
    LUID want;
    std::memcpy(&want, luid8, sizeof want);
    wchar_t tag[64];
    swprintf(tag, 64, L"luid_0x%08X_0x%08X_", (unsigned) want.HighPart, (unsigned) want.LowPart);
    wchar_t self[96];
    swprintf(self, 96, L"pid_%lu_%ls", (unsigned long) GetCurrentProcessId(), tag);
    if (collect(query) != ERROR_SUCCESS) return false;
    auto sum = [&](PDH_HCOUNTER c, const wchar_t* prefix, bool anywhere, unsigned long long* out) -> bool {
        DWORD bytes = 0, n = 0;
        if (get_arr(c, PDH_FMT_LARGE, &bytes, &n, nullptr) != (PDH_STATUS) PDH_MORE_DATA || bytes == 0) return false;
        std::vector<unsigned char> buf(bytes);
        auto* items = (PPDH_FMT_COUNTERVALUE_ITEM_W) buf.data();
        if (get_arr(c, PDH_FMT_LARGE, &bytes, &n, items) != ERROR_SUCCESS) return false;
        unsigned long long total = 0;
        bool any = false;
        for (DWORD i = 0; i < n; ++i) {
            const wchar_t* name = items[i].szName;
            const bool match = anywhere ? wcsstr(name, prefix) != nullptr : _wcsnicmp(name, prefix, wcslen(prefix)) == 0;
            if (!match || items[i].FmtValue.CStatus != ERROR_SUCCESS) continue;
            total += (unsigned long long) items[i].FmtValue.largeValue;
            any = true;
        }
        *out = total;
        return any;
    };
    unsigned long long adapter = 0, mine = 0;
    if (!sum(adapter_c, tag, false, &adapter)) return false;
    (void) sum(process_c, self, false, &mine);   // 0 before this process has allocated anything
    last = now;
    std::memcpy(last_luid, luid8, 8);
    last_others = adapter > mine ? adapter - mine : 0;
    *others = last_others;
    return true;
}

namespace {
std::mutex& guard_mu() {
    static std::mutex m;
    return m;
}
std::map<const void*, std::vector<void*>>& guards() {
    static std::map<const void*, std::vector<void*>> g;
    return g;
}
}  // namespace

int guard_address_range(const void* owner, const void* p, unsigned long long n) {
    if (p == nullptr || n == 0) return 0;
    static const bool dbg = std::getenv("STRATA_DBG_ARENA") != nullptr;
    const uintptr_t G = 65536, P = 4096;                       // the allocation granularity, the page size
    const uintptr_t b = (uintptr_t) p, e = b + (uintptr_t) n;
    std::vector<void*> got;
    bool committed = false;
    // the whole range at once when all of it is free (the usual case), else its free regions one by one
    const uintptr_t rb = b & ~(G - 1), re = (e + P - 1) & ~(P - 1);
    if (void* r = VirtualAlloc((void*) rb, re - rb, MEM_RESERVE, PAGE_NOACCESS)) {
        got.push_back(r);
    } else {
        for (uintptr_t a = b; a < e;) {
            MEMORY_BASIC_INFORMATION mbi{};
            if (VirtualQuery((void*) a, &mbi, sizeof mbi) == 0) break;   // past the user address space: nothing can land there
            const uintptr_t rs = (uintptr_t) mbi.BaseAddress, rend = rs + mbi.RegionSize;
            if (mbi.State == MEM_COMMIT) {
                committed = true;
            } else if (mbi.State == MEM_FREE) {
                const uintptr_t s = (std::max(a, rs) + G - 1) & ~(G - 1), t = std::min(re, rend) & ~(P - 1);
                if (s < t)
                    if (void* r = VirtualAlloc((void*) s, t - s, MEM_RESERVE, PAGE_NOACCESS)) got.push_back(r);
            }
            if (rend <= a) break;
            a = rend;
        }
    }
    if (dbg)
        std::fprintf(stderr, "strata dbg: alias guard: %p..%p (%llu B): %zu region(s) reserved%s\n", p, (const void*) e, n,
                     got.size(), committed ? "; PART OF IT IS COMMITTED MEMORY ALREADY" : "");
    if (!got.empty()) {
        std::lock_guard<std::mutex> lk(guard_mu());
        auto& v = guards()[owner];
        v.insert(v.end(), got.begin(), got.end());
    }
    return committed ? -1 : 1;
}

void unguard_address_range(const void* owner) {
    std::vector<void*> v;
    {
        std::lock_guard<std::mutex> lk(guard_mu());
        auto it = guards().find(owner);
        if (it == guards().end()) return;
        v.swap(it->second);
        guards().erase(it);
    }
    for (void* r : v) VirtualFree(r, 0, MEM_RELEASE);
}
#else
bool wddm_local_memory(const char*, unsigned long long*, unsigned long long*) { return false; }
bool wddm_others_local_usage(const char*, unsigned long long*) { return false; }
int guard_address_range(const void*, const void*, unsigned long long) { return 0; }
void unguard_address_range(const void*) {}
#endif

#if defined(_WIN32)
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    HANDLE self = GetCurrentProcess();
    SIZE_T min_ws = 0, max_ws = 0;
    DWORD flags = 0;
    if (!GetProcessWorkingSetSizeEx(self, &min_ws, &max_ws, &flags)) {
        r.note = "GetProcessWorkingSetSizeEx failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    // Locked pages count against the minimum working set, so it must grow by the region plus headroom for the
    // rest of the process. Soft limits: the maximum is not enforced, only the minimum is raised.
    const SIZE_T margin = (SIZE_T) 512 << 20;
    const SIZE_T new_min = min_ws + (SIZE_T) bytes + margin;
    const SIZE_T new_max = max_ws > new_min + margin ? max_ws : new_min + margin;
    if (!SetProcessWorkingSetSizeEx(self, new_min, new_max,
                                    QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE)) {
        r.note = "SetProcessWorkingSetSizeEx(" + std::to_string((unsigned long long) (new_min >> 20)) +
                 " MiB) failed (error " + std::to_string(GetLastError()) + ")";
        return r;
    }
    const uint64_t chunk = 1ull << 30;
    uint8_t* base = (uint8_t*) p;
    for (uint64_t off = 0; off < bytes; off += chunk) {
        const uint64_t n = bytes - off < chunk ? bytes - off : chunk;
        if (!VirtualLock(base + off, (SIZE_T) n)) {
            r.note = "VirtualLock stopped at " + std::to_string((unsigned long long) (off >> 20)) + " of " +
                     std::to_string((unsigned long long) (bytes >> 20)) + " MiB (error " +
                     std::to_string(GetLastError()) + ")";
            r.ok = off > 0;
            return r;
        }
        r.locked_bytes = off + n;
    }
    r.ok = true;
    r.note = "locked " + std::to_string((unsigned long long) (bytes >> 20)) + " MiB via working-set minimum + VirtualLock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p == nullptr || bytes == 0) return;
    const uint64_t chunk = 1ull << 30;
    for (uint64_t off = 0; off < bytes; off += chunk)
        VirtualUnlock((uint8_t*) p + off, (SIZE_T) (bytes - off < chunk ? bytes - off : chunk));
}
#else
LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult r;
    if (p == nullptr || bytes == 0) { r.note = "nothing to lock"; return r; }
    if (mlock(p, bytes) != 0) { r.note = "mlock failed (raise ulimit -l)"; return r; }
    r.ok = true;
    r.locked_bytes = bytes;
    r.note = "mlock";
    return r;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}
#endif

}  // namespace strata::platform
