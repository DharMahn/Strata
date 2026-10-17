// include/strata/platform/memory.hpp - plan v0.3 P0.1/P1: keep a large host region resident.
//
// `cudaHostRegister` refuses the 31.6 GiB expert arena on Windows, and an unlocked arena is trimmed under memory
// pressure (the CPU pool then swung 2x between runs). With the n-gram table out of RAM there is headroom to lock
// it instead: raise the process's minimum working set by the region's size, then VirtualLock it (Windows needs
// only SeIncreaseWorkingSetPrivilege, which ordinary accounts hold). Linux: mlock.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< may be less than requested; the rest stays pageable
    std::string note;            ///< what was done or why it failed, for the startup print
};

/// Lock [p, p + bytes) into physical memory. Partial success is reported, not hidden.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// Windows: the local-memory (VRAM) budget the OS grants this process on the adapter with this 8-byte LUID, and
/// what the process uses of it (DXGI QueryVideoMemoryInfo).  Device memory past the budget is backed by system RAM.
/// false elsewhere, or when DXGI cannot tell.
bool wddm_local_memory(const char* luid8, unsigned long long* budget, unsigned long long* usage);

/// Windows: dedicated VRAM that OTHER processes hold on the adapter with this LUID (the "GPU Adapter Memory" minus this
/// process's "GPU Process Memory" performance counters - what Task Manager shows).  The DXGI budget above does not
/// shrink for them: with the desktop and a browser holding 3-4 GB of a 16 GB card, an engine that fills its budget
/// gets its last allocations (the draft head) backed by system RAM.  Sampled at most every 500 ms.  false elsewhere,
/// or when the counters are unavailable.
bool wddm_others_local_usage(const char* luid8, unsigned long long* others);

/// Windows: keep the numeric address range [p, p + n) out of this process's future allocations by reserving every
/// free part of it (no memory is committed).  Used for the DEVICE addresses the HIP runtime hands out: they are not
/// reserved in the process's own address space, so the OS can give the same numbers to an ordinary allocation - and
/// HIP then takes that host buffer for device memory (hip_compat/cuda_runtime.h, "alias guard").
/// Returns 1 when the range is now reserved (by this call, or already by someone else), -1 when part of it is
/// committed memory already (a host allocation lives there: the collision exists), 0 when there is nothing to do
/// (not Windows, an empty range).  `owner` names the reservation for `unguard_address_range`.
int guard_address_range(const void* owner, const void* p, unsigned long long n);
void unguard_address_range(const void* owner);

}  // namespace strata::platform
