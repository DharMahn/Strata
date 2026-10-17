#pragma once

// CUDA device intrinsics used by Strata kernels that HIP does not provide on RDNA3 / RDNA4 (wave32).
// This header is included only from the HIP cuda_runtime compatibility shim.
#if defined(__HIPCC__)

#include <cstdint>

namespace strata::hip_compat {

__device__ __forceinline__ int signed_byte(uint32_t word, int lane) {
    const unsigned value = (word >> (lane * 8)) & 0xffu;
    return value < 0x80u ? static_cast<int>(value) : static_cast<int>(value) - 0x100;
}

// CUDA's signed __dp4a: four signed byte products accumulated modulo 2^32.
__device__ __forceinline__ int dp4a(int a, int b, int c) {
#if (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1200__) || \
     defined(__gfx1201__)) && __has_builtin(__builtin_amdgcn_sudot4)
    // RDNA3 and RDNA4 expose the signed/unsigned dot4 form (v_dot4_i32_iu8). Mark
    // both packed operands signed to preserve CUDA __dp4a semantics; keep the
    // portable path for other HIP compilers/targets.
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif (defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1034__)) && \
      __has_builtin(__builtin_amdgcn_sdot4)
    // RDNA2 has the plain signed dot4 (v_dot4_i32_i8, "dot1-insts"), the same instruction llama.cpp's HIP
    // backend uses on these cards; clamp off keeps CUDA __dp4a's wrap-around accumulation.
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t sum = static_cast<uint32_t>(c);
#pragma unroll
    for (int lane = 0; lane < 4; ++lane)
        sum += static_cast<uint32_t>(signed_byte(ua, lane) * signed_byte(ub, lane));
    return static_cast<int>(sum);
#endif
}

// CUDA's packed byte subtract wraps independently in each unsigned byte lane.  SWAR (Hacker's Delight 2-18): the low
// 7 bits of each lane subtract with the lane's top bit forced on, so no borrow crosses a lane; the top bit is then
// a ^ b ^ borrow-in.  6 ALU ops instead of the 4-lane extract/insert loop (~16): the i-quant expert dot products
// (vec_dot_iq2_*/iq3_*) apply their signs with one of these per 4 weights.
__device__ __forceinline__ int vsub4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    return static_cast<int>(((ua | 0x80808080u) - (ub & 0x7f7f7f7fu)) ^ ((ua ^ ~ub) & 0x80808080u));
}

// CUDA's packed signed-byte saturating subtract.
__device__ __forceinline__ int vsubss4(int a, int b) {
    const uint32_t ua = static_cast<uint32_t>(a);
    const uint32_t ub = static_cast<uint32_t>(b);
    uint32_t out = 0;
#pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        int value = signed_byte(ua, lane) - signed_byte(ub, lane);
        value = value < -128 ? -128 : (value > 127 ? 127 : value);
        out |= (static_cast<uint32_t>(value) & 0xffu) << (lane * 8);
    }
    return static_cast<int>(out);
}

// CUDA's four-lane byte compare, returning 0xff for each unequal lane and 0 otherwise.  SWAR: a lane of x = a ^ b is
// nonzero iff its low 7 bits + 0x7f reach bit 7 (no carry leaves the lane) or its bit 7 is set; that top bit t then
// widens to 0xff as (t - (t >> 7)) | t.  7-8 ALU ops (a constant b folds the xor) instead of 4 compare/select pairs.
__device__ __forceinline__ int vcmpne4(int a, int b) {
    const uint32_t x = static_cast<uint32_t>(a) ^ static_cast<uint32_t>(b);
    const uint32_t t = (((x & 0x7f7f7f7fu) + 0x7f7f7f7fu) | x) & 0x80808080u;
    return static_cast<int>((t - (t >> 7)) | t);
}

// CUDA's __byte_perm(x, y, s): output byte i is byte ((s >> 4i) & 7) of the 8-byte value {y:x}.  HIP's own version
// indexes a local byte array with the selector: a constant selector folds away, but a run-time one (the i-quant table
// lookups, the Q2_0 crumb unpacking) becomes a long 64-bit shift/mask sequence per byte, which made the IQ4_XS/IQ4_NL
// GEMVs ~6x slower than IQ3_S on gfx1030 (tuning/LOG.md).  v_perm_b32 is the same selection in one instruction; its
// selector has one BYTE per output byte (0-7 pick from {S0:S1}, S1 the low word), hence the nibble-to-byte spread.
__device__ __forceinline__ unsigned int byte_perm(unsigned int x, unsigned int y, unsigned int s) {
#if __has_builtin(__builtin_amdgcn_perm)
    const unsigned int sel = (s & 0x7u) | ((s & 0x70u) << 4) | ((s & 0x700u) << 8) | ((s & 0x7000u) << 12);
    return __builtin_amdgcn_perm(y, x, sel);
#else
    const unsigned long long v = ((unsigned long long) y << 32) | x;
    unsigned int r = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) r |= (unsigned int) ((v >> (8u * ((s >> (4 * i)) & 7u))) & 0xffu) << (8 * i);
    return r;
#endif
}

// CUDA's mask argument describes participating lanes. The current kernel set uses full
// wave32 masks; reject future partial-mask use instead of silently dropping its semantics.
__device__ __forceinline__ void require_full_wave_mask(uint32_t mask) {
    if (mask != 0xffffffffu) __builtin_trap();
}

template <typename T>
__device__ __forceinline__ T shfl_xor_sync(uint32_t mask, T value, int lane_mask, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_xor(value, lane_mask, width);
}

template <typename T>
__device__ __forceinline__ T shfl_down_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_down(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T shfl_up_sync(uint32_t mask, T value, unsigned delta, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T shfl_sync(uint32_t mask, T value, int source_lane, int width = 32) {
    require_full_wave_mask(mask);
    return __shfl(value, source_lane, width);
}

__device__ __forceinline__ unsigned ballot_sync(uint32_t mask, int predicate) {
    require_full_wave_mask(mask);
    return static_cast<unsigned>(__ballot(predicate));
}

}  // namespace strata::hip_compat

#define __dp4a(a, b, c) (::strata::hip_compat::dp4a((a), (b), (c)))
#define __byte_perm(x, y, s) (::strata::hip_compat::byte_perm((x), (y), (s)))
#define __vsub4(a, b) (::strata::hip_compat::vsub4((a), (b)))
#define __vsubss4(a, b) (::strata::hip_compat::vsubss4((a), (b)))
#define __vcmpne4(a, b) (::strata::hip_compat::vcmpne4((a), (b)))
#define __shfl_xor_sync(...) (::strata::hip_compat::shfl_xor_sync(__VA_ARGS__))
#define __shfl_down_sync(...) (::strata::hip_compat::shfl_down_sync(__VA_ARGS__))
#define __shfl_up_sync(...) (::strata::hip_compat::shfl_up_sync(__VA_ARGS__))
#define __shfl_sync(...) (::strata::hip_compat::shfl_sync(__VA_ARGS__))
#define __ballot_sync(mask, predicate) (::strata::hip_compat::ballot_sync((mask), (predicate)))
// AMD's sleep instruction accepts only 0..15; the synchronization loops use it as a
// backoff hint, so use its smallest portable delay independently of CUDA cycle counts.
#define __nanosleep(cycles) __builtin_amdgcn_s_sleep(1)

#endif  // defined(__HIPCC__)
