# Decode path review (2026-10-02)

A second-opinion review of the token generation path, done by a separate agent. It only read the code and the logs in
`tuning/`: nothing was built or run, so every gain below is an estimate to verify. Line numbers refer to commit
065f9d3.

## Summary

About 6 ms of each ~33-35 ms window (~18%) doesn't show up in any bucket of the STRATA_DECODE_TIMING breakdown. Two
host-side costs probably account for most of it, and both are exposed on Windows. After those, the biggest bit-exact
GPU lever is the hyper-connection reads. Figures below are for the p4 IQ3_XXS window (34.6 ms: wait 16.7, CPU 7.7,
draft 2.9).

## Likely wins, ranked

1. **Adaptive-tier swaps appear to stall the decode loop**
   - **Where:** generate.cpp 4860-4863 and 4897; adapt 3962-4019; res_upload 3938.
   - **Evidence:** window − verify − commit − draft is 2.6-3.4 ms/window in every run checked (o4-all, p-ref,
     opts-ref, iq3-ref, dm-on, p4), and no timer covers it.
   - **Suspected cause:**
     - Every 4th window, up to 96 swaps of 1.4-2.3 MB each go out as separate `cudaMemcpyAsync` calls.
     - Their sources are mostly the unpinned arena, which HIP stages synchronously in the calling thread.
     - `res_upload`, a synchronous `cudaMemcpy`, runs on the same thread, queued behind those copies.
     - The thread is joined after the draft, so any time beyond ~3 ms lands on the critical path.
   - **Fix:** publish the evictions before issuing the copies, make the copies truly asynchronous, and don't join
     before the next window.
   - **Gain:** ~2-3 ms per window (6-9%), with no output change if the publish order is kept.
   - **Measure:** add join, apply and swap-count timers. Quick A/B: `--adapt-swaps 24 / 48`, which has never been
     swept.

2. **Window graph launch, ~2 ms (inferred)**
   - **Where:** verify.cpp 1031.
   - **Evidence:** verify − wait − host − stage = 3.1-4.2 ms, while the real GPU tail is ~1.2 ms. The window graph
     has ~2,250 nodes, and the Windows graph is only submitted at the query.
   - **Fix:** split the graph: layers 0-3 as graph A, the rest plus the head as graph B launched from a helper thread.
   - **Gain:** ~1.5-2 ms (4-6%), bit-exact.

3. **Hyper-connection reads, ~6 ms per window, all before the doorbell** (fused_gr.cu)
   - **Measured:** 235-345 GB/s.
   - **Norm:** `gr_norm_multi_kernel` (142-180) runs only T blocks and makes a second global pass. Fix: keep the
     values in registers and write once (~1 ms).
   - **Up:** `gr_up_multi` (257-290) loads its weights row by row. Fix: unroll and hoist the loads (~0.6-0.8 ms).
   - **Down:** more blocks didn't help before; profile it first.
   - All bit-exact.

4. **`resident_plan_partial` before the doorbell** (verify.cpp 638-648)
   - **Why it costs:** a single-thread O(n²) kernel that delays the CPU experts' start on every layer.
   - **Fix:** move it after `doorbell_publish`. Bit-exact, ~0.3-1 ms.

5. **Too many kernels on the critical path** (~2.5 us each)
   - The always-empty PCIe grouped call (4 kernels, ~0.59 ms per window) can be skipped when `device_plan_all_`
     (verify.cpp 731). This also removes graph nodes.
   - `copy_rows`, `hit_add`, combine and cvec can be fused into one kernel.
   - `copy_rows_from_mapped` uses 128 threads per row (elementwise.cu 235-262).
   - **Gain:** ~0.5-1 ms, bit-exact.

6. **Host trims in mode 2** (expert_source.cpp)
   - The zeroing of hit rows (1133-1136) is dead work.
   - The single-threaded activation quantization (1112-1118) costs 0.32-0.49 ms per window.
   - **Gain:** ~0.3-0.5 ms.

## Speculative ideas

- **CPU pool:** pin the workers to SMT siblings (the kernels are bound by codebook lookups). Maybe 10-20% on the CPU
  share.
- **`--spec-split`:** re-test it on IQ3, now that the CPU share is twice as large.
- **MTP layer's experts:** they keep 675 MiB of VRAM, about 390 cache slots.
- **MMVQ at ncols = 4:** it dips there (after-perm.txt). Try ROWS = 2.
- **`--pcie-frac`:** it is forced to 0 under DEVICE_PLAN=2.

## Measurement pitfalls

- The "CPU experts 5.7 (7 entries), VRAM hits 16.7" figures are counts per layer-window, not ms.
- iq3-ref, the IQ3 baseline, waited 1.9-2.2 ms per window on PLE reads; later runs show ~0.01. Comparisons against it
  are biased by ~5%, so check `[ple]` in every A/B.
- STRATA_VERIFY_PROFILE stamps are kernels of ~2.5 us each, which inflates the small stages.
- The suffix-draft policy's `observe` includes the adapt join, so its cost model sees noise.
- No rocBLAS/hipBLAS on the decode path; the BF16 GEMM issue only affects the prompt path.
- No correctness bug found.
