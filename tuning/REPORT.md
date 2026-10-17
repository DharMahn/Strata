# Strata on the RX 6950 XT (Windows, gfx1030): performance report

2026-10-01. Ryzen 9 9950X, 64 GB DDR5-6000, RX 6950 XT 16 GB, Windows 11, ROCm 7.13, engine 0.1.30 + the owner's
Windows/RDNA2 port. Model: Qwen3.8-Flash-Next IQ2_XS, MTP drafts, 64K context, int8 KV.

## Result

| | baseline (port as found, owner's config) | final (installed) | |
|---|---:|---:|---:|
| decode, short prompts (3 prompts x 512 tokens) | 26.6 t/s | **63.1 t/s** | 2.4x |
| decode after an 8K-token prompt (256 tokens) | 26.9 t/s | **61.1 t/s** | 2.3x |
| prompt reading, 8K tokens | 175 t/s | **401 t/s** | 2.3x |
| run-to-run stability | bimodal: 2 of 10 starts ~35% slower | no slow starts in 30+ runs since the fix | |

Medians of 3 fresh engine starts with `tuning/bench.py` (method below). Through the real server
(`serve/server.py`, OpenAI API, thinking off) a short question was answered at 53 t/s on the first request.

The target was 40-50 t/s. The gap to upstream's RTX 5070 figure (~63 t/s at 64K) is closed for this workload;
it came from three Windows/AMD-specific problems, not from the hardware.

## What was wrong, in order of impact

1. **Windows put part of the "VRAM" in system RAM** (fixed in 9f12dbd). The engine sizes its expert cache from
   `hipMemGetInfo`, which does not see what other programs hold - the desktop compositor alone keeps 1.4 GiB, 1.6
   GiB in all here. Windows does not fail an allocation past the process's budget; it quietly backs it with system
   memory that the GPU reads over PCIe. The last ~1.4 GiB of the expert cache and the MTP draft head (allocated
   after it) were read at 28 GB/s instead of ~565. The draft head alone cost 5.2 ms per draft step (0.27 ms in
   VRAM). Because the prompt path borrows the cache's last slots, it also made the engine bimodal between starts
   (prompt 175 vs 289 t/s, decode 27 vs 17 t/s). Fix: on Windows, `cudaMemGetInfo` (the HIP shim) now reports
   min(HIP's free figure, the DXGI budget left to this process). The engine's own sizing and its existing
   "is the reserve still free after writing the slots" check then work as intended. The startup log prints the
   budget.
2. **HIP's `__byte_perm` is slow with a run-time selector** (fixed in 7afb1ce). HIP implements it by indexing a
   local byte array. The IQ4_XS/IQ4_NL codebook lookup and the Q2_0 unpacking use run-time selectors, so every
   IQ4_XS matrix-vector product (QKV, Q, gate, out-proj, shared expert, and the 337 MB LM head) ran at ~67 GB/s,
   and 17 GB/s at 5-6 tokens per window, against 410 GB/s for the IQ3_S kernel next to it. Fix: `__byte_perm` maps
   to `v_perm_b32`, and the IQ4 lookup takes llama.cpp's own HIP form. Bit-exact: kernel outputs identical (48
   checksums), and engine output token-identical on all prompts under the strict gate. 8-21x faster GEMVs.
3. **The prompt path's MMQ kernels were left out of the Windows build** (b76d2a5, `setup.py`). Upstream uses
   them by default on CUDA, and the Linux AMD build has them. Built in: 8K prompt 323 -> 404 t/s. They use 8-bit
   activations for the prompt's experts, so text after a long prompt differs slightly. Upstream measured the shift
   as about the size of changing the prompt chunk size (bench/results/2026-09-28-prefill-speed: top-1 agreement
   85.9% vs 89.8% for the chunk-size yardstick, perplexity 8.63 vs 9.21 reference). `STRATA_PREFILL_MMQ=0` turns it off.
4. **Settings:** `--pool-workers 12` instead of 8, now that the CPU experts are ~15-23% of a decode step. 12 was
   best of 6/8/10/12/15 in the screening (+4.6%), and +1.2% in the 3-run check. `--vram-reserve-mib 1024` for
   headroom (see the recommended config).

## Time breakdown (one verify window on the essay prompt, ~1.8 tokens out per window)

| | baseline | final |
|---|---:|---:|
| GPU verify phase (host waits for each layer) | 46.8 ms (fast start) / 77.6 ms (slow start) | 16.2 ms |
| MTP draft | 13.1 ms | 2.6 ms |
| CPU experts + per-layer host work | 4.2 ms | 4.6 ms |
| staging + commit | 3.5 ms | 2.7 ms |
| **window** | **75.9 / 107.8 ms** | **31.6 ms** |

The GPU's ~24 ms of the final window: hyper-connection reads + router 6.9, dense GEMVs 6.7, cached experts 4.0,
GPU-side handoff waits ~3.0, attention/recurrence/misc ~3. Nothing dominates anymore. Windows launch-to-sync
latency on top of the draft graphs' GPU time: ~0.37 ms for the round graph, ~0.08 ms per extra step.

Diagnosis order (Phase 2): GPU kernels dominated (~85% of a step at baseline; CPU experts ~5%), so kernel work
came before the settings sweep. The CPU and RAM were never the limit. RAM: 2x32 GB at 6000 MT/s (EXPO is on;
6000 is the usual 1:1 point on Zen 5), no action needed.

## Changes kept (branch `tuning/best`, one commit each, revertible independently)

| commit | what | effect |
|---|---|---|
| e6b7376 | the owner's port, committed as found | baseline |
| 8e8ba0b | `tuning/`: benchmark harness, sweep driver, GEMV microbenchmark | - |
| 7afb1ce | HIP `__byte_perm` -> `v_perm_b32`, llama.cpp's AMD IQ4 lookup | 26.6 -> 41.8 t/s, bit-exact |
| f3d2e62 | diagnostics, off by default: `STRATA_MTP_PROFILE`, `STRATA_VRAM_PROBE` | - |
| 9f12dbd | Windows: VRAM sized by the WDDM budget | 41.8 -> 61.2 t/s, prompt 175 -> 321 t/s |
| b76d2a5 | setup.py: MMQ prompt path on Windows | prompt 323 -> 404 t/s |

## Tried and not kept

- **Hyper-connection down kernel with fewer rows per block** (branch `exp/gr-rows`). 41 blocks leave half the 80
  CUs idle, but the stage did not get faster (1.01 -> 1.10 ms); +1.3% end to end, inside the noise.
- **Settings inside the noise** (2-6% run-to-run, from text-dependent draft acceptance and cache adaptation
  timing): `--spec 5` (+3% screening, +0.4% in 3 runs), `--spec-min-p` 0.5 / 0.6 (+2%), `--pcie-frac 0.2` (+2%),
  `--adapt-every 2` (+0.8%). Worse: `--adapt-every 0` (-5%), 6 workers (-2%), `--spec-min-p 0.8`, `--spec 3`.
- **Fixed VRAM reserves instead of the budget fix:** 2304 MiB was not enough (the draft head still spilled:
  47.6 t/s); 2560-3072 MiB reach the same ~61-62 t/s as the budget fix, but only for today's desktop load.

## Recommended launch config (installed)

`strata-iq2_xs.json` = the owner's arguments with two changes:

    --pool-workers 12          (was 8)
    --vram-reserve-mib 1024    (new; default 700)

The reserve costs ~2-3% against 700 (fewer cache slots). It leaves ~450 MiB of VRAM free instead of ~150, which
clears the engine's own "LOW: requests may stall" warning and gives room for programs opened after the model
loads. Under heavy GPU use by other programs (games, video), expect a slowdown: Windows can still demote
allocations later. Restarting the server re-sizes the cache to the new budget.

Installed: `engine\strata.exe` is the final build (the previous one is `engine\strata.exe.port-original`), and
the config's previous version is `strata-iq2_xs.json.before-tuning`. To undo both, copy those two back.

Do not set an explicit `--expert-cache N` on Windows: an explicit size skips the budget check (one run with
`--expert-cache 5800` spilled and fell to 22 t/s). Keep `auto`.

## Notes for the owner

- The repository is checked out on `tuning/best`. `main` is still upstream's commit, without the port. The port
  as you left it is commit e6b7376 on `tuning/baseline`.
- Correctness: greedy output is reproducible only between starts with the same timing, because adaptive expert
  swaps land asynchronously and GPU and CPU round slightly differently. The kernel fix was verified token-exact
  with `--adapt-every 0 --suffix-draft 0`. Everything else changes which experts the GPU computes (cache size) or
  the prompt numerics (MMQ), which moves greedy text the way any cache-size change does (upstream:
  bench/results/2026-09-27-cache-parity).
- The control vector in the config (`experimental-speed-projection`) is a refusal-direction ablation per its
  README, not a speed setting. It was left on as configured.
- Next levers, if wanted: per-graph launch latency on Windows (~0.5 ms x 3-4 graphs per window); the BF16
  hyper-connection kernels (~7 ms/window at ~150-300 GB/s); the cached-expert kernel's short 640-value down rows.
  Each is single-digit percent.
- Everything is in `tuning/`: LOG.md (every experiment, including failures), results/ (per-run JSON with output
  token ids and engine logs), sweeps/ (the exact configurations run).
