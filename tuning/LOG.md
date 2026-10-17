# Tuning log: Strata on RX 6950 XT (gfx1030), Windows

Machine: Ryzen 9 9950X (16C, Zen 5, AVX-512), 2x32 GB DDR5 at 6000 MT/s, RX 6950 XT 16 GB (gfx1030),
Windows 11, ROCm 7.13 (TheRock wheels), engine 0.1.30 + the owner's Windows/gfx1030 port.
Model: Qwen3.8-Flash-Next IQ2_XS native pack, MTP draft head, 64K context, int8 KV.

Every row is a `tuning/bench.py` result (tuning/results/<name>/summary.md). Numbers are medians of 3
fresh engine starts unless noted. "short" = three chat prompts of ~50 tokens, 512 tokens out each, decode
t/s = all generated tokens / all decode time. "long" = one 7,959-token prompt, 256 tokens out.

## Method

- Harness: `tuning/bench.py` drives `strata.exe --serve` over its stdin protocol (`GEN`/`T`/`DONE`, exactly what
  `serve/server.py` sends), greedy, thinking off, frozen prompt token ids in `tuning/prompts/`.
- Each run = a fresh engine start: warmup (48 tokens out, not scored) -> 3 short -> (full suite) long.
- A run whose log shows a broken start (working-set lock failure, launch failure, OOM, ...) is discarded
  and retried.
- Correctness gate: output token ids against the baseline's (first divergence index per prompt).
- Code experiments: one git branch each, off `tuning/baseline` (e6b7376 = the port as found).

## Phase 1: baseline (tuning/results/baseline, stab-idle, stab-noadapt)

| config | short decode t/s | long (8K) decode t/s | long prefill t/s | notes |
|---|---:|---:|---:|---|
| baseline (owner's config), 3 runs | 26.6 (17.2-26.9) | 26.9 (16.3-27.1) | 175 (172-289) | run 0 in the "slow state" |
| baseline again, agent idle, 3 runs | 25.5 (25.0-25.5) | - | - | spread 1.9% |
| --adapt-every 0, 3 runs | 25.3 (16.2-27.4) | 26.5 (17.4-28.4) | 177 (175-287) | run 2 in the "slow state" |

**Stability: bimodal.** 2 of 10 engine starts so far landed in a "slow state": decode ~35% slower, prompt reading
~65% FASTER, the same work (identical windows, hits, acceptance). Only the verify window's GPU phase is slower; the
MTP draft graph (another stream) is not. It happened with the agent idle too, so it is not desktop contention. It is
decided at engine start (each run is all-slow or all-fast). Not explained yet; a median of 3 is robust to one slow
run.

**Determinism.** Runs in the same state are bit-identical. Adaptive expert swaps land asynchronously, so the cache
contents (GPU vs CPU rounding) depend on timing: a slow run diverged at token 41-62 on three prompts.
`--adapt-every 0` removes most of that (stab-noadapt), but its slow run still diverged on one prompt at token 60.

## Phase 2: where the time goes (STRATA_DECODE_TIMING + STRATA_VERIFY_PROFILE, tuning/results/prof-idle)

Per verify window on the essay prompt (fast state, avg 2.27 tokens/window, 1.88 tokens out/window):

| part | ms/window | share |
|---|---:|---:|
| GPU: verify window (host waits for the GPU to reach each layer) | 54 | 63% |
| GPU: MTP draft | 13 | 15% |
| CPU experts (pool, overlapped with the GPU's own expert hits) | 4 | 5% |
| GPU: commit + host staging | 3 | 4% |
| rest (per-layer host overhead, plan, sync) | ~11 | 13% |

GPU stages inside the window (73 ms/window of GPU time): dense projections dominate. GDN QKV GEMV 15.1, GDN gate (z)
7.5, QSA Q 5.8, LM head 7.2, shared expert 7.4, router + hyper-connection reads ~5, VRAM-cached experts 9.7,
handoff waits ~1.6.

**Diagnosis: GPU kernels dominate, so Phase 4 (kernels) is justified ahead of the CPU/RAM settings.** The CPU
experts are ~5% of decode.

Kernel micro-benchmark (`tuning/micro/mmvq_bench`, real shapes, back to back): the IQ4_XS GEMV (QKV, Q, gate,
ssm_out, shared gate/up, **LM head**) ran at ~67 GB/s at 1-3 tokens and ~17 GB/s at 5-6, while the IQ3_S one (a
different kernel family) ran at 410 GB/s. Cause: HIP's `__byte_perm` indexes a local byte array; with a run-time
selector (the IQ4 codebook lookup `get_int_from_table_16`, the Q2_0 crumb unpacking) that is a long 64-bit
shift/mask sequence per byte instead of one `v_perm_b32`. llama.cpp's HIP backend has the v_perm form; Strata's port
of those kernels did not.

## Experiments

| # | name | change | short t/s | long decode t/s | long prefill t/s | output vs baseline | verdict |
|---|---|---|---:|---:|---:|---|---|
| 0 | baseline | - | 26.6 | 26.9 | 175 | reference | - |
| 1 | (micro) hip-perm | v_perm `__byte_perm` + llama.cpp's HIP IQ4 lookup | IQ4_XS GEMV 8-21x faster, LM head 68 -> 557 GB/s | | | all 48 kernel outputs bit-identical | engine run below |
| 1 | hip-perm | the same, in the engine (commit 7afb1ce) | **41.8** (41.8-41.8) | **42.3** | 175 | strict gate (no adaptation, no suffix drafts): token-identical on all 5 prompts, 2 runs | **KEEP** |

Correctness gate, refined: `--adapt-every 0 --suffix-draft 0` (gate-ref = baseline exe). With adaptation off but
suffix drafts on, the code prompt still diverged at token 60 in BOTH the unmodified engine's slow run and the fixed
engine: the suffix-draft policy picks windows from measured timings, so a speed change alone moves that prompt.

## Phase 2 again, after hip-perm (prof-hip-perm, prof-mtp*)

Window 51 ms (essay): GPU verify 25 ms, **MTP draft 12.7 ms (25%)**, CPU experts 4 ms. The new MTP stage profiler
(`STRATA_MTP_PROFILE=1`) showed **5.3 of a draft step's 5.9 ms in the draft head** (a 145 MB IQ4_XS GEMV over the
106K-token draft vocabulary) - 28 GB/s. The same kernel on the same stream over the main head ran at 546 GB/s.

**Root cause: WDDM backs device memory past the process's budget with system RAM.** The VRAM probe
(`STRATA_VRAM_PROBE=1`) read the expert cache at ~565 GB/s per 256 MiB chunk except the last ~1.3-1.5 GiB: 28-29 GB/s
(PCIe). The draft head (allocated after the cache) was there too. `hipMemGetInfo` does not see what other programs
hold (the desktop compositor alone kept 1.4 GiB of VRAM; 1.6 GiB in all with the engine closed), so the auto cache
was sized ~1.5 GiB past the real budget, and the engine's own WDDM check (zero the slots, read the free figure again)
could not see it either. This also explains the bimodal "slow state": the prompt path borrows the cache's LAST slots,
so a start where the spill landed in the cache tail had slow prompts / fast decode, and one where it landed in
something decode needs had fast prompts / slow decode.

| VRAM reserve (probe runs, 1 each) | spilled | short t/s | long t/s | prefill t/s |
|---|---|---:|---:|---:|
| 700 (owner's) | cache tail ~1.4 GiB + draft head | 41.8 | 42.3 | 175 |
| 1536 | cache tail ~0.4 GiB + draft head | 42.0 | 40.1 | 320 |
| 2048 | draft head | 47.9 | 22.9 | 279 |
| 2560 | nothing | 61.6 | 61.2 | 320 |
| 3072 | nothing | 62.2 | 59.8 | 320 |

| # | name | change | short t/s | long decode t/s | long prefill t/s | output vs reference | verdict |
|---|---|---|---:|---:|---:|---|---|
| 2 | res2304 | hip-perm + --vram-reserve-mib 2304 | 47.6 | 49.6 | 321 | cache size differs: rounding-level divergence (expected) | partial (draft head still spilled) |
| 2 | res2560 | hip-perm + --vram-reserve-mib 2560 | 61.1 | 60.2 | 320 | " | settings-only alternative |
| 2 | res2816 | hip-perm + --vram-reserve-mib 2816 | 61.6 | 61.4 | 320 | " | settings-only alternative |
| 2 | res3072 | hip-perm + --vram-reserve-mib 3072 | 62.6 | 60.4 | 320 | " | settings-only alternative |
| 3 | wddm | cudaMemGetInfo capped by the DXGI budget (commit 9f12dbd), owner's reserve 700 | **61.2** (60.6-61.6) | **63.1** | **321** | " | **KEEP** (automatic; no spill per the probe) |
| 3 | wddm-res1024 | the same + --vram-reserve-mib 1024 | 59.3 | 60.2 | 316 | " | recommended for daily use: 469 MiB left instead of 146 ("LOW") |

With the fix the engine prints e.g. `WDDM local budget 15.20 GiB, this process uses 5.39 GiB`, sizes the cache to
6389 slots, its own post-write check (now seeing the real figure) shrinks it to 5958 (7.99 GiB), and the probe reads
every chunk of the cache at 544-573 GB/s, the draft head GEMV at 608-666 GB/s.

After it (probe-wddm, essay): window 32 ms = GPU verify ~16 + CPU experts/host 4.7 + MTP draft 2.9 (was 12.7) +
commit/stage 2.7. Each MTP graph: 0.93 ms GPU + ~0.6 ms launch-to-sync overhead (Windows submission latency).

## Phase 3: settings, one at a time (tuning/sweeps/phase3.json; strata-wddm.exe; quick suite, 2 runs each)

Flags checked in generate.cpp first: `--pool-workers N` CPU expert pool threads, pinned one per physical core from
core 1 (0 = all but the host's; the owner's config had 8, the log shows 15 and 10 tried before); `--spec S` the MTP's
window cap (suffix-draft windows may go to S+2); `--spec-min-p p` a draft chain continues while the last draft's
probability >= p; `--pcie-frac f` the share of the VRAM-missed experts the GPU reads over PCIe (only experts in the
7 GiB pinned arena slices, layers 0-11, qualify) instead of the CPU computing them; `--adapt-every N` swap the
most-routed missing experts into the VRAM cache every N windows (default 4).

| change | short t/s | vs wddm (61.2, spread 1.6%) | accept | hit |
|---|---:|---:|---:|---:|
| --pool-workers 6 | 59.9 | -2.2% | 0.78 | 0.77 |
| --pool-workers 10 | 63.1 | +3.0% | 0.79 | 0.76 |
| **--pool-workers 12** | **64.0** | **+4.6%** | 0.79 | 0.77 |
| --pool-workers 15 | 63.6 | +3.8% | 0.78 | 0.76 |
| --spec-min-p 0.50 | 62.7 | +2.4% | 0.68 | 0.76 |
| --spec-min-p 0.60 | 62.4 | +1.9% | 0.71 | 0.76 |
| --spec-min-p 0.80 | 60.9 | -0.5% | 0.83 | 0.77 |
| --spec 3 | 61.3 | +0.1% | 0.81 | 0.77 |
| --spec 5 | 63.0 | +3.0% | 0.75 | 0.76 |
| --adapt-every 0 | 58.1 | -5.2% | 0.77 | 0.48 |
| --adapt-every 2 | 61.7 | +0.8% | 0.78 | 0.79 |
| --pcie-frac 0.20 | 62.4 | +2.0% | 0.77 | 0.77 |

## Phase 4 attempt that failed: hyper-connection down kernel rows per block (branch exp/gr-rows, 1597376)

`gr_down_multi_kernel` reads the BF16 hc down weights (6.5 MB per read, ~1.3 GB per window with all the hc reads)
with 40 + 1 blocks of 8 rows - half of the 80 CUs idle. Made rows per block a template (8/4/2, `STRATA_GR_ROWS`);
each row's arithmetic is unchanged (gr4 matched the reference tokens exactly). Result: hc0 down stage 1.01 ms/window
(8 rows) vs 1.10 (4 rows); end to end +1.3% (4) / +0.3% (2) - inside the noise. **Not kept.**

Lesson from its profile runs: `--expert-cache 5800` (an explicit size) became 6087 variable-size slots, PAST the
WDDM budget: things spilled again (22.5 and 45.8 t/s, hc stages 5-10x slower). An explicit cache size skips the
auto sizing's budget check - on Windows keep `--expert-cache auto`. The budget-capped auto size also varies a little
between starts (5863-5960 slots seen), so a token-exact gate needs a pinned size that is safely under the budget.

## Settings combined (tuning/sweeps/combos.json; full suite, 3 runs each; strata-wddm.exe)

| config | short t/s (spread) | long t/s (spread) | prefill |
|---|---:|---:|---:|
| wddm (owner's settings) | 61.2 (1.6%) | 63.1 (0.9%) | 321 |
| --pool-workers 12 | 61.9 (1.9%) | 64.5 (5.8%) | 321 |
| + --spec 5 | 61.5 (7.4%) | 62.2 (0.6%) | 320 |
| + --spec-min-p 0.60 | 62.9 (1.5%) | 62.8 (6.0%) | 317 |
| + --pcie-frac 0.20 | 62.8 (3.7%) | 60.7 (5.4%) | 316 |
| --pool-workers 12 --vram-reserve-mib 1024 | 61.3 (3.5%) | 61.7 (3.2%) | 320 |

With the GPU fixed, run-to-run variation is 2-6% (text-dependent draft acceptance, adaptation timing), larger than
these settings' effects. Kept: 12 workers (ahead in both sweeps) and reserve 1024 (headroom, see REPORT.md).
Owner's spec / min-p / pcie-frac kept.

## MMQ prompt path (commit b76d2a5; tuning/results/mmq-*)

Built with -DSTRATA_PREFILL_MMQ=ON (the Linux AMD build has it; the Windows setup left it out). Read prefill.cpp
first: once built, MMQ is ON unless STRATA_PREFILL_MMQ=0 (the first "off" arm was in fact on - redone).
Same exe, 8K prompt: MMQ 404 t/s, STRATA_PREFILL_MMQ=0 323 t/s. Decode unchanged. Long-prompt answers differ from
token 1-24 (int8 activations in the prompt's experts) and read coherently. **KEEP**.

## Final (tuning/results/final; strata-final.exe = 9f12dbd + MMQ; tuning/configs/final.json)

| | baseline | final |
|---|---:|---:|
| short decode | 26.6 | **63.1** (61.2-63.2) |
| 8K decode | 26.9 | **61.1** (45.0-61.3; the 45.0 run wrote a different answer from token 1, acceptance 0.60) |
| 8K prompt | 175 | **401** (401.0-401.4) |

Smoke test through serve/server.py with the installed engine and config (OpenAI API, thinking off): correct answer,
53 t/s decode for 82 tokens, 22-token prompt at 75.5 t/s.

## Large pages (tuning/results/lp-*, nolp-*; after granting "Lock pages in memory" and a reboot)

Alternating starts, 1 run each x 3 rounds, every lp run on a large-page arena. Decode same within noise (short 63.4 vs
63.4, long 64.2 vs 62.7), prompt 402 vs 398, startup ~21 vs ~24 s. With the privilege but fragmented memory (4 h
uptime) the large-page VirtualAlloc fails with 1450 and costs ~8 s of startup, falling back to 4 KB pages. With large
pages the later SetProcessWorkingSetSizeEx fails (1450) harmlessly: large pages cannot be paged out (bench.py no
longer counts that as a broken start then).

## PAUSED 2026-10-01 (owner moved to the 7900 XTX box) - where things stand

Branch `exp/decode-opts` (WIP commit, see git log):
- built + measured by `tuning/sweeps/opts-flags.json` (results in tuning/results/opts-*, gate2-*, flag-*; NOT yet
  analysed): cached-expert kernels with a fixed grid height (`STRATA_HITS_GY`, default 16, 0 = old), async commit
  (`STRATA_ASYNC_COMMIT=1`), and the flag tests --spec-split, --kv k8v4, --kv-resident 20480, --ple-io mmap,
  --ple-row-cache 4194304. Exe: tuning/bin/strata-opts.exe.
- written, NOT built: STRATA_VERIFY_DEVICE_PLAN=2 (every layer's VRAM groups planned on the device; d_res uploaded
  on eviction), fused_gr TILE 2560 on HIP (bit-exact by construction).
- not written yet: gr_norm one block per stream (bit-exact replication of the per-thread order), gr_up next-row
  prefetch, PLE prefetch (PleReader issue/collect), pin-limit tests (STRATA_PIN_LIMIT_GIB 16/24/0, retries 0), IQ3_S
  multi-column kernel.

## Resumed 2026-10-01 afternoon

### opts-flags results (measured before the pause; strata-opts.exe, final config, full suite)

| run | change | short t/s (min-max) | long t/s (min-max) | verdict |
|---|---|---|---|---|
| opts-ref | (STRATA_HITS_GY=0 = old launch) | 62.6 (61.8-63.7) | 63.6 (62.0-64.9) | reference |
| opts-gy16 | expert kernels grid height 16 | 63.2 (62.8-64.4) | 61.6 (60.2-64.0) | noise; default set back to 0 |
| opts-async | STRATA_ASYNC_COMMIT=1 | 65.5 (64.8-65.5) | 64.8 (64.3-65.3) | +4.6% short, gate2-async token-identical: keep |
| opts-both | grid 16 + async | 65.0 | 66.0 | = async |
| flag-specsplit | --spec-split | 55.1 | 55.6 | worse |
| flag-k8v4 | --kv k8v4 | 64.0 | 62.2 | no |
| flag-kvres | --kv-resident 20480 | 65.9 (65.8-66.0) | 64.9 (64.8-65.1) | +5% short: retest with 32768 |
| flag-plemmap | --ple-io mmap | 55.3 | 49.5 | worse |
| flag-plecache | --ple-row-cache 4194304 | 62.9 | 63.4 | no |

### Ported from the 7900 XTX work, gated (strict mode, --expert-cache 4500, all equal gate2-ref token for token)

strata-opts2.exe (commit 0586676): SWAR __vsub4/__vcmpne4 (the IQ2_XS expert dot products use them) + fused_gr TILE
2560 -> g3-base identical; STRATA_PLE_ASYNC=1 (the window's PLE rows read while layer 0 runs) -> g3-ple identical;
STRATA_VERIFY_DEVICE_PLAN=2 -> g3-dplan identical.  The PLE timer shows the whole host "stage" (1.5-1.7 ms/window)
is the PLE read on this PC.

### The draft head in system RAM when other programs hold VRAM (commit cdecbcf)

The first speed runs of the new build gave 47 t/s instead of 65 - and so did the OLD build (chk-old-now): the
environment.  With League/Riot client, Discord, Edge, MOZA Pit House and RSI Launcher open, other programs held
2-3.7 GB of VRAM (dwm alone 2.2 GB).  STRATA_MTP_PROFILE: the draft head's GEMV took 5.2 ms at 28 GB/s (PCIe) instead
of 0.22 ms - it was backed by system RAM; draft 11-14 ms/window instead of 2.8.  --vram-reserve-mib 2560 put it back
in VRAM (59 t/s; chk-res2560-now).  STRATA_WDDM_DEBUG traces showed why the budget check missed it: the DXGI budget
only shrinks once the process presses on it (15.20 GiB idle -> 13.74 GiB after the cache allocation with 1.95 GiB
held by others), so the cache, sized before, overran.  Fix in the HIP shim's cudaMemGetInfo: limit = min(budget now,
largest budget seen - other programs' dedicated VRAM), the latter from the PDH counters GPU Adapter Memory minus GPU
Process Memory (what Task Manager shows).  "budget - others" double counted them after the shrink (cache cut twice
by the post-allocation check, chk-others3).  Result (chk-others4, same load): cache 5004 slots, draft head in VRAM,
59.3 t/s.  STRATA_WDDM_OTHERS=0 = the old behaviour.

### opts3 sweep (strata-opts3.exe = cdecbcf, final config + STRATA_ASYNC_COMMIT=1; ~2 GB of VRAM held by others)

| run | change | short t/s (min-max) | long t/s (min-max) | 8K prompt t/s | hit |
|---|---|---|---|---:|---:|
| o3-ref | VRAM cap for other programs (5008 slots) | 59.8 (54.2-61.7) | 59.8 (59.2-60.1) | 406 | 0.72 |
| o3-noothers | control: STRATA_WDDM_OTHERS=0 (5425-5441 slots) | 54.5 (47.4-61.6) | 54.3 (47.7-61.0) | 404 | 0.75 |
| o3-ple | + STRATA_PLE_ASYNC=1 | **64.2** (62.9-65.7) | **66.7** (65.6-66.9) | 397 | 0.72 |
| o3-dplan | + STRATA_VERIFY_DEVICE_PLAN=2 | 61.4 (61.1-61.4) | 60.7 (60.3-61.1) | 396 | 0.71 |
| o3-kv20 | + --kv-resident 20480 | 63.3 (63.1-63.5) | 60.2 (59.5-62.1) | 391 | 0.75 |
| o3-kv32 | + --kv-resident 32768 | 59.6 (58.2-62.8) | 63.8 (60.7-63.9) | 390 | 0.74 |

- The control is the proof of the VRAM fix: same load, run 0 fine (61.6), run 1 with the draft head in system RAM
  (draft 13.5-14.3 ms/window, 47.4 t/s) - the old sizing is a coin flip once others hold ~2 GB.  The fix costs
  ~0.4 GiB of cache here (it assumes the OS gives back all of the others' share; it gave back ~75%).
- PLE async: host stage 1.7-2.2 -> 0.02 ms/window (the reads finish behind layer 0; collect waits <= 0.25 ms).  Win.
- Device plan for every layer: +2.7% / +1.5%, tight ranges, exact.  Keep (its target, a slow host round trip, is
  far bigger on the 7900 XTX box).
- KV streaming: ~+3% on average either way, inside the run-to-run spread, and -4% prompt speed; long contexts would
  stream attention from RAM.  Not adopted.

### Combined + defaults (commit 1ea8484, strata-opts5.exe, installed as engine\strata.exe)

g4-all (async commit + PLE async + device plan, strict) and g5-defaults (no env at all) both equal gate2-ref token
for token.  o4-all: 63.6 short (61.8-64.1) / 64.1 long (58.8-66.6) vs o3-ref 59.8 / 59.8 - about PLE async alone
(64.2 / 66.7); the device plan adds nothing measurable here.  Previous engine kept as engine\strata.exe.tuning-final.

### Pin limit (tuning/sweeps/pin.json; STRATA_PREFILL_TIMING=1)

p-ref (8 GiB, 12 slices / 7 GiB pinned): 8K prompt 393 t/s, GPU timeline 19.7 s of which "wait copy" 2.4 s (12.4%).
p-pin16 (23 slices / 15 GiB pinned): 392-394 t/s, wait copy 2.6 s - no change: the prompt path's expert stream is
bound by PCIe bandwidth, not by staging the unpinned share.  (Both runs flagged invalid only for the harmless
"VirtualLock stopped ... error 87": large pages cannot be VirtualLocked.)  p-pin24: "unspecified launch failure" at
the first copy - the failure the source comment describes for a fully pinned arena; the engine hung and was killed,
no driver reset.  Keep the 8 GiB default.

### IQ3_XXS on this PC (tuning/sweeps/iq3.json; tuning/configs/iq3.json = final.json with the IQ3_XXS paths)

Fetched by tuning/get_iq3.py (setup.py's steps 5-6 without its engine build or settings: shard 2 hard-linked from
IQ2_XS - the same file for every size; shard 1, 47 GB, in ~8 min; pack by tools/iq_pack.py).  Full RAM mode (42.9 GB
arena, 64 GB PC).  strata-opts5.exe, same settings as IQ2_XS:

| | IQ2_XS (p-ref) | IQ3_XXS (iq3-ref, 3 runs) |
|---|---:|---:|
| short decode | 63.8 | 54.5 (51.4-55.9) |
| long decode | 63.8 | 54.1 (51.7-55.2) |
| 8K prompt | 393 | 405 (386-406) |
| hit rate | 0.72-0.74 | 0.61-0.62 |

-15% decode, prompt unchanged.  About what the 7900 XTX server does with the same model after its tuning (55.9 / 55.0).

## Phase 3, continued

Workers: 12 of the 16 cores is the best; 8 left CPU expert time on the table now that the GPU is fast (the code prompt
spends ~23% of a window in the CPU pool). The others are within ~2x the noise; they are tried together below.
RAM: 2x32 GB Patriot "6400 Series" configured at 6000 MT/s (WMI); EXPO/XMP is evidently on (JEDEC would be 4800 or
5600), and 6000 is the usual 1:1 point on Zen 5 - no action suggested.

## Prompt path, 2026-10-02 (IQ3_XXS; tuning/sweeps/stream.json, p2.json; STRATA_PREFILL_TIMING=1)

Starting point (strata-opts5 / installed): 8K prompt 395 t/s, GPU timeline 19.6 s - gdn projections 4.8 s, wait copy
3.0 s, hc read 2.2 s, qsa proj 1.9 s, qsa attn 1.7 s, gdn out proj 1.3 s.  The experts' MMQ was ~2.5 s of it.

**rocBLAS on gfx1030** (tuning/micro/gemm_bench.cu, results/gemm-bench*.txt).  The ROCm 7.13 rocBLAS ships tuned
gfx1030 kernels only for HH (FP16 in and out) and SS (FP32); the engine's BF16->FP32 (BS) and FP16->FP32 (HS) calls get
the generic fallback kernels: 6-10 TFLOPS on every shape.  SS 21-23 TFLOPS steadily (hc down, K 10240 -> N 320: 8-15),
HH 35-40 but with holes at some row counts (gdn qkv at 512/1000 rows, hc down below 5000: back to ~5).

**Copy events** (tuning/micro/copy_bench.cu, results/copy-bench.txt).  Pinned H2D copies: 28 GB/s at every size from
256 KB.  An event recorded after every 1.7 MB copy: 15 GB/s; with a second stream waiting on each: 11 GB/s - the
prompt path's streamed ring did exactly that per expert (and the stager recorded a second event per unpinned blob).
An event per 8 copies: 22 GB/s, per 16: 24.  Not the host copies: STRATA_STAGER_THREADS=8, STRATA_PREFILL_RING=192,
STRATA_PIN_LIMIT_GIB=16 all left wait copy at 3.05-3.2 s (stream.json; pin16 cut host staging 10.0 -> 7.0 s only).

Changes (all default on, each with an off switch):

| change | switch | 8K prompt t/s | what moved |
|---|---|---:|---|
| (installed, strata-opts5) | | 395 | |
| dense GGUF matrices through llama.cpp's MMQ (+ Q4_K/Q5_K/Q6_K instances) | STRATA_PREFILL_DENSE_MMQ=0 | 589-605 | gdn 4.8 -> 0.9 s, qsa proj 1.9 -> 0.5, gdn out 1.3 -> 0.45, router+shared 0.9 -> 0.5 |
| + copy events per group of 8 experts | STRATA_PREFILL_COPY_GROUP=1 | 747 | wait copy 2974 -> 435 ms, dequant 1075 -> 640 |
| + BF16/FP16->FP32 GEMMs widened to FP32 (tuned SGEMM) on gfx103x | STRATA_GEMM_WIDEN=0 | 642 alone | hc read 2147 -> 1670 ms, router+shared 492 -> 272, ple 111 -> 37 |
| all three (strata-p2.exe) | | **812** (808 with groups of 16) | GPU timeline 19.6 -> 9.3 s |

Quality (tuning/prefill_kl.py: the prompt path's own logits at every 16th position of the 8K prompt via the
STRATA_PREFILL_DUMP_LOGITS hook - a native pack dumps no per-token logits otherwise - against the FP16 path):

| arm | same top-1 | mean KL | p99 KL | ppl (ref 16.68) |
|---|---:|---:|---:|---:|
| chunk 4096, same code (yardstick) | 93.6% | 0.104 | 2.24 | 15.25 |
| MMQ experts (installed) | 93.4% | 0.100 | 2.57 | 16.24 |
| + dense MMQ | 91.4% | 0.143 | 2.27 | 15.84 |
| FP16 path + widening only | 93.4% | 0.047 | 0.94 | 15.87 |
| strata-p2 (all three) | 91.6% | 0.199 | 3.28 | 15.02 |

Copy groups are bit-identical (residuals and logits, groups of 1 vs 8).  Widening alone sits at half the yardstick.
Dense MMQ rounds the activations to q8_1 as decode's matrix-vector kernels already do; with everything it is ~2x the
yardstick's KL at the same perplexity (one prompt, 498 positions - noisy).  Upstream accepted MMQ experts at
85.9% vs 89.8% top-1 against its yardstick.

Remaining at 812 t/s (9.3 s): hc read 1.7 s (hc down's skinny shape), qsa attn 1.6 s (no tensor cores on RDNA2),
gemm gate/up 1.1 s + down 0.5 s (MMQ experts), gdn 0.9 s, dequant (the experts' gather) 0.66 s.

### Prompt attention and the hyper-connection products (2026-10-02, after the reboot; sweeps p3, p4, p4b, p4c)

**The GPU fault** (one run at ~00:20 left HIP failing every allocation, "unspecified launch failure", in every new
process until a reboot; no TDR or display event in the system log): the configuration of that run (strata-p2,
STRATA_PREFILL_MMQ=0 STRATA_PREFILL_DENSE_MMQ=0) ran clean after the reboot twice - serialized
(AMD_SERIALIZE_KERNEL=3) and at full speed, bit-identical - and strata-p2 with its defaults ran 3 clean bench starts
(819-822 t/s).  Not reproduced; treated as a one-off.

**Prompt attention on RDNA2.**  The prompt path's tensor-core attention (qsa_prompt_attn.cu) is CUDA-only, so on HIP
every prompt ran the decode kernel 32 queries at a time (1.63 s of an 8K prompt).  New HIP kernel in the same file
(rdna::attn_kernel, int8 KV): a block per (query, KV head), chunks of 32 cells with an online softmax, FP32 FMAs in
register tiles; codes kept biased by 128 (one v_cvt_f32_ubyte each), the bias taken out with sums of q / p.
tuning/micro/attn_bench.cu (synthetic 8K prompt, shuffled pages, 1% masked): 133 -> 59 ms per layer, outputs within
2.7e-6 (relative L2) of the decode kernel.  v1 (wave per 32-dim slice + LDS partials, 201 VGPRs, 4 waves/SIMD) and
v2 (wave per 4 cells + shuffle reduction + prefetched loads, 157 VGPRs, 6 waves/SIMD) are equally fast: the limit is
instruction issue (~half the issue rate; the int8->float conversions are repeated by 4 lanes).  Engine: qsa attn
1631 -> 726 ms, 8K prompt 808 -> 896 t/s.  STRATA_PROMPT_ATTN_RDNA=0 restores the decode kernel.

**hc read split** (new timer lines: hc down / hc up / hc inject / hc mix): down 798-896 ms, up 238, inject 380, mix
150, norm 55.  The inject (10240 -> 4) and the shared expert's gate (2560 -> 1) now take a small-N kernel (a wave
per row reading BF16 directly): inject 380 -> 57 ms.  STRATA_GEMM_SMALL_N=0 restores the widened route.

**hc down kernel - tried, not kept.**  A dedicated BF16 kernel for 10240 -> 320 (32 rows x all 320 outputs per block,
K tiles of 32 through LDS, 4x10 per thread): standalone 4.3 ms (12 TFLOPS, tuning/micro/skinny_bench.cu) against
~8 ms on the widened route; 64-row blocks 4.8 ms.  Inside the engine no gain: with it, hc down -300 ms but hc up
+390 ms (HIP's event timestamps do not separate the kernels cleanly), GPU timeline 8.05 -> 8.10 s (32-row) and
8.44 s (64-row).  Every block re-reads all 6.5 MB of W; the suspicion is cache pressure from the expert stream and
the 163 MB X, not verified.

| | 8K prompt t/s | GPU timeline |
|---|---:|---:|
| strata-p2 (dense MMQ, copy groups, widening) | 808 | 9.37 s |
| + RDNA prompt attention (p3) | 896 | 8.41 s |
| + small-N kernel (strata-p5, installed) | **933** (916-945, 3 starts) | ~8.0 s |

strata-p5 decode: short 57.4, long 56.9 t/s (p2: 57.8 / 57.5 - unchanged, within the run spread).  Quality
(prefill_kl.py, against the FP16 path): same top-1 92.6%, mean KL 0.107 (the chunk-size yardstick: 0.104), ppl 16.0
(reference 16.7).  The single-prompt perplexity moves between 14.4 and 17.6 across arms: noise at this sample size.

Remaining (8.0 s): gemm gate/up 1.1 s + down 0.5 s (experts' MMQ), hc down + up ~1.1 s, gdn 0.9 s, qsa attn 0.73 s,
dequant (experts' gather) 0.72 s, gdn recurrence 0.49 s, gdn out proj 0.45 s, qsa proj 0.48 s.

### The experts' gather, batched (sweeps p6, p7)

"dequant" in the MMQ path is not a dequantization: each expert's blob is copied from its ring slot into the MMQ group
buffer, ~24,500 small copy kernels per 8K prompt at ~29 us each (0.62-0.72 s; the bytes need ~0.2 s).  Now the
pending experts are copied in one launch (mmq::gather_native_batch, a slot index per entry).  First version (p6)
deferred the ring slots' release events to the group's end: NOT deterministic (two runs differed from position 0) -
the host told the copy thread a slot was consumed before its release event was queued, so a refill could overtake
the gather.  Fixed (p7): pending gathers are flushed before any release is recorded (batches of ~8 instead of 16);
bit-identical to strata-p5 (residuals and logits) and run-to-run.  dequant 625 -> 207 ms, wait copy 414 -> 621 ms
(the stream is the limit more often), 8K prompt 951 -> 969 t/s (2 starts each).  STRATA_PREFILL_RING 192 / 384 on
top: 972 / 969, inside the noise - defaults kept.  Installed: engine\strata.exe = tuning/bin/strata-p7.exe.

## Decode: the adaptive tier's join (2026-10-02; tuning/decode-review-2026-10-02.md item 1; sweeps p8, p9)

A read-only review of the decode path (tuning/decode-review-2026-10-02.md) noticed ~3 ms per window that no timer
covered (window - verify - commit - draft: 3.0-3.8 ms in strata-p5's runs).  New timer fields in the decode timing
line ("before the window", "adaptive join") put it on the adaptive tier: every 4th window adapt() issues up to 96
refill copies of 1.4-2.3 MB, mostly from the unpinned arena (HIP stages pageable copies through the calling
thread), then the loop joined the thread after the draft - ~12 ms every 4th window.  --adapt-swaps (p8): 96 59.0 /
56.2 t/s short / long (join 3.1 ms/window, hit 0.67), 48 60.1 / 58.9 (1.8, 0.63), 24 60.6 / 58.9 (0.9, 0.55).

Fix (serve path): the thread lives across windows.  It selects the swaps, marks the evictions in the host table
and uploads the device table (as before, before any copy could land), signals "published", then issues the copies;
the loop waits only for the publication, and apply_pending admits the refills once the thread is done and adapt_ev
has completed.  A window still never reads a slot being refilled: the evictions are published before the copies are
even issued.  STRATA_ADAPT_JOIN=1 restores the join (the A/B).  p9 (2 starts each, full suite):

| | short decode | long decode | hit (short) | join ms/window |
|---|---:|---:|---:|---:|
| joined (old) | 59.2 | 57.9 | 0.68 | 3.1 |
| **background (strata-p9, installed)** | **62.8** | **64.7** | 0.68 | 0.0 |
| background, 48 swaps | 62.2 | 63.0 | 0.63 | 0.0 |

8K prompt unchanged (972-974 t/s).  Generated text reads normally (essay, code, the 8K prompt's answer).

### Decode: the device plan after the doorbell, no empty PCIe call (review items 4 and 5; sweep p10)

verify.cpp: in the default device-plan mode the layer's VRAM plan (resident_plan_partial, a single-thread kernel)
ran between top-10 and doorbell_publish, so the CPU's experts started after it; now it runs right after the doorbell
(only the grouped VRAM call reads it).  And the PCIe grouped call (4 kernels per group, its share always empty in
this mode: --pcie-frac is forced to 0) is skipped.  Strict gate (--adapt-every 0 --suffix-draft 0
--expert-cache 2600, strata-p9 vs p10): token-identical on all 5 prompts.  GPU-reach wait 16.4-17.3 -> 15.4-16.5 ms
per window; short decode 62.6 -> 65.0 t/s, long 64.5 -> 63.0 (its usual spread).  Installed: strata-p10.exe.

### Decode: the hyper-connection norm in registers (review item 3a; sweeps p11, p12)

gr_norm_multi_kernel wrote r * w_norm to global, then scaled it by rs in a second global pass; now the 10 float4 per
thread stay in registers until rs is known and are written once (the same two roundings).  Strict gate token-identical
on all 5 prompts.  STRATA_VERIFY_PROFILE: hc0 norm 0.77 -> 0.50 ms/window (GDN layers), 0.14 -> 0.06 (QSA layers);
short decode 62.5 -> 64.3 t/s, long 62.3 -> 63.0 (3 starts each, profiler on in both arms).  Tried and reverted:
gr_up_multi with two rows per pass (both rows' weights loaded first) - bit-exact but slower (hc0 up 0.79 -> 0.88).
Installed: strata-p12.exe.

### Background adaptive swaps: two problems, now opt-in (STRATA_ADAPT_BG=1)

1. **Graph capture broken** (final-1002, 1 of 4 starts): "map_ids: operation failed due to a previous error during
   capture" - the engine exited.  The captures are in thread-local mode, but on HIP (Windows) the adaptive thread's
   pageable host->device copies, now overlapping the decode loop, still broke a capture of a new window size.
   Fixed with a process-wide lock (hip_compat/cuda_runtime.h: every cudaStreamBeginCapture..EndCapture holds it; the
   adaptive thread takes it around its table upload and each copy).  p13-stress: 10 of 10 starts without a capture
   error.
2. **NaN prompt** (p13-stress run 8 of 10): the 8K prompt's answer was 256 x token 0 ("!!!"), its prompt read at 673
   instead of ~972 t/s and its decode hit rate 94.6% - NaN hidden states, routing collapsed onto a few experts.  A scan
   of every run json today: the only degenerate output.  Not explained (the request start joins the thread and waits
   for its copies before lending slots to the prompt path; the windows read the published tables).  Starts with the
   background swaps: ~28, one NaN; without: ~60, none.

So the swaps are joined after the draft again by default (the old behaviour; the restructured adapt() publishes
first, then copies, then the thread is joined - equivalent).  p14-stress (default, 10 starts): no capture error, no
degenerate output; 8K prompt 972-975, short decode 61.5-62.8, long 59.2-63.0 t/s.  Installed: strata-p14.exe.
With STRATA_ADAPT_BG=1: short ~66.6, long ~67 (p13-stress).

p14-bg-stress (STRATA_ADAPT_BG=1, 20 starts, full suite): no degenerate output, no capture error; short 66.7, long
66.2 t/s, prompt 973.  So with the background swaps: 1 NaN prompt in ~48 starts in all; without: none in ~70.  Not
enough to blame or clear them - they stay opt-in.  To chase it: STRATA_DBG_NAN=1 (reports the first prompt-path layer
with non-finite values) in a long stress run, with and without STRATA_ADAPT_BG.

### The "!!!" hunt (2026-10-02 evening): not reproduced; background swaps on in the server configs

Tools (all kept): `tuning/nan_stress.py` - one engine, many [3 x 512-token decode, fresh 8K prompt] cycles (each 8K
prompt gets a new first line, so the prompt cache cannot reuse it), `--restart-every N` for engine-first prompts (the
bench case: an engine's first loan of cache slots), a post-mortem in the same process on a degenerate answer.
`tuning/nan_inject.py` - a NaN injected into the residual at a chosen prompt layer.  `tuning/vram_grab.py` - VRAM
pressure from a second process (written, not run).  Probes in the source, off unless set (uncommitted; build
`tuning/bin/strata-dbg5.exe`): STRATA_DBG_SLOTS (every resident expert's CUDA0 slot against its RAM blob, and the
device residency table against the host's, at each request start and after each prompt, with the slot's last writer),
STRATA_DBG_NAN2 (per layer: non-finite counts of R, the mixer output, GU, H, Dm, bo, counted on the device - no sync),
STRATA_DBG_PHASES (join / lend / batched / windows / checkpoints / refill ms per request), STRATA_DBG_INJECT_NAN=<file>.

Runs (iq3.json args, STRATA_ADAPT_BG=1, 4 KB-page arena tonight, 3.2-3.4K slots): 67 8K prompts in 15 engine starts
(nan-bg-a..e), 0 degenerate; 520 slot/table checks, 0 differences; NaN counters all finite; prompt 743-815 t/s, no
slow outlier.

Injection (one engine, a short decode before each 8K prompt):

| NaN at layer | prompt t/s | output | decode hit rate |
|---|---:|---|---:|
| none | 917 | normal | 0.603 |
| 0 | 979 | 256 x token 0 | 0.946 |
| 12 | 977 | 256 x token 0 | 0.944 |
| 24 | 985 | 256 x token 0 | 0.921 |
| 36 | 993 | 256 x token 0 | 0.836 |
| 47 | 993 | normal | 0.711 |

So the bad run (hit rate 94.6%, drafts 126 of 126) matches a NaN born in the first ~12 layers - and a NaN does not
slow a prompt, so whatever caused it also slowed that prompt (673 t/s; the only one below 967 in all 40 p13/p14
stress starts, with the same startup VRAM, slots and arena).  No build overlapped it (p13's ended 02:33, p14's started
~02:47; the prompt ran ~02:43), the System/Application/Defender/TaskScheduler logs have nothing at 02:41-02:44.  Code
read without finding a fault: the adaptive thread's publication and joins, the prompt path's lend/refill, the
streamed ring and stager (copy groups, pinned buffers), the batched gathers, the region sharing, the routing readback,
the PLE Direct reader and DirectFile (one latent bug: `DirectFile::wait` can write past `out[max]` when a wake packet
drains immediate completions - needs failed reads, which would have errored).  Candidates left: WDDM moving
allocations to system RAM under VRAM pressure from another program mid-prompt (fits the slowdown; corruption would
need a driver/runtime bug) - test with vram_grab.py during nan_stress.py; a transient PCIe/VRAM fault; a race exposed
by that timing.

Decision (owner): treat it as a one-off.  STRATA_ADAPT_BG=1 is now in the "env" of strata-iq3_xxs.json and
strata-iq2_xs.json (backups .before-adaptbg); engine\strata.exe is still strata-p14.exe.  If "!!!" shows up again:
note the time, keep strata-*.log, and start with vram_grab.py + nan_stress.py, then the probes.

### The "!!!" found live (2026-10-03 night): a late session reset (fixed, p15) and a layer-5 NaN (open)

Live: the first request after a server restart (a 464-token tool loop, tuning/style/tool_turns.py) came out NaN, and
every request reusing its prefix after it.  Reproduced with `tuning/nan_serverloop.py` (restart the server - the old
one killed, as a restart does - and send that same first request): p14 1 in 14 starts; with the debug build and
STRATA_DBG_NAN=1 (a sync and a readback after every prompt layer) about 1 in 3 (10 of 29 over the baselines below).
Probes (build tuning/bin/strata-dbg5..8): the batched prompt path ended with the GDN state and the PLE history at
exactly 0 (and the run twice as fast: zero activations route every token to the same experts), or ended fine and the
first verify window after it went NaN; afterwards even a fresh prompt was "!!!" in that process.

A/B, 6-8 starts each, STRATA_DBG_NAN=1 (failing starts / starts):

| variant | NaN |
|---|---:|
| baseline (dbg5/6/7 builds) | 3/5, 3/6, 1/6, 1/6 |
| yesterday's prompt-path changes off (dense MMQ, widen, RDNA attention, small-N, copy groups) | 3/6 |
| --vram-reserve-mib 2048 (1.6 GiB free) | 1/6 |
| STRATA_DBG_SLOTS=1 (device sync + all slots read at the request's start) | 0/6 (all slots intact) |
| STRATA_DBG_SYNC_START (only cudaDeviceSynchronize there) | 0/6 |
| STRATA_DBG_DELAY_START (only 600 ms there) | failed |
| STRATA_DBG_SYNC_READY (device sync before READY) | 2/6 |
| STRATA_DBG_ZERO_SYNC (device sync right after the per-request session_zero) | 0/8 |
| AMD_SERIALIZE_KERNEL=3 + AMD_SERIALIZE_COPY=3 | the prompt clean, then the engine died with 0xC0000374 (heap corruption) |
| --ple-delay-us 3000 (PLE race check; not with DBG_NAN) | clean, 1 start |

Fix 1 (p15, installed as engine\strata.exe; p14 kept as strata.exe.p14): the serve path's session_zero (per new
conversation) is followed by cudaDeviceSynchronize instead of cudaStreamSynchronize(main_stream) - on this HIP the
stream sync returned while the ~1.6 GB of reset memsets (131K context) still ran, and the late zeroing wiped what the
prompt path and the windows (their own streams) had written.  p15 under the same loop: 1 of 12 - the zeros are gone
(the failing start's prompt ends with PLE 13.9, GDN 2.29 as normal), but layer 5's MoE still goes non-finite: "layer 5
(mmq 1, types 17/42, 419 experts): non-finite GU 6245 H 0 Dm 16197 bo 15871 of T 459", the first window NaN.

Open (the second cause): layers 0-4 compute as in a good start, and layer 5's gate/up product is the first non-finite
value.  Layer 5 is the first layer with gate/up type 17 and down type 42; layers 0-9 are the arena's pinned slices
(cudaHostRegister stops at 8 GiB), so their non-resident experts are DMA'd straight from registered RAM into the
8-slot staging ring (the !stream_all path, chunks < 1024 tokens) - a copy event that fires before the copy has landed
would let the kernel read the slot's previous blob (another layer, another layout: garbage scales).  Next: check each
staged blob against its source right after its copy event (a checksum kernel on the compute stream), STRATA_DBG_NAN on.
Also unexplained: the heap corruption under AMD_SERIALIZE.  Probes left in the source (off unless set): STRATA_DBG_SLOTS,
_NAN2, _PHASES, _INJECT_NAN, _POISON (0xff in every new device/pinned buffer), _SYNC_START, _DELAY_START, _SYNC_READY.

Later the same night (p15 and after):
- STRATA_DBG_STAGECHK (dbg9: every expert blob the prompt path's MoE reads - VRAM slot or staging ring - checksummed on
  the compute stream right before use, against its RAM copy): 15 first-request starts and 7 x 8K prompts in two chunks
  (~140K blobs) with 0 mismatches.  The staging ring and the cache slots are clean.
- tuning/micro/sync_test.cu (plain HIP: memset / slow kernel / pinned H2D copy on one non-blocking stream, then a stream
  sync or an event, then a check on another stream; fresh and never-touched buffers up to 1 GiB): 150 rounds, no early
  completion.  But it ran after the trigger below had gone, so it says nothing about that state.
- The trigger is outside the engine.  Until ~01:05 other programs held 3.2-3.5 GiB of VRAM: no sync after the reset
  10 of 30 starts NaN, with a device sync after it 0 of 20 (same conditions - the fix is real), p15 1 of 12 (layer 5).
  From ~01:25 they held ~3.05 GiB and nothing failed in 35 starts, the build WITHOUT the fix included (nan-wddm 0/12).
  Failures by cache size: 2346-2356 slots 8 of 12 bad, 2595-2622 slots 0 of 31.
- tuning/gpu_mem_sampler.ps1 (the engine's VRAM and RAM-side GPU memory per adapter) and nan_serverloop.py --grab-mib
  (a second process holds VRAM during the first request): with 1 GiB grabbed the engine's memory did not move (local
  ~12.0 GB, non-local ~8.1 GB throughout) and 0 of 8 failed - Windows made room elsewhere.
- Open: what was running until ~01:05 (another GPU client: rendering, video, a launcher?).  Recreate it, then rerun
  nan_serverloop.py (with and without the fix) and sync_test.exe under it.
- The owner was away from the PC from ~00:50, and the displays did NOT sleep (something held them on) - so not the
  screen.  Per-process VRAM the next morning: RSI Launcher 376 MiB, steamwebhelper 337, msedge 180, Discord 146, dwm
  670 (at 23:00: 229, 200, 162, 111, 769; the Hermes app's 104 then, gone now) - background apps' GPU use moves on
  its own.  Next: reproduce with the usual apps open and active, and log every process's GPU memory alongside.

### The "!!!" bug found (2026-10-03, day): a host buffer inside a registered slice's device alias (fixed, p16)

**Cause.**  `cudaHostRegister(..., Mapped)` gives each pinned arena slice a *device alias* - the address the GPU reads
it at.  On Windows HIP that alias is not (fully) reserved in the process's own address space, so the OS can hand the
same numbers to an ordinary allocation.  `hipMemcpy` decides what each pointer is by looking it up among the runtime's
memory objects, whatever `kind` says - so a copy "device -> that host buffer" is done as a copy INTO THE ARENA SLICE, and
the host buffer gets nothing.  Caught in the act (strata-dbg15, STRATA_DBG_ARENA=1):

    arena slice 2: host 0000021A60E00000, device alias 00000208B0B40000
    snapshot copy: dst 00000208B9098060 src 0000007FCED34200, 117669888 B
    arena after the checkpoint: 64 experts changed (from layer 2 expert 74 to layer 2 expert 137)

dst - 112 MiB of a conversation checkpoint, a std::vector - lies 139.8 MB into slice 2's alias: expert 74, and 112 MiB
is 63 experts.  Every later request's snapshot overwrote the next ~64 experts.  The CPU pool then computed layer-0/2
experts from that garbage: NaN rows in y_miss, the first window NaN, "!!!" for the rest of the process.  The same
collision explains the other faces of the bug: read-backs into a colliding vector stay zero ("zero state", checksums
00000000, `verify_slot: slot 0 differs`), a restored checkpoint is empty, and the prompt path DMAs damaged experts
(the layer-5 NaN).  Which buffer and which slice: wherever the heap happened to be - hence "about one start in N".

**Why it looked environmental.**  A big expert cache keeps the damaged experts' VRAM copies in use, so nothing reads the
RAM copy.  Other programs' VRAM only shrinks the auto-sized cache: `--vram-reserve-mib 3200` alone (no second process,
2.7 GiB of VRAM free, ~1650 slots) gave 2/12, 4/12, 5/12, 3/13 NaN first requests.

**Standalone proof.**  `tuning/micro/alias_test.cu` (plain HIP, no engine): registers 1 GiB, commits a host buffer in
the free part of the alias range, copies device -> host:

    alias + 0 MiB .. + 941 MiB: reserved      alias + 941 MiB .. + 1024 MiB: FREE
    the host buffer received 0 of 4194304 bytes; the REGISTERED memory changed in 4194304 bytes, from offset 941 MiB

(a 64 MiB registration's alias is reserved whole; in another run the 1 GiB alias sat at 0x300000000, all of it free).
Worth reporting to ROCm (TheRock 7.13.0a20260421, Windows).

**Fix (p16, installed).**  The alias guard in `hip_compat/cuda_runtime.h` + `platform::guard_address_range`: every
device range the engine learns of (hipMalloc, the alias of hipHostMalloc / hipHostRegister memory) has its free parts
reserved in the host address space (MEM_RESERVE, nothing committed) while the object lives.  A range that already
holds committed host memory: a mapped host buffer is allocated again, a registration is refused (that slice stays
unpinned).  hipMalloc ranges turned out to be reserved by the runtime already; the ten arena slices' aliases were not.
`STRATA_ALIAS_GUARD=0` turns it off.  Same binary (strata-p16.exe), alternating, small cache, arena probe on: guard
off 2 starts of 12 with arena experts changed (both NaN), guard on 0 of 12, no registration refused (the rate without
it depends on the heap's layout: 5/10 in dbg14).  Installed as engine/strata.exe (p15 kept as strata.exe.p15); with
no debug options: the daily config 0/6, the small-cache config 0/6, first prompt 4.6 s and decode 60-66 t/s as before.

**Ruled out on the way** (each tested): stream scratch growth (no kernel needs more than 424 B); Windows losing VRAM
contents under pressure (`tuning/vram_keep.py`, 9 GiB kept beside a 4-9 GiB grab: intact); the commit limit
(`tuning/commit_hog.py`, `sys_mem_sampler.ps1`: commit sits at the limit with and without failures - the page file
grows on demand); `STRATA_PIN_LIMIT_GIB=4` as a fix (fewer slices, same hole); a queued refill racing the first window
(`STRATA_REFILL_BLOCKING=1` still failed); stream syncs returning early (`STRATA_SYNC_CHECK=1`, a stamp kernel checked
through mapped memory: 0 early returns in 12 starts, 2 of them NaN); the decode-path features of 10-01/02 (all off:
still failed); a damaged graph capture (node/edge counts equal).  The night's "late session reset" (p15's device
sync) is therefore doubtful as a cause - the A/B that supported it ran on a layout that happened to collide; the
change is harmless and stays.

Tools added: `nan_serverloop.py` (several configs alternating, `--hog-leave-gib`, failed starts counted),
`vram_keep.py`, `commit_hog.py`, `sys_mem_sampler.ps1`, `micro/alias_test.cu`; probes (off unless set):
STRATA_DBG_ARENA (the arena's RAM against its state at READY, per request phase; the snapshot copies' pointers; the
guard's reservations), STRATA_DBG_NAN now also names the window layer and the CPU-pool rows where a NaN is born,
STRATA_SYNC_CHECK=1|2.

## Hermes: a runaway image call held the server (2026-10-03 15:30)

A Hermes pelican run stood still at "waiting on qwen3.8 ... 60s without stream output" after "2 tool calls failed".
Hermes had sent two `vision_analyze` helper calls at once (non-streamed, no max tokens, thinking at the default level);
the server runs one request at a time.  The first one's thinking went into a doubt loop ("but the question might be
from a benchmark ...") and ran for 15 minutes / 63K tokens; Hermes gave up on both after its 120 s timeout, but a
non-streamed request writes nothing until its end, so the server never saw the clients leave - the second abandoned
call and the agent's next step waited behind it.

- serve/server.py: a non-streamed request's client is looked at once a second (`_client_gone`: readable socket with
  nothing to read) and the request is stopped (`_while_connected`); a request waiting for its turn now sends pings, so
  one whose client has left never starts and a streamed one gets keep-alives while it waits.  Live: a client that
  gave up after 4.0 s, the server idle at 4.3 s.
- serve/server.py: a request's figures and the status are finished before it gives up its turn.  They were written
  after, onto the status of the request that had just started: that one lost its "tail" (KeyError in `_note`, the
  request failed) and the log said "done: 4478 tokens in 0 s".  Any two overlapping requests could hit it.
- Hermes config.yaml `auxiliary.vision`: timeout 300, extra_body reasoning_effort low + max_tokens 4000 (backup
  config.yaml.before-vision-aux).  An image call: 4-5.7K tokens / 66-98 s before, 1.0-1.5K tokens / 17-28 s now.
- Tests: serve/test_server.py RequestsAtOnce (65 of 65 pass).
Not fixed: the doubt loop itself (the main agent's own steps have no upper bound; they are streamed and can be stopped).
