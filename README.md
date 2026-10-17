# Strata for AMD RDNA2 on Windows (RX 6800 / 6900 series, RX 6700 XT)

A fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) (engine 0.1.30) that runs Qwen3.8-Flash-Next on
**Windows with an AMD RDNA2 card**, tuned on an **RX 6950 XT (gfx1030, 16 GB)**. The upstream project documents the
model, the sizes and the web app; this README covers only what the fork changes. The upstream README is kept as
[docs/UPSTREAM_README.md](docs/UPSTREAM_README.md).

> **Status: experimental, one tested machine.** Everything below was measured on a Ryzen 9 9950X, 64 GB DDR5,
> RX 6950 XT 16 GB, Windows 11, ROCm 7.13.0a20260421. The **RX 6700 XT (gfx1031, 12 GB) is wired in but has never been
> run** by the author; see [RX 6700 XT notes](#rx-6700-xt-gfx1031-what-to-expect).

## What you get over upstream

| Area | Change |
|---|---|
| Windows + RDNA2 port | gfx1030 `sdot4` path, integrated-Radeon handling (Ryzen 7000/9000 iGPU), VS 2026 STL workaround, no hipBLASLt for gfx103X. Details: [docs/AMD_HIP.md](docs/AMD_HIP.md) |
| Stability on Windows | Scratch-memory warm-up (fixes the driver-timeout hang), bounded host-wait spins, 8 GiB pin limit, **alias guard** (fixes the intermittent "first answer is `!!!`" NaN bug, see `tuning/LOG.md`) |
| VRAM sizing | The VRAM budget is the WDDM budget minus other programs' usage (the desktop alone holds ~1.5 GiB). Upstream's `hipMemGetInfo` figure over-commits and silently spills into system RAM |
| Speed | `v_perm_b32` for `__byte_perm` (IQ4 kernels 8-21x faster), MMQ prompt path on Windows, RDNA2 prompt-attention kernel, FP32-widened GEMMs, async commit / PLE rows, batched expert gathers |
| Server | Dropped-client detection for non-streamed requests, queue keep-alives, optional request log, optional thinking opening (**see the breaking-change note below**) |
| Setup | `START-HERE.bat --backend hip` installs ROCm into `.venv` (no admin), builds the engine, detects RDNA2 cards from the registry |
| Tuning kit | `tuning/`: benchmark harness, sweep driver, micro-benchmarks, memory samplers, hang watcher |

Measured on the 6950 XT (IQ3_XXS, 64K+ context, int8 KV): about **62-72 tokens/s** decode and **~970 tokens/s**
prompt reading on an 8K prompt. IQ2_XS: 63 tokens/s decode. These are from the author's `tuning/bench.py` runs; expect
different numbers on other cards, and see `tuning/REPORT.md` for method and breakdown.

## Breaking change: the thinking-opening patch (`strata_think_prefix`)

The model (Qwen3.8-Flash-Next at 2-3 bit) thinks in one of two styles, and the first words decide which one it keeps.
At the default "High" reasoning effort it tends to "clipped notes" (`We need answer user: ...`), which gave plainer
drawings and texts. The fork can **write the first words of the thinking itself**:

- after a user message: `Let me think about this. The user`
- after a tool result (agent loops): `Let me`

This changes the prompt the model sees and what it continues from, so it is a behaviour change for every client. It is
**off by default** (the stock behaviour), and it is controlled from the run config:

```jsonc
// strata-<model>.json  (written by setup, next to the repo)
"sampling": {
  "strata_think_prefix": true,        // "Let me think about this. The user"  (or your own words as a string)
  "strata_think_prefix_tool": true,   // "Let me" after a tool result         (or your own words)
  "preserve_thinking": false          // optional: see docs/THINK_PREFIX.md
}
```

or at setup: `START-HERE.bat --think-prefix on` / `--think-prefix off` (`--setup` re-runs setup on an existing install).
Requests can override it per call (`"strata_think_prefix": false`), and the web app's "Start thinking in prose" switch
follows the server's setting until you change it. Full details, the evidence, and the caveats:
**[docs/THINK_PREFIX.md](docs/THINK_PREFIX.md)**.

## Requirements

- Windows 10/11, an AMD **RX 6800 / 6800 XT / 6900 XT / 6950 XT** (Navi 21, gfx1030, 16 GB), or the untested
  **RX 6700 XT** (gfx1031, 12 GB). Other RDNA2 cards (gfx1032 / 1034, 8 GB and under) are not detected by setup.
- RAM: the experts that do not fit in VRAM live in system RAM. 64 GB is what this was run with; see the upstream
  README for how much each model size needs.
- ~6 GB for Visual Studio Build Tools (C++); setup offers to install them (winget). ~2.7 GB for ROCm in `.venv`.
- Disk: the model (70-120 GB depending on the size).

## Install and run

```bat
git clone <this repo>
cd Strata
START-HERE.bat --backend hip
```

First run: installs ROCm (TheRock wheels, ROCm 7.13) into `.venv`, compiles the engine for your card (10-20 minutes),
downloads the model into `..\Strata-data`, writes `strata-<model>.json` and `run-<model>.bat`, and starts the server on
port 8080. Later runs just start it. Add `--think-prefix on` to turn on the thinking opening. To change a finished install later, edit the config's
`sampling` block (restart the server), or run `START-HERE.bat --setup --think-prefix on|off`.

If your PC has a Ryzen with an integrated Radeon, that is expected and handled (its code is compiled in as a
"companion" architecture and never used).

## Tuning for your own machine

The numbers in `tuning/LOG.md` are for a 6950 XT. On other cards (especially the 12 GB 6700 XT) the best
`--vram-reserve-mib`, `--pool-workers` and expert-cache size will differ. The kit in [tuning/](tuning/README.md) measures
that: `python tuning/bench.py --name base --suite full --runs 3`, then `python tuning/sweep.py tuning/sweeps/example.json`.

## Useful environment variables (set in the run config's `"env"` block)

| Variable | Default | Effect |
|---|---|---|
| `STRATA_GPU=<n>` | auto | Pick a GPU by HIP ordinal when more than one is usable |
| `STRATA_PIN_LIMIT_GIB=<n>` | 8 | How much host memory is page-locked on Windows HIP (0 = everything) |
| `STRATA_SCRATCH_WARM_BYTES=<n>` | 512 | Per-stream scratch warm-up, the driver-timeout fix (0 = off; costs ~43 MiB VRAM per stream) |
| `STRATA_ALIAS_GUARD=0` | on | Turn off the host-address-range guard (the `!!!` fix) |
| `STRATA_PREFILL_MMQ=0` | on | Prompt path without llama.cpp's MMQ kernels (FP16 path) |
| `STRATA_ADAPT_BG=1` | off | Background adaptive expert swaps; faster, opt-in |
| `STRATA_ASYNC_COMMIT=0`, `STRATA_PLE_ASYNC=0` | on | Go back to the synchronous variants |
| `STRATA_REQUEST_LOG=<file>` | off | One JSON line per request (what was sent, how the thinking began) |
| `STRATA_DECODE_TIMING=1`, `STRATA_VERIFY_PROFILE=1`, `STRATA_MTP_PROFILE=1`, `STRATA_VRAM_PROBE=1` | off | Profiling output |
| `STRATA_DBG_*`, `STRATA_SYNC_CHECK` | off | NaN-hunt diagnostics (see `tuning/LOG.md`); leave off for normal use |

## RX 6700 XT (gfx1031): what to expect

Setup recognises the card (PCI ids 0x73C0-0x73DF), uses the same ROCm `gfx103X` wheels and compiles for `gfx1031`;
`hip_compat` already treats gfx1031 like gfx1030 (same wave32, LDS size and `sdot4`). That is all that has been done.

- **Untested.** Nobody has built or run this on a 6700 XT. If the build fails or the engine prints
  `device kernel image is invalid`, open an issue with the log.
- **12 GB instead of 16 GB**: fewer experts fit in VRAM, so decode will be slower than the 6950 XT figures and the
  expert-cache hit rate lower. Try `--vram-reserve-mib` 1024-2048 and the smaller IQ2_XS model first.
- **40 CUs instead of 80**: some launch shapes were tuned on 80 CUs (`tuning/LOG.md`, `STRATA_HITS_GY`). Re-run the
  sweep rather than trusting the 6950 XT's optimum.
- rocBLAS kernels for gfx1031 should ship in the same wheel family as gfx1030; if the prompt path is unexpectedly slow,
  `tuning/micro/gemm_bench.cu` (`set ARCH=gfx1031` before `build_gemm.bat`) shows whether the BLAS calls hit tuned kernels.

## Known issues

- The "first answer is `!!!`" NaN bug was traced to Windows HIP handing out host-address ranges that collide with the
  device alias of registered memory, and fixed with the alias guard (0 of 12 starts afterwards, was 2-5 of 12). It is a
  runtime bug, so other driver or ROCm versions may behave differently; if answers start with `!!!` and the log shows
  NaN, report it with `STRATA_DBG_NAN=1` output.
  `tuning/micro/alias_test.cu` is a standalone repro.
- `STRATA_ADAPT_BG` is opt-in because of one NaN in ~48 starts during its development, before the alias guard existed.
- Image input is not supported on AMD (upstream's vision helper is NVIDIA-only; setup's `--vision` does not apply here).
- Linux AMD (gfx1100 / gfx1201) is untouched upstream code. The `xtx/linux` experiments are not part of this fork.

## What is deliberately not in the repository

Benchmark results (`tuning/results/`), engine builds (`tuning/bin/`), build and run logs, request logs, the
hang-watcher output (`hang-watch/`), config backups (`*.before-*`), and the NaN-hunt stress scripts. They are ignored
by `.gitignore`; `tuning/LOG.md` keeps the conclusions.

## Layout

```
README.md              this file
docs/THINK_PREFIX.md   the thinking-opening patch: config, behaviour, evidence
docs/AMD_HIP.md        the HIP backend, including the Windows + RDNA2 section
docs/UPSTREAM_README.md  upstream's README
serve_watched.py       optional wrapper for serve/server.py that logs RAM and /health and dumps stacks on a hang
tuning/                benchmark harness, sweeps, micro-benchmarks, samplers, notes (tuning/README.md)
```

## License and credit

MIT, as upstream (see `LICENSE`). The engine, model packing and web app are Niko1221's work; this fork's changes are the
Windows/RDNA2 port, the tuning, and the server patches listed above.
