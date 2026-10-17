# tuning/ - measuring and tuning Strata on your own RDNA2 card

The numbers in `LOG.md` and `REPORT.md` come from an RX 6950 XT (16 GB, 80 CUs). A different card needs its own
measurements; these scripts are the kit. Results go to `tuning/results/` and engine copies to `tuning/bin/` - both are
git-ignored.

| Path | What it is |
|---|---|
| `bench.py` | The harness. One run = a fresh engine start reading fixed prompts (warmup, three short chat prompts, one ~8K-token prompt), greedy, driven over `strata --serve`'s stdin protocol, so token ids and the engine's own prompt/decode clocks are exact. Compares output tokens against a reference run (the correctness gate). |
| `sweep.py` + `sweeps/example.json` | Runs a list of `bench.py` configurations one after another; finished entries are skipped, so it can be restarted. |
| `prompts/` | The fixed prompts (token ids) the harness uses. |
| `micro/` | Stand-alone HIP micro-benchmarks and their build scripts: `mmvq_bench`, `gemm_bench` (rocBLAS shapes), `attn_bench`, `copy_bench` (PCIe), `skinny_bench`, `probe`, `alias_test` (repro of the Windows HIP host-alias bug). Build with the matching `build_*.bat`; `set ARCH=gfx1031` first for a 6700 XT (default gfx1030). Needs the engine built once (it uses `.venv`'s ROCm and `build-hip\msvc-overlay`). |
| `prefill_kl.py` | Compares prompt-path logits between two builds/settings (KL divergence), for judging numerical changes such as the MMQ path. |
| `get_iq3.py` | Fetches the IQ3_XXS model next to an IQ2_XS install, for an A/B. |
| `sys_mem_sampler.ps1`, `gpu_mem_sampler.ps1` | CSV samplers for commit/RAM and per-process GPU memory (is Windows backing "VRAM" with system RAM?). |
| `style/` | Scripts behind the thinking-opening patch, with the write-up in `style/README.md`. See `../docs/THINK_PREFIX.md`. |
| `pelican/` | The SVG-pelican quality comparison scripts (`run*.py` run against a live server, `build.py` makes a comparison page); the saved samples are not in the repository. |
| `LOG.md`, `REPORT.md`, `decode-review-2026-10-02.md` | What was tried, what worked, what did not (including negative results). Paths to `tuning/results/...` and `tuning/bin/...` in these files refer to local output that is not in the repository. |

## First measurement on a new machine

```bat
rem 1. install and build (once)
START-HERE.bat --backend hip

rem 2. reference run: the run config setup wrote, the engine it built (engine\strata.exe)
python tuning\bench.py --name base --suite full --runs 3

rem 3. try settings against it
python tuning\sweep.py tuning\sweeps\example.json
```

`bench.py` picks the newest `strata-*.json` in the repo as the config (`--config` to choose) and `engine\strata.exe` as
the engine (`--exe` to measure another build). `--set --flag=value` replaces an engine flag, `--env K=V` sets an
environment variable, `--compare tuning/results/base` checks that the output tokens still match the reference.
Stop the server first: the harness starts its own engine and needs the VRAM.

## What to look at first on a 6700 XT / other cards

1. `--vram-reserve-mib` (1024 / 2048): how much VRAM stays free for Windows. 12 GB is tight.
2. `--pool-workers`: CPU expert threads, best at 12 on a 16-core Ryzen; scale with your core count.
3. `STRATA_PIN_LIMIT_GIB`, `STRATA_ADAPT_BG`, `STRATA_PREFILL_MMQ=0` (see the sweep file).
4. The `STRATA_HITS_GY` grid height and the prompt-attention kernel were tuned for 80 CUs; the `micro/` benchmarks run
   them in isolation.

A "valid" run needs the engine log to be free of the failure markers listed in `bench.py`; an invalid run is retried.
