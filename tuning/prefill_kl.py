"""tuning/prefill_kl.py - what a prompt-path change does to the model's predictions (upstream's measure,
bench/results/2026-09-28-prefill-speed: same top-1, KL, perplexity against a reference arm, with a different
prompt chunk of the reference engine as the yardstick - a change everyone already accepts).

With a native pack the engine sends the prompt's last token straight into a verify window and dumps no per-token
logits, so the engine's STRATA_PREFILL_DUMP_LOGITS hook writes them instead: every 16th position of the 8K
benchmark prompt, from the prompt path's final residuals through the token path's own head.  Each arm also writes
the final residual streams at every 64th position (STRATA_PREFILL_DUMP_R): their relative error, for reference.

    python tuning/prefill_kl.py run [arm ...]
    python tuning/prefill_kl.py compare

One-shot engine, IQ3_XXS config, a fixed expert cache (--expert-cache 3000: the same slots in every arm), no
adaptive swaps or suffix drafts, greedy.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tuning" / "results" / "prefill-kl"
EXE = ROOT / "tuning" / "bin" / "strata-dmmq.exe"
IDS = ROOT / "tuning" / "prompts" / "l_code8k.ids"
# name: (env, prompt chunk)
ARMS = {
    "fp16": ({"STRATA_PREFILL_MMQ": "0", "STRATA_PREFILL_DENSE_MMQ": "0"}, "8192"),     # the reference
    "fp16-c4096": ({"STRATA_PREFILL_MMQ": "0", "STRATA_PREFILL_DENSE_MMQ": "0"}, "4096"),  # the yardstick
    "mmq-experts": ({"STRATA_PREFILL_DENSE_MMQ": "0"}, "8192"),                          # installed engine's path
    "mmq-dense": ({}, "8192"),                                                           # + the dense matrices
    # strata-p2.exe: + BF16/FP16 GEMMs widened to FP32, copy events per group (the grouping must change nothing)
    "p2-g1": ({"STRATA_PREFILL_COPY_GROUP": "1"}, "8192", "strata-p2.exe"),
    "p2": ({}, "8192", "strata-p2.exe"),
    # the reference's FP16 path with only the widening: summation order, nothing else
    "fp16-widen": ({"STRATA_PREFILL_MMQ": "0", "STRATA_PREFILL_DENSE_MMQ": "0"}, "8192", "strata-p2.exe"),
    "fp16-rerun": ({"STRATA_PREFILL_MMQ": "0", "STRATA_PREFILL_DENSE_MMQ": "0", "STRATA_GEMM_WIDEN": "0"}, "8192",
                   "strata-p2.exe"),
    # strata-p4.exe: + the RDNA prompt attention kernel, the small-N and hc-down kernels
    "p4": ({}, "8192", "strata-p4.exe"),
    "p4-fp16": ({"STRATA_PREFILL_MMQ": "0", "STRATA_PREFILL_DENSE_MMQ": "0"}, "8192", "strata-p4.exe"),
    "p5": ({}, "8192", "strata-p5.exe"),   # installed 2026-10-02: p4 without the hc-down kernel
    "p6": ({}, "8192", "strata-p6.exe"),   # + the experts' gather batched (must be bit-identical to p5)
    "p5b": ({}, "8192", "strata-p5.exe"),   # reruns: determinism
    "p6b": ({}, "8192", "strata-p6.exe"),
    # p6 raced: a deferred slot release let the copy thread refill a slot before its gather (fixed in p7)
    "p7": ({}, "8192", "strata-p7.exe"),
    "p7b": ({}, "8192", "strata-p7.exe"),
}


def args_for(chunk, n):
    c = json.loads((ROOT / "tuning" / "configs" / "iq3.json").read_text(encoding="utf-8"))
    a = list(c["args"])
    a[a.index("--expert-cache") + 1] = "3000"
    a[a.index("--prefill") + 1] = chunk
    return a + ["--adapt-every", "0", "--suffix-draft", "0", "--greedy", "--tokens-file", str(IDS), "--max-new", "1"], c


def run(names):
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(IDS.read_text().replace(",", " ").split())
    for name in names or ARMS:
        env_extra, chunk, *exe = ARMS[name]
        args, c = args_for(chunk, n)
        out, lg = OUT / f"{name}.r", OUT / f"{name}.logits"
        out.unlink(missing_ok=True)
        env = dict(os.environ, **env_extra, STRATA_PREFILL_DUMP_R=str(out), STRATA_PREFILL_DUMP_LOGITS=str(lg))
        env["PATH"] = os.pathsep.join(c["lib_dirs"]) + os.pathsep + env["PATH"]
        with open(OUT / f"{name}.log", "w") as log:
            r = subprocess.run([str(EXE.parent / exe[0] if exe else EXE), *args], cwd=c["cwd"], stdout=log,
                               stderr=subprocess.STDOUT, env=env)
        print(name, "exit", r.returncode, flush=True)


def load(p, d=10240):
    raw = np.fromfile(p, dtype=np.uint8)
    rec = 8 + 4 * d
    n = raw.size // rec
    raw = raw[:n * rec].reshape(n, rec)
    return raw[:, :8].copy().view(np.int64)[:, 0], raw[:, 8:].copy().view(np.float32)


def lsm(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(axis=-1, keepdims=True)))


def compare():
    ids = [int(x) for x in IDS.read_text().replace(",", " ").split()]
    V = 248320
    lpos, lref = load(OUT / "fp16.logits", V)
    rpos, rref = load(OUT / "fp16.r")
    keep = lpos + 1 < len(ids)
    lpos, lref = lpos[keep], lsm(lref[keep])
    nxt = np.array([ids[p + 1] for p in lpos])
    ppl_ref = np.exp(-lref[np.arange(len(nxt)), nxt].mean())
    print(f"reference fp16 (8192 chunk): {len(lpos)} positions with logits, perplexity {ppl_ref:.3f}")
    print(f"{'arm':<14}{'same top-1':>11}{'mean KL':>9}{'median KL':>11}{'p99 KL':>8}{'ppl':>8}{'resid rel err':>15}")
    for name in ARMS:
        if name == "fp16" or not (OUT / f"{name}.logits").exists():
            continue
        pb, b = load(OUT / f"{name}.logits", V)
        b = lsm(b[keep])
        assert (pb[keep] == lpos).all(), name
        kl = (np.exp(lref) * (lref - b)).sum(axis=1)
        same = (lref.argmax(1) == b.argmax(1)).mean()
        ppl = np.exp(-b[np.arange(len(nxt)), nxt].mean())
        _, r = load(OUT / f"{name}.r")
        rel = np.linalg.norm(r - rref, axis=1) / np.linalg.norm(rref, axis=1)
        print(f"{name:<14}{100 * same:>10.1f}%{kl.mean():>9.4f}{np.median(kl):>11.5f}{np.percentile(kl, 99):>8.3f}"
              f"{ppl:>8.3f}{rel.mean():>15.4f}")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2:]), "compare": compare}[sys.argv[1]]()
