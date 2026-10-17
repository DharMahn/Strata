"""tuning/bench.py - the benchmark harness for the RDNA2 (RX 6950 XT) tuning work (see tuning/LOG.md).

Each RUN is one fresh engine start that reads a fixed set of prompts:

    warmup   a short question, 48 tokens out (not scored: graphs, caches and clocks settle)
    short    three ~60-token chat prompts (prose, code, explanation), 512 tokens out each
    long     (suite "full") one ~8K-token prompt (a C++ file + a question), 256 tokens out
             (suite "prompt": warmup + long only, for prompt-path sweeps)

The engine is driven directly over `strata --serve`'s stdin protocol - the same `GEN <max_new> <ids>` /
`T <id>` / `DONE ...` lines serve/server.py exchanges with it - so the output token ids and the engine's own
prompt/decode clocks are recorded exactly.  Greedy decoding, thinking off (the chat template's
enable_thinking=False), as the server does for a client that sets no sampling.

    python tuning/bench.py --name baseline --suite full --runs 3
    python tuning/bench.py --name w12 --set --pool-workers=12 --compare tuning/results/baseline
    python tuning/bench.py --name pf55 --set --pcie-frac=0.55 --env STRATA_FOO=1 --exe tuning/bin/other-build.exe

Results go to tuning/results/<name>/: run<i>.log (the engine's stderr), run<i>.json, summary.json, summary.md.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import queue
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNING = ROOT / "tuning"
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

PROMPTS = [
    # name, kind, max_new, messages builder (text or file)
    ("warmup", "warmup", 48, "Hello! In two sentences, what is a GPU?"),
    ("s_essay", "short", 512,
     "Write a detailed essay of roughly 1,200 words about the history of the printing press and how it changed "
     "European society. Use several paragraphs and do not use headings."),
    ("s_code", "short", 512,
     "Write a complete, well-documented Python module implementing a thread-safe LRU cache with per-entry TTL "
     "expiry. Include type hints, docstrings, and a thorough set of pytest unit tests at the end."),
    ("s_explain", "short", 512,
     "Explain in depth how a modern out-of-order CPU core executes instructions: fetch, decode, register renaming, "
     "scheduling, execution and retirement, plus branch prediction and the cache hierarchy. Be thorough and "
     "concrete."),
    ("l_code8k", "long", 256, "@l_code8k"),
]
LONG_TARGET_TOKENS = 8000
LONG_SOURCE = ROOT / "src" / "core" / "session.cpp"

# a run whose log says one of these did not measure the configuration (it measured a broken start)
INVALID_PATTERNS = [
    r"SetProcessWorkingSetSizeEx\(\d+ MiB\) failed",
    r"VirtualLock stopped",
    r"unspecified launch failure",
    r"issue #29",
    r"driver timeout",
    r"hipError",
    r"out of memory",
]
NOTE_PATTERNS = {
    "workers": r"(\d+) expert-pool workers",
    "cache_slots": r"expert cache (\d+) slots, ([\d.]+) GiB of VRAM",
    "vram_free_mib": r"(\d+) MiB of VRAM free with everything loaded",
    "arena": r"expert arena: (.*)",
    "prompt_chunk": r"prompt chunk auto: (\d+) tokens",
    "loaded": r"loaded ([\d.]+) GiB at ([\d.]+) GiB/s",
}


# ------------------------------------------------------------------------------------------------ prompts
def load_tokenizer(cfg: dict):
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = (tpath / "merges.txt").read_text(encoding="utf-8").split("\n")
    types = json.loads((tpath / "token_type.json").read_text())
    tok = ST.Tokenizer(tokens, merges, types)
    tpl = tpath / "chat_template.jinja"
    return tok, ChatTemplate(tpl if tpl.exists() else ROOT / "serve" / "chat_template.jinja")


def render_ids(tok, template, user_text: str) -> list[int]:
    prompt = template.render([{"role": "user", "content": user_text}], tools=None, enable_thinking=False)
    return tok.encode(prompt, parse_special=True)


def long_prompt_text(tok, template) -> str:
    """A C++ source file cut so that the whole chat prompt is about LONG_TARGET_TOKENS tokens."""
    src = LONG_SOURCE.read_text(encoding="utf-8")
    head = "Here is a C++ source file from an LLM inference engine:\n\n```cpp\n"
    tail = ("\n```\n\nExplain what this file does, section by section, and point out anything that looks like a "
            "bug or a performance problem.")
    lo, hi = 1000, len(src)
    while hi - lo > 200:                                  # bisect on characters
        mid = (lo + hi) // 2
        n = len(render_ids(tok, template, head + src[:mid] + tail))
        lo, hi = (mid, hi) if n <= LONG_TARGET_TOKENS else (lo, mid)
    cut = src[:lo]
    cut = cut[:cut.rfind("\n") + 1]                       # whole lines
    return head + cut + tail


def prompt_set(cfg: dict) -> list[dict]:
    """The prompts as frozen token ids: tuning/prompts/<name>.ids is made once and then always used as it is."""
    pdir = TUNING / "prompts"
    pdir.mkdir(exist_ok=True)
    tok = template = None
    out = []
    for name, kind, max_new, text in PROMPTS:
        f = pdir / f"{name}.ids"
        if not f.exists():
            if tok is None:
                tok, template = load_tokenizer(cfg)
            if text == "@l_code8k":
                text = long_prompt_text(tok, template)
                (pdir / f"{name}.txt").write_text(text, encoding="utf-8")
            ids = render_ids(tok, template, text)
            f.write_text(",".join(map(str, ids)), encoding="utf-8")
        ids = [int(x) for x in f.read_text(encoding="utf-8").split(",")]
        out.append({"name": name, "kind": kind, "max_new": max_new, "ids": ids})
    return out


# ------------------------------------------------------------------------------------------------ config
def split_args(args: list[str]) -> list[list[str]]:
    """["--a", "1", "--b", "--c", "4", "44"] -> [["--a","1"], ["--b"], ["--c","4","44"]]."""
    out: list[list[str]] = []
    for a in args:
        if a.startswith("--") or not out:
            out.append([a])
        else:
            out[-1].append(a)
    return out


def apply_overrides(args: list[str], sets: list[str], dels: list[str]) -> list[str]:
    groups = split_args(args)
    for d in dels:
        groups = [g for g in groups if g[0] != d]
    for s in sets:                                        # --flag=value[,value2] or --flag (a switch)
        flag, _, val = s.partition("=")
        vals = val.split(",") if val else []
        for g in groups:
            if g[0] == flag:
                g[1:] = vals
                break
        else:
            groups.append([flag, *vals])
    return [x for g in groups for x in g]


def free_ram_gib() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()[:16]


def git_rev() -> str:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True,
                               text=True).stdout.strip()
        branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"{branch}@{rev}{'+dirty' if dirty else ''}"
    except OSError:
        return "?"


# ------------------------------------------------------------------------------------------------ one run
def one_run(exe: str, args: list[str], cfg: dict, env: dict, prompts: list[dict], log_path: Path,
            ready_timeout=900.0) -> dict:
    from serve.server import child_env
    from serve.winjob import contain
    full_env = child_env(cfg)
    full_env.update(env)
    ram0 = free_ram_gib()
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([exe, "--serve", *args], cwd=cfg.get("cwd"), stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8", bufsize=1,
                                env=full_env)
        contain(proc)
        lines: queue.Queue = queue.Queue()

        def pump():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)
        threading.Thread(target=pump, daemon=True).start()

        def next_line(deadline):
            left = deadline - time.time()
            if left <= 0:
                raise TimeoutError
            line = lines.get(timeout=left)
            if line is None:
                raise RuntimeError(f"engine exited (code {proc.poll()})")
            return line

        res = {"free_ram_gib_before": round(ram0, 1), "requests": [], "error": None}
        try:
            dl = time.time() + ready_timeout
            while True:
                line = next_line(dl)
                if line.startswith("READY"):
                    break
            res["startup_s"] = round(time.time() - t0, 1)
            for p in prompts:
                r0 = time.time()
                proc.stdin.write(f"GEN {p['max_new']} {','.join(map(str, p['ids']))}\n")
                proc.stdin.flush()
                toks = []
                dl = time.time() + 3600
                while True:
                    line = next_line(dl)
                    if line.startswith("T "):
                        toks.append(int(line[2:]))
                    elif line.startswith("DONE"):
                        f = line.split()
                        req = {"name": p["name"], "kind": p["kind"], "generated": int(f[1]), "prompt_tokens": int(f[2]),
                               "prompt_ms": float(f[3]), "decode_ms": float(f[4]), "finish": f[5],
                               "drafts_accepted": int(f[6]), "drafts_offered": int(f[7]), "reused": int(f[8]),
                               "hits": int(f[9]) if len(f) > 9 else None, "lookups": int(f[10]) if len(f) > 10 else None,
                               "wall_s": round(time.time() - r0, 2), "tokens": toks}
                        fresh = req["prompt_tokens"] - req["reused"]
                        req["prompt_tps"] = round(1000.0 * fresh / req["prompt_ms"], 2) if req["prompt_ms"] > 0 else None
                        req["decode_tps"] = round(1000.0 * req["generated"] / req["decode_ms"], 2) if req["decode_ms"] > 0 else None
                        res["requests"].append(req)
                        break
                    elif line.startswith("ERR"):
                        raise RuntimeError(line.strip())
        except (TimeoutError, RuntimeError, OSError) as e:
            res["error"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
                proc.wait(timeout=60)
    text = log_path.read_text(encoding="utf-8", errors="replace")
    res["invalid"] = [pat for pat in INVALID_PATTERNS if re.search(pat, text)]
    if re.search(r"expert arena: .*large pages \(\d+ B\)", text):
        # a large-page arena cannot be paged out at all: the working-set lock it then skips (and may fail) is moot
        res["invalid"] = [p for p in res["invalid"] if not p.startswith("SetProcessWorkingSetSizeEx")]
    notes = {}
    for k, pat in NOTE_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            notes[k] = m.group(1) if m.lastindex == 1 else list(m.groups())
    res["notes"] = notes
    res["decode_timing"] = re.findall(r"strata decode timing: (.*)", text)
    res["decode_gpu_stages"] = re.findall(r"strata decode GPU stages \(ms/window\):(.*)", text)
    res["prefill_timing"] = re.findall(r"strata prefill timing: (.*)", text)
    res["warnings"] = sorted(set(re.findall(r"(?im)^.*(?:warning|failed|error).*$", text)))[:20]
    return res


# ------------------------------------------------------------------------------------------------ summary
def first_divergence(a: list[int], b: list[int]) -> int | None:
    """None when identical; else the first index where they differ (or the shorter length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def run_metrics(run: dict) -> dict:
    rq = {r["name"]: r for r in run["requests"]}
    short = [r for r in run["requests"] if r["kind"] == "short"]
    m = {}
    if short:
        gen = sum(r["generated"] for r in short)
        ms = sum(r["decode_ms"] for r in short)
        m["short_decode_tps"] = round(1000.0 * gen / ms, 2)
        off = sum(r["drafts_offered"] for r in short)
        m["short_accept"] = round(sum(r["drafts_accepted"] for r in short) / off, 3) if off else None
        look = sum(r["lookups"] or 0 for r in short)
        m["short_hit_rate"] = round(sum(r["hits"] or 0 for r in short) / look, 3) if look else None
    if "l_code8k" in rq:
        r = rq["l_code8k"]
        m["long_decode_tps"] = r["decode_tps"]
        m["long_prefill_tps"] = r["prompt_tps"]
        m["long_accept"] = round(r["drafts_accepted"] / r["drafts_offered"], 3) if r["drafts_offered"] else None
        m["long_hit_rate"] = round(r["hits"] / r["lookups"], 3) if r["lookups"] else None
    for r in run["requests"]:
        m[f"{r['name']}_decode_tps"] = r["decode_tps"]
    return m


def stat(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    med = statistics.median(vals)
    return {"median": round(med, 2), "min": round(min(vals), 2), "max": round(max(vals), 2),
            "spread_pct": round(100.0 * (max(vals) - min(vals)) / med, 1) if med else None, "n": len(vals)}


def summarize(name: str, meta: dict, runs: list[dict], compare: Path | None) -> dict:
    valid = [r for r in runs if not r["error"] and not r["invalid"]]
    keys = sorted({k for r in valid for k in run_metrics(r)})
    summary = {"name": name, "meta": meta, "runs_total": len(runs), "runs_valid": len(valid),
               "invalid_runs": [{"i": i, "error": r["error"], "invalid": r["invalid"]} for i, r in enumerate(runs)
                                if r["error"] or r["invalid"]],
               "metrics": {k: stat([run_metrics(r).get(k) for r in valid]) for k in keys}}
    # determinism inside this config
    det = {}
    if valid:
        for p in valid[0]["requests"]:
            seqs = [next((q["tokens"] for q in r["requests"] if q["name"] == p["name"]), []) for r in valid]
            divs = [first_divergence(seqs[0], s) for s in seqs[1:]]
            det[p["name"]] = {"identical": all(d is None for d in divs), "first_div": divs}
    summary["run_to_run"] = det
    # against a reference configuration's run 1 (the correctness gate)
    if compare is not None and (compare / "summary.json").exists():
        ref_runs = [json.loads(p.read_text()) for p in sorted(compare.glob("run*.json"))]
        ref_runs = [r for r in ref_runs if not r["error"] and not r["invalid"]]
        gate = {}
        for p in (valid[0]["requests"] if valid else []):
            refs = [next((q["tokens"] for q in r["requests"] if q["name"] == p["name"]), None) for r in ref_runs]
            refs = [x for x in refs if x is not None]
            if not refs:
                continue
            mine = [next((q["tokens"] for q in r["requests"] if q["name"] == p["name"]), []) for r in valid]
            divs = [first_divergence(refs[0], s) for s in mine]
            gate[p["name"]] = {"identical_to_ref_run1": all(d is None for d in divs),
                               "matches_some_ref_run": all(any(first_divergence(rf, s) is None for rf in refs) for s in mine),
                               "first_div_vs_ref_run1": divs, "ref_len": len(refs[0])}
        summary["vs_reference"] = {"reference": str(compare), "prompts": gate}
    return summary


def write_md(out: Path, s: dict) -> None:
    L = [f"# {s['name']}", "", f"- source: `{s['meta']['git']}`, exe `{s['meta']['exe']}` ({s['meta']['exe_sha']})",
         f"- args: `{' '.join(s['meta']['args_diff']) or '(baseline config)'}`  env: `{s['meta']['env'] or '-'}`",
         f"- runs: {s['runs_valid']} valid of {s['runs_total']}", ""]
    if s["invalid_runs"]:
        L += [f"- INVALID run {x['i']}: {x['error'] or ''} {', '.join(x['invalid'])}" for x in s["invalid_runs"]] + [""]
    L += ["| metric | median | min | max | spread % |", "|---|---:|---:|---:|---:|"]
    for k, v in s["metrics"].items():
        if v:
            L.append(f"| {k} | {v['median']} | {v['min']} | {v['max']} | {v['spread_pct']} |")
    L += ["", "Run-to-run identical output: " + ", ".join(f"{k} {'yes' if v['identical'] else 'NO ' + str(v['first_div'])}"
                                                      for k, v in s["run_to_run"].items())]
    if "vs_reference" in s:
        L += ["", f"Against `{s['vs_reference']['reference']}`:", ""]
        for k, v in s["vs_reference"]["prompts"].items():
            L.append(f"- {k}: identical to ref run 1: {v['identical_to_ref_run1']}, matches some ref run: "
                     f"{v['matches_some_ref_run']}, first divergence {v['first_div_vs_ref_run1']} (of {v['ref_len']})")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--config", default=None,
                    help="the run config setup wrote (default: the strata-*.json in the Strata folder; with several, "
                         "the newest)")
    ap.add_argument("--exe", default=str(ROOT / "engine" / ("strata.exe" if os.name == "nt" else "strata")),
                    help="the engine to measure (default: engine/strata[.exe]; copy a build elsewhere to compare it)")
    ap.add_argument("--set", action="append", default=[], metavar="--FLAG=V[,V2]",
                    help="replace or add an engine flag (a bare --flag adds a switch)")
    ap.add_argument("--del", dest="dels", action="append", default=[], metavar="--FLAG", help="remove an engine flag")
    ap.add_argument("--env", action="append", default=[], metavar="K=V")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2, help="extra starts allowed when a run is invalid")
    ap.add_argument("--suite", choices=["quick", "full", "prompt"], default="quick")
    ap.add_argument("--compare", default=None, help="a results dir to check the output tokens against")
    ap.add_argument("--note", default="")
    # "--set --flag=v": argparse cannot take a value that starts with "--", so pair them by hand first
    argv, fixed = sys.argv[1:], []
    i = 0
    while i < len(argv):
        if argv[i] in ("--set", "--del") and i + 1 < len(argv):
            fixed.append(f"{argv[i]}={argv[i + 1]}")
            i += 2
        else:
            fixed.append(argv[i])
            i += 1
    a = ap.parse_args(fixed)

    exe = Path(a.exe)
    a.exe = str(exe if exe.is_absolute() else (ROOT / exe).resolve())   # the engine starts in the config's cwd
    if a.config is None:
        found = sorted(ROOT.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise SystemExit("[bench] no strata-*.json in the Strata folder: run START-HERE.bat / setup first, or pass --config")
        a.config = str(found[0])
    print(f"[bench] config {a.config}", flush=True)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    args = apply_overrides(cfg["args"], a.set, a.dels)
    env = dict(kv.split("=", 1) for kv in a.env)
    env.setdefault("STRATA_DECODE_TIMING", "1")
    prompts = prompt_set(cfg)
    if a.suite == "quick":
        prompts = [p for p in prompts if p["kind"] != "long"]
    elif a.suite == "prompt":   # the prompt path's sweeps: the warmup and the 8K prompt only
        prompts = [p for p in prompts if p["kind"] in ("warmup", "long")]
    out = TUNING / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)
    base = cfg["args"]
    args_diff = [f"set {s}" for s in a.set] + [f"del {d}" for d in a.dels]
    meta = {"git": git_rev(), "exe": os.path.relpath(a.exe, ROOT), "exe_sha": file_hash(Path(a.exe)),
            "args": args, "args_diff": args_diff, "env": env, "suite": a.suite, "note": a.note,
            "prompts": {p["name"]: len(p["ids"]) for p in prompts}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    print(f"[bench] {a.name}: {a.suite} suite, {a.runs} runs, {' '.join(args_diff) or 'baseline args'}, env {env}",
          flush=True)
    runs, attempts = [], 0
    while len([r for r in runs if not r["error"] and not r["invalid"]]) < a.runs and attempts < a.runs + a.retries:
        i = attempts
        attempts += 1
        r = one_run(a.exe, args, cfg, env, prompts, out / f"run{i}.log")
        (out / f"run{i}.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        runs.append(r)
        m = run_metrics(r)
        tag = "INVALID " + ", ".join(r["invalid"]) if r["invalid"] else (r["error"] or "ok")
        print(f"[bench]   run {i}: {tag}; start {r.get('startup_s')} s, free RAM {r['free_ram_gib_before']} GiB; "
              + ", ".join(f"{k} {v}" for k, v in m.items() if k.startswith(("short_", "long_"))), flush=True)
        time.sleep(5)                                     # the engine's RAM goes back to the system
    s = summarize(a.name, meta, runs, Path(a.compare) if a.compare else None)
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    write_md(out / "summary.md", s)
    print((out / "summary.md").read_text(encoding="utf-8"), flush=True)
    return 0 if s["runs_valid"] >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
