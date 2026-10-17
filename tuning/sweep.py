"""tuning/sweep.py - run a list of bench.py configurations one after another.

    python tuning/sweep.py tuning/sweeps/phase3a.json

The sweep file is a JSON list of {"name": ..., "set": ["--flag=v", ...], "del": [...], "env": {"K": "V"},
"exe": ..., "suite": "quick"|"full", "runs": N, "note": ...}; every entry is compared against "compare"
(default tuning/results/baseline).  A finished entry (its summary.json exists) is skipped, so an interrupted
sweep can simply be started again.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def main() -> int:
    entries = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))   # PowerShell writes a BOM
    for e in entries:
        out = ROOT / "tuning" / "results" / e["name"]
        if (out / "summary.json").exists():
            print(f"[sweep] {e['name']}: done already, skipped", flush=True)
            continue
        cmd = [PY, "-u", str(ROOT / "tuning" / "bench.py"), "--name", e["name"],
               "--suite", e.get("suite", "quick"), "--runs", str(e.get("runs", 3)),
               "--compare", e.get("compare", str(ROOT / "tuning" / "results" / "baseline"))]
        for s in e.get("set", []):
            cmd += ["--set", s]
        for d in e.get("del", []):
            cmd += ["--del", d]
        for k, v in e.get("env", {}).items():
            cmd += ["--env", f"{k}={v}"]
        if e.get("exe"):
            cmd += ["--exe", e["exe"]]
        if e.get("config"):
            cmd += ["--config", e["config"]]
        if "retries" in e:
            cmd += ["--retries", str(e["retries"])]
        if e.get("note"):
            cmd += ["--note", e["note"]]
        print(f"[sweep] {e['name']}: {' '.join(cmd[3:])}", flush=True)
        r = subprocess.run(cmd, cwd=ROOT)
        print(f"[sweep] {e['name']}: exit {r.returncode}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
