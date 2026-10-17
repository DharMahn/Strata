"""Runs serve/server.py unchanged, with a hang-diagnosis side channel (optional: use it in place of serve/server.py when chasing a hang).

Writes to <the Strata folder>\\hang-watch\\ :
  server-<start>.log   everything the server prints to its console, with a time on every line
  watch-<start>.log    every 60 s: free RAM, commit, /health round-trip; every probe that is slow (>1 s) or fails
  stacks-<start>.txt   every thread's stack, written when /health fails twice in a row, or when this watcher itself
                       has not run for 90 s (the whole Python process frozen, e.g. a regex holding the GIL)
Cost: one tiny GET /health every 10 s and a few lines of text.  Keeps the 12 newest files of each kind.
"""
import ctypes
import faulthandler
import os
import runpy
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "hang-watch"
OUT.mkdir(exist_ok=True)
STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
KEEP = 12
PROBE_EVERY = float(os.environ.get("HANGWATCH_PROBE_S", "10"))      # the env overrides are for testing
FREEZE_AFTER = int(os.environ.get("HANGWATCH_FREEZE_S", "90"))


def now() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def prune(prefix: str):
    files = sorted(OUT.glob(prefix + "-*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[KEEP:]:
        try:
            p.unlink()
        except OSError:
            pass


for pre in ("server", "watch", "stacks"):
    prune(pre)

server_log = open(OUT / f"server-{STAMP}.log", "a", encoding="utf-8", errors="replace", buffering=1)
watch_log = open(OUT / f"watch-{STAMP}.log", "a", encoding="utf-8", errors="replace", buffering=1)
stacks_file = open(OUT / f"stacks-{STAMP}.txt", "a", encoding="utf-8", errors="replace", buffering=1)
faulthandler.enable(file=stacks_file, all_threads=True)      # a native crash of this process leaves its stacks too


class Tee:
    """The console as before, plus a copy with a time at the start of every line."""
    def __init__(self, real):
        self.real = real
        self.at_line_start = True
        self.lock = threading.Lock()

    def write(self, s):
        n = self.real.write(s)
        with self.lock:
            out = []
            for part in s.splitlines(keepends=True):
                if self.at_line_start:
                    out.append(now() + " ")
                out.append(part)
                self.at_line_start = part.endswith("\n")
            server_log.write("".join(out))
        return n

    def flush(self):
        self.real.flush()

    def __getattr__(self, name):
        return getattr(self.real, name)


sys.stdout = Tee(sys.stdout)
sys.stderr = Tee(sys.stderr)


class MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def memory() -> str:
    m = MemStatus()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return f"ram_free={m.ullAvailPhys >> 20} MiB commit_free={m.ullAvailPageFile >> 20} MiB"


def port() -> int:
    a = sys.argv
    for i, x in enumerate(a):
        if x == "--port" and i + 1 < len(a):
            return int(a[i + 1])
    return 8095


def probe() -> tuple[bool, float, str]:
    t0 = time.time()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port()}/health", timeout=5) as r:
            r.read()
        return True, (time.time() - t0) * 1000, ""
    except Exception as e:  # noqa: BLE001 - the point is to record whatever it was
        return False, (time.time() - t0) * 1000, repr(e)[:120]


def dump(why: str):
    stacks_file.write(f"\n===== {now()} {why} ({memory()}) =====\n")
    faulthandler.dump_traceback(file=stacks_file, all_threads=True)
    stacks_file.flush()


def watcher():
    fails, last_line, started_ok = 0, 0.0, False
    while True:
        # re-armed every loop: if this thread cannot run for FREEZE_AFTER s, a C thread dumps every stack (no GIL needed)
        faulthandler.dump_traceback_later(FREEZE_AFTER, repeat=False, file=stacks_file)
        time.sleep(PROBE_EVERY)
        ok, ms, err = probe()
        started_ok = started_ok or ok
        line = f"{now()} health={'ok' if ok else 'FAIL'} {ms:.0f} ms {memory()} {err}"
        if not ok:
            if started_ok:                  # before the first ok the model is just loading
                fails += 1
                watch_log.write(line + f" (fail #{fails})\n")
                if fails == 2:
                    dump("health failed twice in a row")
        else:
            fails = 0
            if ms > 1000 or time.time() - last_line > 60:
                watch_log.write(line + "\n")
                last_line = time.time()


threading.Thread(target=watcher, daemon=True, name="hang-watcher").start()

here = Path(os.environ.get("HANGWATCH_TARGET") or ROOT / "serve" / "server.py")
sys.argv[0] = str(here)
sys.path.insert(0, str(here.parent))
runpy.run_path(str(here), run_name="__main__")
