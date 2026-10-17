# turns.py's question with the written opening on: after a CLIPPED turn in the history (its thinking sent back, as the
# llama.cpp-style page and agents do, or the answer only), does the next turn think in prose when it starts with
# "Let me think about this. The user"?  Turn 1 ("say hi ...") made once in each style; turn 2 asks something else.
# Scored like prefix.py: 150 tokens of turn 2's own thinking (the written words removed), clipped markers vs prose.
import json, re, sys, time, urllib.request
from pathlib import Path
URL = "http://127.0.0.1:8080/v1/chat/completions"
UI = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-turns-prefix", "reasoning_effort": "high",
      "temperature": 0.6, "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True}
PREFIX = "Let me think about this. The user"
T1 = "say hi, and say something fun too"
T2 = {"CPU": "Explain how a modern CPU works.", "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "story": "Write a short story about a robot who learns to paint."}
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")


def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({**UI, **body}).encode(),
                     {"Content-Type": "application/json"}), timeout=3600))["choices"][0]["message"]


def own(th):
    th = th or ""
    return th[len(PREFIX):] if th.startswith(PREFIX) else th


def style(th):
    c, p = len(CLIP.findall(th)), len(PROSE.findall(th))
    return "prose" if p > c else "clipped" if c > p else "mixed"


first = {}
for name, pre in (("clipped", False), ("prose", True)):
    for _ in range(8):
        m = post({"messages": [{"role": "user", "content": T1}], "max_tokens": 1500, "strata_think_prefix": pre})
        if style(own(m.get("reasoning_content"))) == name and m.get("content"):
            first[name] = m
            break
    print(f"turn 1 ({name}): {(first.get(name) or {}).get('reasoning_content', 'NOT MADE')[:140]!r}", flush=True)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
cases = {"cold (no history), prefix on": (None, False, True),
         "prose turn kept with thinking, prefix on": ("prose", True, True),
         "CLIPPED turn kept with thinking, prefix on": ("clipped", True, True),
         "clipped turn, answer only, prefix on": ("clipped", False, True),
         "CLIPPED turn kept with thinking, prefix OFF": ("clipped", True, False)}
out, t0 = {}, time.time()
for label, (kind, keep, pre) in cases.items():
    if kind is not None and kind not in first:
        continue
    hist = []
    if kind is not None:
        a = {"role": "assistant", "content": first[kind]["content"]}
        if keep:
            a["reasoning_content"] = first[kind]["reasoning_content"]
        hist = [{"role": "user", "content": T1}, a]
    got = {}
    for qn, q in T2.items():
        got[qn] = []
        for _ in range(N):
            th = post({"messages": hist + [{"role": "user", "content": q}], "max_tokens": 150,
                       "strata_think_prefix": pre}).get("reasoning_content")
            got[qn].append({"style": style(own(th)), "thinking": (th or "")[:300]})
    prose = sum(r["style"] == "prose" for v in got.values() for r in v)
    out[label] = got
    print(f"{label:46s} turn 2 prose {prose:2d}/{N * len(T2)}  " +
          "  ".join(f"{k} {sum(r['style'] == 'prose' for r in v)}" for k, v in got.items()), flush=True)
Path(__file__).with_name("turns_prefix.json").write_text(json.dumps({"turn1": {k: v.get("reasoning_content") for k, v in first.items()},
                                                                     "cases": out}, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"done in {time.time() - t0:.0f} s")
