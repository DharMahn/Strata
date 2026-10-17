# Does the thinking of earlier turns carry its style into the next one?  The llama.cpp-style page sends each answer's
# thinking back in the history (reasoning_content; the template renders it); the web app sends only the answer text.
# Turn 1 ("say hi ..."), made once in each style; turn 2 asks something else, at High with the web app's sampling.
# Scored like prefix.py: 150 tokens of turn 2's thinking, the clipped style's markers against the prose style's.
import json, re, sys, time, urllib.request
from pathlib import Path
URL = "http://127.0.0.1:8080/v1/chat/completions"
UI = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-turns", "reasoning_effort": "high", "temperature": 0.6,
      "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True}
T1 = "say hi, and say something fun too"
T2 = {"CPU": "Explain how a modern CPU works.", "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "story": "Write a short story about a robot who learns to paint."}
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({**UI, **body}).encode(),
                     {"Content-Type": "application/json"}), timeout=3600))["choices"][0]["message"]
def style(th):
    c, p = len(CLIP.findall(th)), len(PROSE.findall(th))
    return "prose" if p > c else "clipped" if c > p else "mixed"
# turn 1 in each style: plain High gives the clipped notes; the written opening gives prose
first = {}
for name, extra in (("clipped", {}), ("prose", {"strata_think_prefix": "Let me think about this. The user"})):
    for _ in range(6):
        m = post({"messages": [{"role": "user", "content": T1}], "max_tokens": 1500, **extra})
        th = m.get("reasoning_content") or ""
        if style(th) == name and m.get("content"):
            first[name] = m
            break
    print(f"turn 1 ({name}): {(first.get(name) or {}).get('reasoning_content', 'NOT MADE')[:120]!r}", flush=True)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
cases = {"history with prose thinking (old page)": ("prose", True),
         "history with clipped thinking (old page)": ("clipped", True),
         "prose turn, answer only (web app)": ("prose", False),
         "clipped turn, answer only (web app)": ("clipped", False)}
t0 = time.time()
for label, (kind, keep) in cases.items():
    if kind not in first:
        continue
    a = {"role": "assistant", "content": first[kind]["content"]}
    if keep:
        a["reasoning_content"] = first[kind]["reasoning_content"]
    got = {}
    for qn, q in T2.items():
        msgs = [{"role": "user", "content": T1}, a, {"role": "user", "content": q}]
        got[qn] = [style(post({"messages": msgs, "max_tokens": 150}).get("reasoning_content") or "") for _ in range(N)]
    prose = sum(v.count("prose") for v in got.values())
    print(f"{label:42s} turn 2 prose {prose}/{N * len(T2)}  " + "  ".join(f"{k} {v.count('prose')}" for k, v in got.items()),
          flush=True)
print(f"done in {time.time() - t0:.0f} s")
