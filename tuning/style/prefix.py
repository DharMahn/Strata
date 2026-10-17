# Which written-for-it opening keeps the thinking in prose?  High (xhigh) + the green page's sampling, remover on.
# Each run: 150 tokens of thinking after the opening, scored by the clipped style's markers ("Need ...", "We need",
# "Let's") against the prose style's ("Let me", "I'll", "I should", ...), counted in what the model wrote itself.
import json, re, sys, time, urllib.request
from pathlib import Path
QS = {"pelican": "Generate an SVG of a pelican riding a bicycle", "say hi": "say hi, and say something fun too",
      "CPU": "Explain how a modern CPU works.", "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "HU poem": "Írj egy rövid verset az őszről.", "story": "Write a short story about a robot who learns to paint."}
PREFIXES = ["", "The user", "The user is asking", "Okay, so the user", "Let me think about this. The user",
            "Alright, let me", "I need to"]
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
def run(q, prefix):
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "prefix-search", "max_tokens": 150, "reasoning_effort": "high",
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True,
            "messages": [{"role": "user", "content": q}]}
    if prefix: body["strata_think_prefix"] = prefix
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    th = r["choices"][0]["message"].get("reasoning_content") or ""
    own = th[len(prefix):] if prefix and th.startswith(prefix) else th
    c, p = len(CLIP.findall(own)), len(PROSE.findall(own))
    return ("prose" if p > c else "clipped" if c > p else "mixed"), c, p, th
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
out, t0 = {}, time.time()
for prefix in PREFIXES:
    rows = {qn: [run(q, prefix) for _ in range(N)] for qn, q in QS.items()}
    prose = sum(r[0] == "prose" for v in rows.values() for r in v)
    out[prefix or "(none)"] = {qn: [{"style": s, "clipped_marks": c, "prose_marks": p, "thinking": th[:500]}
                                    for s, c, p, th in v] for qn, v in rows.items()}
    print(f"{prefix or '(none)':36s} prose {prose:2d}/{N * len(QS)}  " +
          "  ".join(f"{qn} {sum(r[0] == 'prose' for r in v)}" for qn, v in rows.items()), flush=True)
Path(__file__).with_name("prefix.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"done in {time.time() - t0:.0f} s")
