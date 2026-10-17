# The model card's thinking temperature (1.0) vs the web app's 0.6, with and without a written opening.
# Same scoring as prefix.py: 150 tokens of thinking after the opening, the clipped style's markers ("Need ...",
# "We need", "Let's") against the prose style's ("Let me", "I'll", "I should", ...).  High, remover on.
#   python tuning/style/temp.py N "opening 1|opening 2"      ("" = none)
import json, re, sys, time, urllib.request
from pathlib import Path
QS = {"pelican": "Generate an SVG of a pelican riding a bicycle", "say hi": "say hi, and say something fun too",
      "CPU": "Explain how a modern CPU works.", "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "HU poem": "Írj egy rövid verset az őszről.", "story": "Write a short story about a robot who learns to paint."}
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
def run(q, prefix, temp):
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "temp-test", "max_tokens": 150, "reasoning_effort": "high",
            "temperature": temp, "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True,
            "messages": [{"role": "user", "content": q}]}
    if prefix: body["strata_think_prefix"] = prefix
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    th = r["choices"][0]["message"].get("reasoning_content") or ""
    own = th[len(prefix):] if prefix and th.startswith(prefix) else th
    c, p = len(CLIP.findall(own)), len(PROSE.findall(own))
    return "prose" if p > c else "clipped" if c > p else "mixed"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
prefixes = sys.argv[2].split("|") if len(sys.argv) > 2 else ["", "The user"]
t0 = time.time()
for temp in [float(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "1.0").split(",")]:
    for prefix in prefixes:
        rows = {qn: [run(q, prefix, temp) for _ in range(N)] for qn, q in QS.items()}
        prose = sum(s == "prose" for v in rows.values() for s in v)
        print(f"temp {temp}  {prefix or '(none)':36s} prose {prose:2d}/{N * len(QS)}  " +
              "  ".join(f"{qn} {v.count('prose')}" for qn, v in rows.items()), flush=True)
print(f"done in {time.time() - t0:.0f} s")
