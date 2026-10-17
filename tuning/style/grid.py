# Which request setting decides the thinking style?  2 x 2 x 2 grid on short requests (20 tokens: the opening only):
#   sampling: greedy (what the old page gets) vs 0.6 / 0.95 / 20 (the green page's)
#   refusal remover (experimental_speed_projection): on vs off
#   tools: the old page's get_datetime + get_info vs none
# Greedy is deterministic, so 2 runs per cell (to confirm it repeats); sampled cells get N runs.
import json, sys, time, urllib.request, collections
from pathlib import Path
DT = {"type": "function", "function": {"name": "get_datetime", "description": "Returns the current local date and time in ISO 8601 format, with the IANA time zone name", "parameters": {"type": "object", "properties": {}, "required": []}}}
INFO = {"type": "function", "function": {"name": "get_info", "description": "Get runtime info (OS name), may call when user asks about local files or shell commands", "parameters": {"type": "object", "properties": {}, "required": []}}}
QS = {"pelican": "Generate an SVG of a pelican riding a bicycle",
      "say hi": "say hi, and say something fun too",
      "CPU": "Explain how a modern CPU works.",
      "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "HU poem": "Írj egy rövid verset az őszről."}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
def style(th):
    if th.startswith(("We ", "Need", "User ", "Simple", "Answer")): return "clipped"
    if th.startswith(("The user", "Okay", "Alright", "I ", "Let me", "So ", "Hmm")): return "prose"
    return "other"
def post(q, greedy, proj, tools):
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-grid", "max_tokens": 20, "reasoning_effort": "high",
            "messages": [{"role": "user", "content": q}], "experimental_speed_projection": proj}
    if not greedy: body.update(temperature=0.6, top_p=0.95, top_k=20)
    if tools: body["tools"] = [DT, INFO]
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return (r["choices"][0]["message"].get("reasoning_content") or "").strip()
rows, t0 = [], time.time()
for qn, q in QS.items():
    for greedy in (True, False):
        for proj in (True, False):
            for tools in (True, False):
                ths = [post(q, greedy, proj, tools) for _ in range(2 if greedy else N)]
                c = collections.Counter(style(t) for t in ths)
                rows.append({"q": qn, "sampling": "greedy" if greedy else "0.6/0.95/20", "remover": "on" if proj else "off",
                             "tools": "yes" if tools else "no", "n": len(ths), "prose": c["prose"], "clipped": c["clipped"],
                             "other": c["other"], "examples": ths[:2]})
                print(f"{qn:10s} {rows[-1]['sampling']:11s} remover {rows[-1]['remover']:3s} tools {rows[-1]['tools']:3s}: "
                      f"prose {c['prose']}/{len(ths)}  clipped {c['clipped']}  other {c['other']}"
                      + (f"   e.g. {[t[:40] for t in ths if style(t) == 'other'][:1]}" if c["other"] else ""), flush=True)
Path(__file__).with_name("grid.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"{len(rows)} cells in {time.time() - t0:.0f} s")
