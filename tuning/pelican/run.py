# Full pelican runs against the running server: the green page's sampling, the projection on vs off, seeds 1-3.
# Saves each reply (JSON), its SVG, and a summary line (thinking style, tokens, time) to tuning/pelican/.
import json, re, sys, time, urllib.request
from pathlib import Path
OUT = Path(__file__).parent
Q = "Generate an SVG of a pelican riding a bicycle"
UI = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "reasoning_effort": "high"}
runs = [(f"{tag}-s{seed}", {**UI, "seed": seed, "experimental_speed_projection": on})
        for seed in (1, 2, 3) for tag, on in (("proj-on", True), ("proj-off", False))]
for name, extra in runs:
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "messages": [{"role": "user", "content": Q}],
            "max_tokens": 32000, **extra}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200))
    m = r["choices"][0]["message"]
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    th, content = (m.get("reasoning_content") or "").strip(), m.get("content") or ""
    svg = re.search(r"<svg\b.*?</svg>", content, re.S)
    if svg:
        (OUT / f"{name}.svg").write_text(svg.group(0), encoding="utf-8")
    style = "terse" if th.startswith(("We ", "Need")) else "prose" if th.startswith("The user") else "other"
    line = (f"{name}: {style} thinking, {r['usage']['completion_tokens']} tokens, {time.time() - t0:.0f} s, "
            f"finish {r['choices'][0].get('finish_reason')}, svg {'yes' if svg else 'NO'}, opens: {th[:70]!r}")
    print(line, flush=True)
    with open(OUT / "summary.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
