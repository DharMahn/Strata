# Full pelican runs at reasoning effort "medium" (the template adds no effort instruction, so the model thinks in
# prose), alone and with a plain "think thoroughly" system message.  The green page's sampling, remover on, seeds 1-3.
import json, re, time, urllib.request
from pathlib import Path
OUT = Path(__file__).parent
Q = "Generate an SVG of a pelican riding a bicycle"
SYS = "Think carefully and thoroughly before answering."
runs = [(f"{tag}-s{seed}", seed, sysmsg) for seed in (1, 2, 3) for tag, sysmsg in (("medium", None), ("medium-thorough", SYS))]
for name, seed, sysmsg in runs:
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": Q}]
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "pelican-medium", "messages": msgs, "max_tokens": 48000,
            "reasoning_effort": "medium", "temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": seed,
            "experimental_speed_projection": True}
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200))
    m = r["choices"][0]["message"]
    (OUT / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")
    th, content = (m.get("reasoning_content") or "").strip(), m.get("content") or ""
    svg = re.search(r"<svg\b.*?</svg>", content, re.S)
    if svg:
        (OUT / f"{name}.svg").write_text(svg.group(0), encoding="utf-8")
    style = "clipped" if th.startswith(("We ", "Need", "User ")) else "prose" if th.startswith("The user") else "other"
    line = (f"{name}: {style} thinking, {r['usage']['completion_tokens']} tokens, {time.time() - t0:.0f} s, "
            f"finish {r['choices'][0].get('finish_reason')}, svg {'yes' if svg else 'NO'}, opens: {th[:70]!r}")
    print(line, flush=True)
    with open(OUT / "summary.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
