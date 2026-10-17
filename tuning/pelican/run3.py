# Full pelican runs at High with the thinking's opening written for the model (strata_think_prefix), the web app's
# sampling (0.6 / 0.95 / 20, or the temperature given), remover on.  Streams, and keeps <name>.partial.txt up to date
# while it runs, so a stopped run still leaves what was written (the first try lost a finished 37K-token pelican).
#   python run3.py "<opening>" [temperature] [seeds, e.g. 1,2,3]
import json, re, sys, time, urllib.request
from pathlib import Path
OUT = Path(__file__).parent
Q = "Generate an SVG of a pelican riding a bicycle"
prefix = sys.argv[1]
temp = float(sys.argv[2]) if len(sys.argv) > 2 else 0.6
seeds = [int(s) for s in (sys.argv[3] if len(sys.argv) > 3 else "1,2,3").split(",")]
for seed in seeds:
    name = f"prefix-t{temp:g}-s{seed}"
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "pelican-prefix", "max_tokens": 60000, "seed": seed,
            "stream": True, "messages": [{"role": "user", "content": Q}], "reasoning_effort": "high",
            "temperature": temp, "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True,
            "strata_think_prefix": prefix}
    t0, last_save, usage, finish = time.time(), 0.0, {}, None
    reasoning, content = [], []
    partial = OUT / f"{name}.partial.txt"
    with urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                                json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=7200) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: {"):
                continue
            c = json.loads(line[6:])
            for ch in c.get("choices", []):
                d = ch.get("delta", {})
                reasoning.append(d.get("reasoning_content") or "")
                content.append(d.get("content") or "")
                finish = ch.get("finish_reason") or finish
            usage = c.get("usage") or usage
            if time.time() - last_save > 5:
                partial.write_text("".join(reasoning) + "\n\n=== answer ===\n\n" + "".join(content), encoding="utf-8")
                last_save = time.time()
    th, answer = "".join(reasoning).strip(), "".join(content)
    (OUT / f"{name}.json").write_text(json.dumps({"reasoning_content": th, "content": answer, "usage": usage,
                                                  "finish_reason": finish}, indent=1, ensure_ascii=False), encoding="utf-8")
    partial.unlink(missing_ok=True)
    svg = re.search(r"<svg\b.*?</svg>", answer, re.S)
    if svg:
        (OUT / f"{name}.svg").write_text(svg.group(0), encoding="utf-8")
    own = th[len(prefix):]
    c = len(re.findall(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|Let's)\b", own))
    p = len(re.findall(r"\b(Let me|I'll|I should|I need|I think|I'm)\b", own))
    style = "prose" if p > c else "clipped"
    line = (f"{name}: {style} thinking, {usage.get('completion_tokens')} tokens, {time.time() - t0:.0f} s, "
            f"finish {finish}, svg {'yes' if svg else 'NO'}, markers clipped {c} prose {p}, opens: {th[:70]!r}")
    print(line, flush=True)
    with open(OUT / "summary.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")
