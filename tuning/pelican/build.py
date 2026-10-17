# Builds compare.html from every run in summary.txt (run.py: High, remover on/off; run2.py: Medium, +/- "think
# thoroughly"), grouped by the thinking level, each drawing labelled with its thinking style, tokens and time.
import re
from pathlib import Path
HERE = Path(__file__).parent
GROUPS = [
    ("High (the green page's default)", "Reasoning effort xhigh: the template's instruction that triggers the clipped "
     "style. Refusal remover on and off, seeds 1-3.", lambda n: n.startswith("proj-")),
    ("High + the written opening", "Reasoning effort xhigh, the thinking's first words written for the model "
     "(strata_think_prefix). Remover on, seeds 1-3.", lambda n: n.startswith("prefix-")),
    ("Medium", "No effort instruction: prose thinking every time, but short.", lambda n: re.fullmatch(r"medium-s\d", n)),
    ("Medium + “Think carefully and thoroughly before answering.”", "The same plus a plain system message.",
     lambda n: n.startswith("medium-thorough-")),
]
runs = {}
for line in (HERE / "summary.txt").read_text(encoding="utf-8").splitlines():
    name = line.split(":")[0]
    if not (HERE / f"{name}.svg").exists():
        continue
    runs[name] = {"style": "prose" if "prose thinking" in line else "clipped",
                  "tokens": int(re.search(r"(\d+) tokens", line).group(1)), "secs": int(re.search(r"(\d+) s,", line).group(1))}
def card(name, r):
    svg = (HERE / f"{name}.svg").read_text(encoding="utf-8")
    svg = re.sub(r"<svg\b", '<svg class="pic"', svg, count=1)
    for i in set(re.findall(r'id="([^"]+)"', svg)):   # several drawings on one page: keep their ids apart
        svg = (svg.replace(f'id="{i}"', f'id="{name}-{i}"').replace(f"url(#{i})", f"url(#{name}-{i})")
                  .replace(f'href="#{i}"', f'href="#{name}-{i}"'))
    remover = "remover on · " if name.startswith("proj-on") else "remover off · " if name.startswith("proj-off") else ""
    seed = name.rsplit("-s", 1)[1]
    temp = f"temperature {name.split('-t')[1].split('-')[0]} · " if name.startswith("prefix-") else ""
    return (f'<figure class="card {r["style"]}"><div class="frame">{svg}</div><figcaption><b>'
            f'{"Prose" if r["style"] == "prose" else "Clipped"} thinking</b> · {remover}{temp}seed {seed}<br>'
            f'<span>{r["tokens"]:,} tokens, {r["secs"] // 60}:{r["secs"] % 60:02d}</span></figcaption></figure>')
sections = []
for title, note, match in GROUPS:
    names = sorted((n for n in runs if match(n)), key=lambda n: (runs[n]["style"] != "prose", n))
    if names:
        sections.append(f'<h2>{title}</h2><p>{note}</p><div class="grid">' + "\n".join(card(n, runs[n]) for n in names)
                        + "</div>")
html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Pelican A/B</title><style>
:root{{--bg:#f6f6f4;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#deded8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161616;--fg:#ececea;--muted:#9a9a94;--card:#202020;--line:#333}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:17px;margin:28px 0 2px}} p{{margin:0 0 14px;color:var(--muted);max-width:75ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
.card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.frame{{background:#fff;aspect-ratio:4/3;display:flex;align-items:center;justify-content:center}}
.pic{{width:100%;height:100%}} figcaption{{padding:10px 12px;font-size:14px}} figcaption span{{color:var(--muted)}}
.card.prose figcaption b{{color:#2f8f5b}} .card.clipped figcaption b{{color:#c0573a}}
</style></head><body>
<h1>Pelican A/B: IQ3_XXS, the green page's sampling (0.6 / 0.95 / 20)</h1>
<p>"Generate an SVG of a pelican riding a bicycle", run to the end. Grouped by the thinking level; within a group, prose
thinking first. Tokens include the thinking.</p>
{"".join(sections)}</body></html>'''
(HERE / "compare.html").write_text(html, encoding="utf-8")
print(f"compare.html: {len(runs)} drawings")
