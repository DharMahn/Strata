# Which opening for an agent's steps after a tool result (strata_think_prefix_tool)?  None left the long revision
# steps of a Hermes pelican run in the clipped notes (the drawing got rewritten in that style); "Let me think about
# this. The user" there restated requests nobody made.  Candidates that do not name the request, on: the short loops
# of tool_turns.py, and a long-context proxy of the revision step - a ~13K-token agent prompt (many tools, rules), a
# first SVG draft, the render's critique - then the model's next thinking.  Scored on the model's own words: style
# (prose vs clipped markers), on task (the task's words), drift (a request that is not in the context).
import json, re, sys, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from tool_turns import CASES, TOOLS, call   # noqa: E402

URL = "http://127.0.0.1:8080/v1/chat/completions"
UI = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-tool-openings", "reasoning_effort": "high",
      "temperature": 0.6, "top_p": 0.95, "top_k": 20, "strata_think_prefix": True}
ARMS = [False, "Let me", "Okay, let me", "Let me think about this. The user"]
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
DRIFT = re.compile(r"\b(proof|prime|theorem|San Francisco|Los Angeles|Binet|Fibonacci|essay|integral|recipe)\b", re.I)


def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({**UI, **body}).encode(),
                     {"Content-Type": "application/json"}), timeout=900))["choices"][0]["message"]


def score(th, arm):
    own = th[len(arm):] if arm and th.startswith(arm) else th
    c, p = len(CLIP.findall(own)), len(PROSE.findall(own))
    return own, "prose" if p > c else "clipped" if c > p else "mixed"


# ---- the long proxy: an agent prompt with many tools, a first draft, the render's critique
def long_context():
    verbs = ["search", "read", "list", "move", "copy", "convert", "inspect", "schedule", "summarize", "translate",
             "compress", "render", "query", "fetch", "watch", "notify", "archive", "diff", "lint", "format"]
    nouns = ["files", "web pages", "images", "calendar events", "emails", "notes", "spreadsheets", "processes",
             "packages", "screenshots", "logs", "repositories"]
    tools = list(TOOLS)
    for i in range(46):
        v, n = verbs[i % len(verbs)], nouns[i % len(nouns)]
        desc = (f"{v.capitalize()} {n} on the user's machine. " +
                " ".join(f"Use this tool when the task needs to {v} {n}; prefer it over shell commands, check the "
                         f"result before relying on it, and report failures plainly (rule {k})." for k in range(9)))
        tools.append({"type": "function", "function": {"name": f"{v}_{n.replace(' ', '_')}_{i}", "description": desc,
                      "parameters": {"type": "object", "properties": {"target": {"type": "string"},
                                     "options": {"type": "string"}}, "required": ["target"]}}})
    rules = " ".join(f"Operating rule {k}: work step by step with the tools, verify every result you rely on, keep "
                     f"the user informed only when the task is done, never invent file contents, and prefer the "
                     f"simplest tool that does the job." for k in range(60))
    svg = (Path(__file__).parents[1] / "pelican" / "proj-on-s1.svg").read_text(encoding="utf-8")
    msgs = [{"role": "system", "content": "You are an autonomous agent on the user's PC. " + rules},
            {"role": "user", "content": "generate me an SVG of a pelican riding a bicycle"},
            call("write_file", {"path": "pelican.svg", "content": svg}, "c1"),
            {"role": "tool", "tool_call_id": "c1", "content": f"Wrote {len(svg)} bytes to pelican.svg"},
            call("vision_analyze", {"image": "pelican.png", "question": "Does this look like a pelican riding a bicycle?"}, "c2"),
            {"role": "tool", "tool_call_id": "c2", "content":
             "The image shows a cartoon bird on a bicycle. The body is a white ellipse, but the neck is short and does "
             "not connect cleanly to the head; the beak and throat pouch are small, so it does not read clearly as a "
             "pelican. The wheels are plain circles without spokes, and the bird's legs do not reach the pedals."}]
    tools.append({"type": "function", "function": {"name": "vision_analyze", "description": "Describe an image.",
                  "parameters": {"type": "object", "properties": {"image": {"type": "string"},
                                 "question": {"type": "string"}}}}})
    return msgs, tools


N_SHORT = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_LONG = int(sys.argv[2]) if len(sys.argv) > 2 else 4
LONG_KEYS = ("pelican", "beak", "neck", "wheel", "spoke", "leg", "pedal", "pouch", "svg")
lmsgs, ltools = long_context()
out = {}
for arm in ARMS:
    name = repr(arm) if arm else "none"
    st, on, drift = {"prose": 0, "clipped": 0, "mixed": 0}, 0, 0
    for label, (msgs, keys) in CASES.items():
        for _ in range(N_SHORT):
            th = post({"messages": msgs, "tools": TOOLS, "max_tokens": 200,
                       "strata_think_prefix_tool": arm}).get("reasoning_content") or ""
            own, s = score(th, arm)
            st[s] += 1
            on += any(k in th.lower() for k in keys)
            drift += bool(DRIFT.search(own))
    print(f"short loops  {name:38s} prose {st['prose']}/{3 * N_SHORT} clipped {st['clipped']}  on task "
          f"{on}/{3 * N_SHORT}  drift {drift}", flush=True)
    st, on, drift, heads = {"prose": 0, "clipped": 0, "mixed": 0}, 0, 0, []
    for _ in range(N_LONG):
        th = post({"messages": lmsgs, "tools": ltools, "max_tokens": 600,
                   "strata_think_prefix_tool": arm}).get("reasoning_content") or ""
        own, s = score(th, arm)
        st[s] += 1
        on += any(k in th.lower() for k in LONG_KEYS)
        drift += bool(DRIFT.search(own))
        heads.append(th[:140].replace("\n", " "))
    out[name] = heads
    print(f"long proxy   {name:38s} prose {st['prose']}/{N_LONG} clipped {st['clipped']}  on task {on}/{N_LONG}  "
          f"drift {drift}", flush=True)
    for h in heads:
        print(f"      {h!r}", flush=True)
Path(__file__).with_name("tool_openings.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
