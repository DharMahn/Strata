# The clipped notes in a real Hermes session, replayed (2026-10-03).  In the pelican run of 10-02 23:19 (state.db:
# Hermes's own system prompt and 30-odd tools, 16K -> 44K tokens of context) the first step had the written opening
# and thought in prose; the steps after a tool result have none since tool_turns.py, and 6 of 18 were clipped notes -
# among them both steps that rewrote the drawing.  This rebuilds each of those 18 requests from Hermes's database
# (the stored tool calls and results stay as they were) and asks for the step's thinking again, under:
#   none        what Hermes gets today (earlier thinking stripped, no opening after a tool result)
#   open:<text> strata_think_prefix_tool = <text>
#   echo        the turn's earlier steps keep their thinking (Hermes: model.reasoning_echo true)
#   echo+open   both
# Scored on the model's own words: prose vs clipped markers, on task (the task's words), drift (a request nobody made).
# Nothing of the session is printed - labels and counts only; the openings go to hermes_replay.json.
#   python tuning/style/hermes_replay.py [runs per step] [max tokens] [arms, comma separated]
import json, re, sqlite3, sys, time, urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
DB = Path.home() / "AppData/Local/hermes/state.db"
SESSION = "20261002_231921_10041e"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 400
ARMS = (sys.argv[3] if len(sys.argv) > 3 else "none,open:Let me,echo,echo+open:Let me").split(",")
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
DRIFT = re.compile(r"\b(proof|prove|prime|theorem|San Francisco|Los Angeles|Binet|Fibonacci|essay|integral|recipe|"
                   r"descent|haiku|poem)\b", re.I)
TASK = ("pelican", "svg", "bicycle", "bike", "beak", "wheel", "neck", "render", "image", "screenshot", "file", "png",
        "browser", "drawing", "pouch", "leg", "pedal")


def load():
    c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sys_hash, tools_hash = c.execute("select system_prompt_hash, tool_names from sessions where id=?", (SESSION,)).fetchone()
    get = lambda h: c.execute("select prompt from system_prompts where hash=?", (h,)).fetchone()[0]   # noqa: E731
    tools = json.loads(get(tools_hash))["tools"]
    msgs = [{"role": "system", "content": get(sys_hash)}]
    for role, content, tcid, tcalls, rc in c.execute(
            "select role, content, tool_call_id, tool_calls, reasoning_content from messages where session_id=? "
            "order by id", (SESSION,)):
        m = {"role": role, "content": content or ""}
        if role == "tool":
            m["tool_call_id"] = tcid
        if tcalls:
            m["tool_calls"] = [{"id": t["id"], "type": "function", "function": t["function"]} for t in json.loads(tcalls)]
        if role == "assistant":
            m["_thinking"] = rc or ""
        msgs.append(m)
    return msgs, tools


def request(msgs, upto, echo):
    out = []
    for m in msgs[:upto]:
        m = dict(m)
        th = m.pop("_thinking", None)
        if echo and th:
            m["reasoning_content"] = th
        out.append(m)
    return out


def post(body):
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    return r["choices"][0]["message"], r.get("usage") or {}


def main():
    msgs, tools = load()
    points = [i for i, m in enumerate(msgs) if m["role"] == "assistant" and msgs[i - 1]["role"] == "tool"]
    was = []
    for i in points:
        th = msgs[i]["_thinking"]
        c, p = len(CLIP.findall(th)), len(PROSE.findall(th))
        was.append("prose" if p > c else "clipped" if c > p else "mixed")
    print(f"{len(points)} steps after a tool result; in the session itself: prose {was.count('prose')}, clipped "
          f"{was.count('clipped')}, mixed {was.count('mixed')}", flush=True)
    out = {}
    for arm in ARMS:
        echo = arm.startswith("echo")
        opening = arm.split("open:", 1)[1] if "open:" in arm else False
        st = {"prose": 0, "clipped": 0, "mixed": 0}
        on = drift = 0
        was_clipped = {"prose": 0, "n": 0}
        rows, t0 = [], time.time()
        for k, i in enumerate(points):
            for _ in range(N):
                m, usage = post({"model": "qwen3.8-flash-next-iq3_xxs", "user": f"hermes-replay-{arm}",
                                 "messages": request(msgs, i, echo), "tools": tools, "max_tokens": MAX_NEW,
                                 "strata_think_prefix_tool": opening})
                th = m.get("reasoning_content") or ""
                own = th[len(opening):] if opening and th.startswith(opening) else th
                c, p = len(CLIP.findall(own)), len(PROSE.findall(own))
                s = "prose" if p > c else "clipped" if c > p else "mixed"
                st[s] += 1
                on += any(w in th.lower() for w in TASK)
                d = bool(DRIFT.search(own))
                drift += d
                if was[k] == "clipped":
                    was_clipped["n"] += 1
                    was_clipped["prose"] += s == "prose"
                rows.append({"step": k, "was": was[k], "style": s, "clip": c, "prose": p, "drift": d,
                             "thinking_chars": len(own), "prompt_tokens": usage.get("prompt_tokens"),
                             "calls": [t["function"]["name"] for t in m.get("tool_calls") or []], "head": th[:300]})
        total = len(points) * N
        print(f"{arm:22s} prose {st['prose']:2d}/{total} clipped {st['clipped']:2d} mixed {st['mixed']:2d} | the 6 "
              f"that were clipped: prose {was_clipped['prose']}/{was_clipped['n']} | on task {on}/{total} drift "
              f"{drift} | {time.time() - t0:.0f} s", flush=True)
        print("    per step: " + " ".join(f"{k}:" + "".join(r["style"][0] for r in rows if r["step"] == k)
                                          for k in range(len(points))), flush=True)
        out[arm] = rows
        Path(__file__).with_name("hermes_replay.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
