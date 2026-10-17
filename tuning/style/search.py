# What puts this model's thinking into prose ("The user wants ... Let me ...") instead of its clipped default
# ("We need answer user ...")?  Bogus tool lists and system prompts, screened on short requests (20 tokens: the
# opening decides the style for the whole answer).  The green page's settings: 0.6 / 0.95 / 20, remover on.
import json, sys, time, urllib.request, collections
from pathlib import Path
def fn(name, desc, props=None, req=None):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters":
            {"type": "object", "properties": props or {}, "required": req or []}}}
S = {"type": "string"}
OLD = [fn("get_datetime", "Returns the current local date and time in ISO 8601 format, with the IANA time zone name"),
       fn("get_info", "Get runtime info (OS name), may call when user asks about local files or shell commands")]
AGENT = [fn("run_shell", "Run a shell command on the user's computer and return its output", {"command": S}, ["command"]),
         fn("read_file", "Read a file from the user's computer", {"path": S}, ["path"]),
         fn("write_file", "Write a file on the user's computer", {"path": S, "content": S}, ["path", "content"])]
SEARCH = [fn("web_search", "Search the web and return the top results", {"query": S}, ["query"])]
NOOP = [fn("noop", "Does nothing. Never needs to be called.")]
CANDIDATES = {
    "baseline (green page)": {},
    "old page's 2 tools": {"tools": OLD},
    "get_info only": {"tools": OLD[1:]},
    "noop tool": {"tools": NOOP},
    "web_search tool": {"tools": SEARCH},
    "agent tools (shell/read/write)": {"tools": AGENT},
    "agent + old + search (6 tools)": {"tools": AGENT + OLD + SEARCH},
    "sys: helpful assistant": {"system": "You are a helpful assistant."},
    "sys: Qwen default": {"system": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    "sys: environment info": {"system": "Current date: 2026-10-01. Operating system: Windows 11. The assistant runs "
                              "locally on the user's PC."},
    "sys: coding agent": {"system": "You are an AI assistant running locally on the user's computer. You help with "
                          "software engineering, writing and general questions."},
    "effort medium (no system block)": {"effort": "medium"},
    "effort low": {"effort": "low"},
    "effort medium + 6 tools": {"effort": "medium", "tools": AGENT + OLD + SEARCH},
    "effort medium + agent tools": {"effort": "medium", "tools": AGENT},
    "medium + sys: the xhigh label only": {"effort": "medium", "system": "Reasoning effort is set to xhigh."},
    "medium + sys: the xhigh sentence only": {"effort": "medium", "system": "Please think carefully through the task, "
        "validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity "
        "in the final answer."},
    "medium + sys: label 'high'": {"effort": "medium", "system": "Reasoning effort is set to high."},
    "medium + sys: think thoroughly": {"effort": "medium", "system": "Think carefully and thoroughly before answering."},
    "medium + sys: helpful assistant": {"effort": "medium", "system": "You are a helpful assistant."},
    "sys: think aloud (names the opener)": {"system": "Think out loud in complete, natural sentences, in the first "
                          "person - start your reasoning with \"The user wants\" or \"The user asks\" and then work "
                          "through it with \"Let me ...\"."},
}
QS = {"pelican": "Generate an SVG of a pelican riding a bicycle", "say hi": "say hi, and say something fun too",
      "CPU": "Explain how a modern CPU works.", "palindrome": "Write a Python function that checks whether a string is a palindrome.",
      "HU poem": "Írj egy rövid verset az őszről.", "story": "Write a short story about a robot who learns to paint."}
PROSE = ("The user", "Okay", "Alright", "I ", "Let me", "So ", "Hmm", "A felhasználó", "The request", "This is")
CLIPPED = ("We ", "Need", "User ", "Simple", "Answer")
def style(th):
    return "prose" if th.startswith(PROSE) else "clipped" if th.startswith(CLIPPED) else "other"
def post(q, cand):
    msgs = ([{"role": "system", "content": cand["system"]}] if "system" in cand else []) + [{"role": "user", "content": q}]
    body = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-search", "max_tokens": 20, "reasoning_effort": cand.get("effort", "high"),
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "experimental_speed_projection": True, "messages": msgs}
    if "tools" in cand: body["tools"] = cand["tools"]
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions",
                  json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
    return (r["choices"][0]["message"].get("reasoning_content") or "").strip()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
only = sys.argv[2].split("|") if len(sys.argv) > 2 else None
out, t0 = {}, time.time()
for cname, cand in CANDIDATES.items():
    if only and cname not in only: continue
    per, others = {}, []
    for qn, q in QS.items():
        ths = [post(q, cand) for _ in range(N)]
        per[qn] = sum(style(t) == "prose" for t in ths)
        others += [t[:45] for t in ths if style(t) == "other"]
    total = sum(per.values())
    out[cname] = {"per_question": per, "prose": total, "of": N * len(QS), "other_examples": others[:3]}
    print(f"{cname:38s} prose {total:2d}/{N * len(QS)}  " + "  ".join(f"{k} {v}" for k, v in per.items())
          + (f"   other: {others[:2]}" if others else ""), flush=True)
Path(__file__).with_name(f"search-n{N}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"done in {time.time() - t0:.0f} s")
