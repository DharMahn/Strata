# The written opening in an agent's tool loop: after a tool result the model is mid-task, and "Let me think about
# this. The user" makes it restate the request - which in a long Hermes session (2026-10-02) drifted into requests
# nobody made.  Here: synthetic tool loops (a task, the model's tool call, the tool's result), then the model's next
# thinking with the opening on and off.  Scored: the style (prose markers vs clipped, the written words removed) and
# whether the thinking is about the task (its keywords) rather than a new request.
import json, re, sys, urllib.request
URL = "http://127.0.0.1:8080/v1/chat/completions"
UI = {"model": "qwen3.8-flash-next-iq3_xxs", "user": "style-tool-turns", "reasoning_effort": "high",
      "temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 200}
PREFIX = "Let me think about this. The user"
TOOLS = [{"type": "function", "function": {"name": "terminal", "description": "Run a shell command and return its output.",
          "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
         {"type": "function", "function": {"name": "write_file", "description": "Write text to a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
SYS = {"role": "system", "content": "You are a helpful agent with tools. Use them to complete the user's task, then report."}


def call(name, args, cid):
    return {"role": "assistant", "content": "", "tool_calls": [{"id": cid, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}]}


CASES = {
    "count files": ([SYS, {"role": "user", "content": "How many .py files are in C:\\proj\\serve? Use the terminal."},
                     call("terminal", {"command": "dir /b C:\\proj\\serve\\*.py"}, "c1"),
                     {"role": "tool", "tool_call_id": "c1", "content": "chat.py\nfrontend.py\nserver.py\ntest_server.py\nwinjob.py"}],
                    ("py", "file", "five", "5")),
    "svg two steps": ([SYS, {"role": "user", "content": "Make an SVG of a pelican riding a bicycle, save it to pelican.svg, "
                                                           "then check that it parses as XML."},
                       call("write_file", {"path": "pelican.svg", "content": "<svg xmlns='http://www.w3.org/2000/svg' "
                            "viewBox='0 0 400 300'><circle cx='120' cy='220' r='50'/><circle cx='280' cy='220' r='50'/>"
                            "<ellipse cx='200' cy='130' rx='60' ry='35' fill='white' stroke='black'/></svg>"}, "c1"),
                       {"role": "tool", "tool_call_id": "c1", "content": "Wrote 231 bytes to pelican.svg"}],
                      ("svg", "pelican", "xml", "pars")),
    "failed command": ([SYS, {"role": "user", "content": "What Python version is installed? Check with the terminal."},
                        call("terminal", {"command": "python3 --version"}, "c1"),
                        {"role": "tool", "tool_call_id": "c1", "content": "'python3' is not recognized as an internal or "
                         "external command,\noperable program or batch file."}],
                       ("python", "version", "py ", "recogni")),
}
CLIP = re.compile(r"(?:^|[.!?:]\s+|\n\s*)(Need|We need|We should|We can|We must|Let's|Probably|Maybe|Could)\b")
PROSE = re.compile(r"\b(Let me|I'll|I should|I need|I think|I'm|I will|I want|I can|I'd|I've)\b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4


def post(body):
    return json.load(urllib.request.urlopen(urllib.request.Request(URL, json.dumps({**UI, **body}).encode(),
                     {"Content-Type": "application/json"}), timeout=600))["choices"][0]["message"]


for label, (msgs, keys) in CASES.items():
    for pre in (True, False):
        style, on_task, heads = {"prose": 0, "clipped": 0, "mixed": 0}, 0, []
        for _ in range(N):
            m = post({"messages": msgs, "tools": TOOLS, "strata_think_prefix": pre})
            th = m.get("reasoning_content") or ""
            own = th[len(PREFIX):] if th.startswith(PREFIX) else th
            c, p = len(CLIP.findall(own)), len(PROSE.findall(own))
            style["prose" if p > c else "clipped" if c > p else "mixed"] += 1
            on_task += any(k in th.lower() for k in keys)
            heads.append(th[:110].replace("\n", " "))
        print(f"{label:15s} opening {'ON ' if pre else 'off'}: prose {style['prose']}/{N}, clipped {style['clipped']}, "
              f"on task {on_task}/{N}", flush=True)
        for h in heads:
            print(f"      {h!r}", flush=True)
