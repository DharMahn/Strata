# The two thinking styles (Qwen3.8-Flash-Next, IQ3_XXS / IQ2_XS)

The model thinks in one of two styles, and the first words decide which one it keeps for the whole answer:

- **clipped notes**: "We need answer user: ... Need produce SVG ... Let's craft." Short (13-16K tokens on the pelican),
  and the drawings and texts it leads to are plain.
- **prose**: "The user wants an SVG of a pelican riding a bicycle. Let me create ..." Longer (19-40K tokens on the
  pelican) and much better results: every prose run at High drew a full scene, every clipped run a bare bird on a bike.

## What picks the style (2026-10-01, IQ3_XXS, the web app's sampling 0.6 / 0.95 / 20)

Short requests (20 tokens: the opening), `grid.py` / `search.py`:

- **The template's effort instruction.** High (= xhigh, the template's default) puts "Reasoning effort is set to xhigh.
  Please think carefully through the task, validate key assumptions, ..." in a system message; that wording (either
  sentence alone, or "Reasoning effort is set to high.") is the trigger for the clipped notes. Medium (no instruction)
  and Low: prose 24/24 - but Medium thinks only 2.3-2.9K tokens on the pelican and draws plainly (`../pelican/`).
- **Not the sampling.** Greedy vs 0.6 / 0.95 / 20: no difference on ordinary questions (clipped every time).
- **The refusal remover** (`experimental_speed_projection`) pushes slightly towards clipped on near-tie questions
  (pelican: 3/10 prose on, 6/10 off).
- **Tools dilute the trigger.** A tool block between the instruction and the question: the old llama.cpp-style page's
  `get_datetime` + `get_info` made the pelican prose 20/20, but most other questions stay clipped; six bogus tools
  16/24. A system prompt never helps at High (the template merges the xhigh instruction into it): 0-2/24, even one
  that says to start with "The user wants".
- A plain "Think carefully and thoroughly before answering." at Medium keeps prose (24/24), thinks a little longer
  (3.4-5.9K on the pelican), still plain drawings.

## The fix: write the thinking's first words (in use since 2026-10-01 23:10)

`strata_think_prefix` (serve/server.py): the server appends the words after the open `<think>` and sends them as the
start of the reasoning.  High keeps its long thinking, the opening keeps it in prose.  `true` = THINK_PREFIX,
"Let me think about this. The user".  Defaults: the web app's "Start thinking in prose" switch (on; it sends true or
false either way) and the run configs' `"sampling": {"strata_think_prefix": true}` for every other client (setup.py
keeps a config's sampling block when it is run again).

Which opening holds (`prefix.py`, `temp.py`: 150 tokens of thinking after it, the clipped style's markers against the
prose style's, six kinds of request):

| opening | prose |
|---|---|
| none | 1/18, 1/24 at temperature 1.0 |
| "The user" | 8/18 (it continues "The user asks: ... Need provide ...") |
| "I need to" | 5/18 |
| "The user is asking" / "Okay, so the user" | 16/18 each |
| "Alright, let me" | 17/18, 20/24 (t 1.0), 21/24 (t 0.6) |
| **"Let me think about this. The user"** | **17/18, 24/24 (t 1.0), 23/24 (t 0.6)** |

The same three-turn chat (pelican, "now make it animated", "explain how a modern CPU works in 200 words") in three
setups, from the request log (STRATA_REQUEST_LOG):

| turn | llama.cpp-style page (2 tools, keeps thinking) | web app, switch off | web app, switch on |
|---|---|---|---|
| pelican | prose, 16,325 tokens, 8,499-char SVG | clipped, 18,845, 3,148 | prose, 41,991, 9,846 |
| animated | prose, 18,488, 14,199 chars | clipped, 29,743, 5,009 | prose, 18,312, 15,700 |
| CPU, 200 words | prose, 1,096 tokens | clipped, 6,058 | prose, 3,408 |

The old page's edge was its tool block (tips near-tie questions such as the pelican) plus sending the thinking back
(`turns.py`: after a prose turn kept with its thinking, the next question was prose 4/9, else 0/9; in the real
session it carried through all turns).  Cold, it is clipped too (the same CPU question in a new chat: "We need answer
user's request ...", 3,445 tokens).

## Online (2026-10-01)

No report of the clipped style or of the effort wording as its trigger was found.  The model card recommends
temperature 1.0 for thinking (the web app uses 0.6).  Reports on the 27B describe xhigh thinking in prose ("The user
is asking for ...") and xhigh drawing the best pelican (22K reasoning tokens) - which matches the prose runs here; the
clipped style may be stronger in these 2-3 bit quantizations than in the full model (not tested).

## With the opening on, after a clipped turn (2026-10-02, `turns_prefix.py`)

Turn 1 made once clipped (opening off) and once in prose; turn 2 (three questions, 4 runs each, High, t 0.6) with
the opening on unless noted.  Scored on turn 2's own thinking (the written words removed):

| history | turn 2 prose |
|---|---:|
| none (cold) | 12/12 |
| prose turn, thinking sent back | 12/12 |
| clipped turn, answer only (the web app; Hermes, which strips `reasoning_content` for non-DeepSeek/Kimi/MiMo providers) | 12/12 |
| clipped turn, thinking sent back (the llama.cpp-style page) | 9/12 (story 1/4: starts in prose, slides into "Need ...") |
| clipped turn, thinking sent back, opening OFF | 0/12 |

So a clipped turn in the history only matters to a client that sends the thinking back, and even there the opening
mostly holds.  If one ever does: the template's `preserve_thinking` false drops the thinking of turns before the last
user message (the answer-only case).

## Not after a tool result (2026-10-02, `tool_turns.py`)

In a long Hermes session (a pelican SVG, 15-28K-token context: Hermes's system prompt and tools, the code, the tool
output; Hermes strips earlier thinking), the opening was written at the start of every step, including those after a
tool result.  "Let me think about this. The user" then had to restate a request 20K+ tokens back, and several steps
restated stock reasoning-benchmark requests instead (a prime proof, Fermat n=4 by descent, an SF-LA route comparison,
Binet's formula; one such step had 98% draft acceptance - recited text), while the tool calls stayed on the task.
In short synthetic loops the opening stayed on task, so the drift needs the long agent context.

Fix (serve/server.py `run`): no opening when the prompt ends with a tool result (`</tool_response><|im_end|>` then the
generation prompt); a turn that starts from the user's message keeps it.  `tool_turns.py` afterwards: the steps after
a tool result begin in the model's own words and on task ("The output shows 5 .py files. Let me verify ...",
"python3 isn't found. Let's try python.").  Tests: serve/test_server.py ThinkPrefix (59 of 59 pass).

## The steps after a tool result, in a real Hermes session (2026-10-03, `hermes_replay.py`)

With no opening after a tool result, Hermes's pelican run of 10-02 23:19 (its own system prompt and tools, 19K -> 44K
tokens of context) had 6 of its 18 such steps in the clipped notes - among them both steps that rewrote the drawing.
`hermes_replay.py` rebuilds those 18 requests from Hermes's database (state.db keeps the system prompt, the tool
definitions and every step's thinking) and asks for each step's thinking again, twice, 400 tokens each:

| setup | clipped replies (of 36) | replies with any clipped marker | clipped markers per 1000 chars | drift |
|---|---:|---:|---:|---:|
| as it was: earlier thinking stripped, no opening | 9 | 13 | 1.9 | 0 |
| opening "Let me" (`strata_think_prefix_tool`) | 3 | 5 | 0.7 | 0 |
| the turn's earlier thinking sent back (Hermes `model.reasoning_echo: true`) | 4 | 7 | 0.6 | 0 |
| **both** | **0** | **4** | **0.3** | **0** |

(The echo rows replay the session's own thinking, clipped steps included; a session made with echo on has fewer of
those to copy.  Echo costs context: +8-10K tokens at the end of that run.)

In use since 2026-10-03: `"strata_think_prefix_tool": true` in both run configs' `sampling` block (backups
`*.before-toolopening`) and `reasoning_echo: true` under `model:` in Hermes's config.yaml (backup
`config.yaml.before-echo`).  A request that turns `strata_think_prefix` off gets no opening after tool results
either, unless it sets `strata_think_prefix_tool` itself.

How long the echoed thinking stays: the chat template keeps all the thinking a client sends back
(`preserve_thinking` undefined) - every earlier request's too, until Hermes compresses the conversation.  In
Hermes's sessions thinking was 6-27% of what a tool-heavy request added and 92% of a small one (3.7K tokens of
thinking, 0.3K of everything else).  So the run configs also have `"preserve_thinking": false` in `sampling`
(serve/server.py `thinking_kept`): the template then keeps the thinking of the steps after the last user message
only.  Within a request the prompt only grows at its end (nothing is read twice); a new message drops the
previous request's thinking, and that request's steps are read again without it (about what Hermes re-read
before echo, one step at a time).  Checked on the 15:08 session: a step inside the request 18,388 tokens (15,449
with the thinking stripped); the next user message 15,603 tokens, 19,414 with everything kept.  Only requests
that carry thinking are affected; a request's own `chat_template_kwargs.preserve_thinking` wins.

Checked end to end with a small Hermes task (two terminal steps, 15:08): the first step began with the main opening,
both steps after a tool result with "Let me", all three in prose; and the second request was 17,039 tokens reused +
136 read of 17,175 - the 2,980 tokens the first step generated (its thinking included) stayed in place.

## Echo taken back out (2026-10-03 16:00)

The first real Hermes run with echo on (pelican, 15 steps, session 15:20): every step after a tool result was prose,
but from step 7 on the thinking was the same 1,332 characters word for word, nine steps in a row (96-98% of drafts
accepted: recited, not thought), while the tool calls went on (image checks, two patches).  With its own earlier
thinking in the context the model copies it; the replay above could not show that - it scored only how each step's
thinking began, against a stored history with varied thinking.  `reasoning_echo: false` again in Hermes's
config.yaml; the tool-step opening stays (3 of 36 clipped in the replay, nothing in the context to recite).
`preserve_thinking: false` stays in the run configs for clients that do send thinking back.

That run's drawing: one draft (4,981 characters, 58 elements; the web app's prose runs: 8.5-15.7K characters), never
rewritten - Hermes's own learned skill `svg-illustration` (in its skills folder, written by the agent after the 10-02
run) says draw once, audit with yes/no image questions, patch coordinates, keep the background clear.
