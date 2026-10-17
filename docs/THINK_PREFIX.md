# The thinking-opening patch (`strata_think_prefix`)

**This is a behaviour change relative to upstream Strata.** It is off unless you turn it on, but once on it changes the
prompt every client sends the model, and it ships a web-app switch, three config keys and three request fields that
upstream does not have. Read this before enabling it for other people's clients.

## What it does

Qwen3.8-Flash-Next (the 2-3 bit quantizations tested here) opens its `<think>` block in one of two styles and keeps it:

- **clipped notes**: `We need answer user: ... Need produce SVG ... Let's craft.` Short, but the drawings and texts that
  follow were plainer.
- **prose**: `The user wants an SVG of a pelican riding a bicycle. Let me create ...` Longer, and better in the
  author's tests (every prose pelican run at High drew a full scene, every clipped run a bare bird).

The template's own "Reasoning effort is set to xhigh" instruction (the default at High) is what pushes it to clipped.
With the patch on, the server appends fixed words after the open `<think>` and streams them to the client as the
start of the reasoning, so the model continues in prose:

| Situation | Words written | Config key |
|---|---|---|
| The prompt ends at a user message | `Let me think about this. The user` | `strata_think_prefix` |
| The prompt ends at a tool result (agent loops) | `Let me` | `strata_think_prefix_tool` |
| Reasoning effort `none` (thinking off) | nothing | - |

After a tool result the long "The user ..." opening made the model restate a request 20K+ tokens back, so tool steps
get the shorter opening, which does not name the request.

## Why it is a breaking change

- The model's output differs: different (usually longer) thinking, different token counts, different answers for the
  same prompt. Do not compare it with upstream's output token for token.
- The written words appear in `reasoning_content` and in the thinking block of Anthropic-style responses, and they
  occupy context like prompt tokens (the reply's token limit is trimmed if the context is nearly full). A client that
  sends the thinking back will see the opening words in its history.
- The behaviour was tuned on one model family and one quantization range. On other models it is meaningless or harmful;
  leave it off for them.
- The web app used to be the thing that turned it on (a switch that defaulted to on and sent `true` with every
  request). **It no longer does that**: the switch follows the server config and only a choice made in the drawer is sent.

## How to turn it on or off

Put the keys into the run config's `sampling` block (the file setup writes: `strata-<model>.json`, next to the
repo). Absent or `false` = off.

```json
"sampling": {
  "strata_think_prefix": true,
  "strata_think_prefix_tool": true,
  "preserve_thinking": false
}
```

| Key | Values | Meaning |
|---|---|---|
| `strata_think_prefix` | `true` / `false` / `"your words"` | The opening after a user message. `true` = `Let me think about this. The user`. A string is used as is (cut at 200 characters). |
| `strata_think_prefix_tool` | `true` / `false` / `"your words"` | The opening after a tool result. `true` = `Let me`. Turned off together with `strata_think_prefix: false` unless set explicitly by the request. |
| `preserve_thinking` | `true` / `false` | **Separate feature, opt-in.** For clients that send thinking back (Hermes with `reasoning_echo`, llama.cpp's page): `false` keeps the thinking of the steps after the last user message only. Without the key the chat template keeps all of it (upstream behaviour). |

Restart the server after editing. A value of the wrong type (for example a number) stops the server at startup with a message naming
the key; a typo'd key name is printed and ignored. The startup log line `[strata] sampling defaults from the config: ...`
shows what is active.

At setup: `START-HERE.bat --setup --think-prefix on` writes the first two keys (and keeps the others), `--think-prefix off`
removes them. A config without a `sampling` block is stock behaviour, and setup keeps an existing `sampling` block when
it is run again.

Per request (these win over the config):

```json
{"strata_think_prefix": false}                 // no opening for this call
{"strata_think_prefix": "Okay, so the user"}   // your own words
{"strata_think_prefix_tool": true}             // tool opening even if the main switch is off
```

The web app (`/`) has "Start thinking in prose" in the settings drawer. Its default is whatever the server config says
(`GET /settings` reports it as `server_defaults.strata_think_prefix`); a value you set there is stored in the browser
and sent with every request, and "use for other apps too" shares it as the server-wide default.

## Evidence (all on the author's machine; the scripts are in `tuning/style/`)

Short requests, thinking cut at 150 tokens, markers of each style counted over six kinds of request:

| opening written | prose |
|---|---|
| none | 1/18, 1/24 at temperature 1.0 |
| `The user` | 8/18 |
| `Alright, let me` | 17/18 |
| **`Let me think about this. The user`** | **17/18, 24/24 (t 1.0), 23/24 (t 0.6)** |

In Hermes tool loops, with no opening after a tool result 9 of 36 replayed steps were clipped; with `Let me` 3 of 36.
Sending the thinking back (`reasoning_echo`) made the model recite its own earlier thinking verbatim after several
steps, so it is not recommended; `preserve_thinking: false` exists for clients that do it anyway.

Full tables and caveats: [tuning/style/README.md](../tuning/style/README.md). Re-run the checks on your quantization with
`tuning/style/prefix.py` (opening comparison) and `tuning/style/hermes_replay.py` (needs a Hermes database).

## Limits and honest caveats

- Sample sizes are small (6-36 per cell) and measured at 2-3 bit; the full-precision model may not show the clipped style.
- The pelican comparisons are one task family. "Better drawings" is the author's reading of 3 samples per setup.
- It cannot be tested against upstream's server without the model, so the unit tests only cover the plumbing:
  `python -m unittest serve.test_server` (`ThinkPrefix`, including `test_config_toggle`, which checks that an empty
  config writes nothing).
