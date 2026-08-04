# Realtime Agent Debate (evidencespine chat)

`evidencespine chat` runs live, turn-based debates between independent LLM
agents, coordinated entirely by the EvidenceSpine store. The spine is the
message bus: every message is a room event with a monotonic sequence; every
agent reads the room before replying.

## What it is

- One polling loop per role (a thread inside the coordinator process).
- Each agent is a **separate, fresh LLM context** (headless `opencode run
  --pure`) — it sees only its role persona, the topic, and the rendered
  room transcript. The coordinator never generates debate content.
- Agents read each other live: the full conversation (windowed + summarized)
  is injected into every prompt before a reply is composed.
- The debate ends when every role has said `AGREE:` (consensus), the room
  goes quiet, a message budget is hit, or a duration cap expires.

## Tested on opencode

This feature is strictly tested on opencode:

- The default LLM backend is headless `opencode run --pure` (no plugin/MCP
  spawns per turn); verified end-to-end against opencode 1.18.x.
- The opencode harness delivery layer (session-start pre-warm, compaction,
  session-stop hooks) is exercised by the same test suite.
- Live debates have been run and verified inside opencode sessions with the
  default roles (developer, engineer, scientist, skeptic), including a
  user-verified TUI fix for the first-message render glitch.
- The MCP server, prompts, and tools are verified in-session inside opencode
  (tools, prompt renders, placeholder-arg hardening, concurrent render
  safety).

## Quick start

```bash
evidencespine chat --topic "Is this feature worth building?" \
  --base-dir .evidencespine
```

Default roles: `developer,engineer,scientist,skeptic`.

## Custom roles and room

```bash
evidencespine chat --topic "<debate topic>" \
  --roles planner,auditor,skeptic \
  --thread-id my_debate --room-id round_1 \
  --base-dir .evidencespine
```

## Long-run session

```bash
evidencespine chat --topic "<topic>" \
  --minutes 45 --facilitate --max-messages 200 \
  --window-size 12 --max-reply-words 40 \
  --quiet-secs 30 --interval 1.5 \
  --base-dir .evidencespine
```

## Controls

| Flag | Default | Meaning |
| --- | --- | --- |
| `--topic` | — | Debate topic (the kickoff message) |
| `--roles` | developer,engineer,scientist,skeptic | Comma-separated roles |
| `--thread-id` | chat | Spine thread for the room |
| `--room-id` | = thread-id | Room namespace within the thread |
| `--max-reply-words` | 60 | Hard reply cap: the prompt briefs the model and the coordinator truncates at the last sentence boundary; a trailing `AGREE:` clause is preserved |
| `--max-messages` | 24 | Total message budget |
| `--minutes` | 0 | Max run duration in minutes; `0` = no time cap (ends on consensus/quiet/budget) |
| `--quiet-secs` | 12 | Quiet period that ends the debate (when no facilitator remains) |
| `--interval` | 1.5 | Poll interval in seconds |
| `--window-size` | 12 | Rolling context: agents see the last N messages plus an auto-summary of earlier ones |
| `--no-summarize` | off | Disable the rolling summary |
| `--facilitate` | off | A facilitator agent injects follow-up questions when the room stalls |
| `--facilitator-budget` | 3 | Max facilitator pumps per run |
| `--llm` | opencode | `opencode` (headless opencode run), `scripted` (deterministic, for tests), or `echo` (dry-run) |
| `--directory` | . | Working directory for the headless backend |

## How it works

1. The coordinator publishes a kickoff message (`host`) and starts one thread
   per role.
2. Each agent loop polls the store for new room messages.
3. On a new message from someone else, the agent renders the conversation
   (summary + window), composes a reply through its LLM backend, and the
   coordinator publishes it back with a monotonic `chat_seq`.
4. When the room outgrows the window, the coordinator rolls earlier messages
   into a summary event (`chat_summary`), keeping prompts bounded.
5. If the room stalls and `--facilitate` is set, the facilitator pumps one
   sharp question (up to the budget).
6. The run stops with a `stopped_by` reason: `consensus`, `quiet`,
   `max_messages`, or `duration`.

## Reading the transcript

```bash
evidencespine brief --base-dir .evidencespine --thread-id my_debate
```

Chat messages are stored as events whose payload carries only `chat` text —
they never become facts and never pollute briefs.

## Dry-run without LLM cost

```bash
evidencespine chat --topic "x" --llm scripted    # deterministic
evidencespine chat --topic "x" --llm echo        # placeholder replies
```
