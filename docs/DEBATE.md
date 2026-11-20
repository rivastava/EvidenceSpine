# Realtime agent debate (evidencespine chat)

`evidencespine chat` runs turn-based debates between independent LLM agents,
coordinated by the EvidenceSpine store. The spine is the message bus: every
message is a room event with a monotonic sequence; each agent reads the room
before replying.

## What it is

- One polling loop per role (a thread inside the coordinator process).
- Each agent is a separate, fresh LLM context: it sees only its role persona,
  the topic, and the rendered room transcript (rolling window plus summary).
  The coordinator never generates debate content.
- Agents read each other live; the full conversation is injected into every
  prompt before a reply is composed.
- The run ends on consensus, quiet, message budget, or a hard deadline (see
  [Termination](#termination-guarantees)).

## LLM backends

| `--llm` | Behavior | Hermetic? |
|---|---|---|
| `opencode` (default) | headless `opencode run --pure` per reply via the `opencode` binary in `--directory` | no (spawns the CLI) |
| `scripted` | deterministic canned replies for tests | yes, never spawns a process |
| `echo` | **dry-run**: placeholder replies, no model call | yes, never spawns a process |

Only `llm == "opencode"` constructs the summarizer and facilitator
subprocess backends. `echo` and `scripted` runs stay hermetic: no binary is
spawned even when `--summarize` or `--facilitate` is requested, so they
cannot hang on a missing/ broken LLM. Each echo reply names its own role
(per-role closure binding), so role identity is not mislabeled.

**A dry run is dry only about the LLM.** Chat messages are still events in the
given store and thread; use a throwaway `--base-dir` if you do not want the
transcript persisted.

## Quick start

```sh
# default backend (spawns headless opencode)
evidencespine chat --topic "Is this feature worth building?" \
  --base-dir .evidencespine

# zero-cost dry run
evidencespine chat --topic "x" --llm echo --base-dir /tmp/chat-demo

# deterministic test run
evidencespine chat --topic "x" --llm scripted --base-dir /tmp/chat-demo
```

Default roles: `developer,engineer,scientist,skeptic`.

## Custom roles and room

```sh
evidencespine chat --topic "<debate topic>" \
  --roles planner,auditor,skeptic \
  --thread-id my_debate --room-id round_1 \
  --base-dir .evidencespine
```

## Long-run session

```sh
evidencespine chat --topic "<topic>" \
  --minutes 45 --facilitate --max-messages 200 \
  --window-size 12 --max-reply-words 40 \
  --quiet-secs 30 --interval 1.5 \
  --base-dir .evidencespine
```

## Controls

| Flag | Default | Meaning |
| --- | --- | --- |
| `--topic` | -- | Debate topic (the kickoff message) |
| `--roles` | developer,engineer,scientist,skeptic | Comma-separated roles |
| `--thread-id` | chat | Spine thread for the room |
| `--room-id` | = thread id | Room namespace within the thread |
| `--max-reply-words` | 60 | Reply cap (prompt brief + last-sentence-boundary truncation; a trailing `AGREE:` clause is preserved) |
| `--max-messages` | 24 | Total message budget |
| `--minutes` | 0 | Max run duration in minutes; 0 = derive a wall-clock backstop |
| `--max-wall-secs` | 0 | Explicit hard wall-clock cap in seconds |
| `--quiet-secs` | 12 | Quiet period that ends the debate (no facilitator left) |
| `--interval` | 1.5 | Poll interval in seconds |
| `--window-size` | 12 | Rolling context: last N messages plus an auto-summary of earlier ones |
| `--no-summarize` | off | Disable the rolling summary |
| `--facilitate` | off | Facilitator injects follow-up questions when the room stalls |
| `--facilitator-budget` | 3 | Max facilitator attempts per run |
| `--llm` | opencode | `opencode`, `scripted`, or `echo` |
| `--directory` | . | Working directory for the headless backend |
| `--timeout` | 180 | Per-call backend timeout (seconds) |

The effective base dir follows `--base-dir` -> `EVIDENCESPINE_BASE_DIR` ->
`.evidencespine`; chat no longer creates a `None/` directory when the flag
is omitted.

## How it works

1. The coordinator publishes a kickoff message (`host`) and starts one thread
   per role.
2. Each agent loop polls the store for new room messages.
3. On a new message from someone else, the agent renders the conversation
   (summary + window), composes a reply through its backend, and the coordinator
   publishes it with a monotonic `chat_seq`.
4. When the room outgrows the window, the coordinator rolls earlier messages
   into a summary event (`chat_summary`), keeping prompts bounded.
5. If the room stalls and `--facilitate` is set, the facilitator pumps one
   sharp question, up to the budget. Attempts are counted **before** the call,
   so a failing facilitator exhausts its budget instead of spinning.
6. The run ends and reports `status` + `stopped_by`.

## Termination guarantees

A run can never hang the CLI:

| `stopped_by` | Trigger |
|---|---|
| `consensus` | every role's latest live message contains `AGREE:` |
| `quiet` | the room is quiet for `--quiet-secs` and no facilitator remains (or none is configured) |
| `max_messages` | message budget reached |
| `duration` | `--minutes` deadline reached |
| `max_wall_secs` | explicit `--max-wall-secs` deadline reached |
| `stalled` | derived wall backstop reached with no agent reply ever published |
| `backend_failure` | consecutive backend errors >= max(3, roles x 2) — a dead LLM cannot loop forever |

Details:

- A wall-clock backstop always exists. With `--minutes 0`, the derived cap is
  `min(180, max(30, quiet_secs * 10))` seconds, so the default configuration
  terminates even if every model call fails.
- The quiet/ facilitator path always sleeps after a pump attempt: a requested
  facilitator with no backend present is treated as quiet, never a busy spin.
- The run result is `{"status": "ok" | "error", "stopped_by", "elapsed_secs",
  "backend_failures", "message_count", "messages"}`. CLI exit code is **3**
  for `"backend_failure"`, `"stalled"` or `"error"` and 0 otherwise, so
  scripts and CI can detect a failed debate.
- The coordinator also seeds its sequence from existing room events, so
  restarting a room does not renumber messages.

## Reading the transcript

```sh
evidencespine brief --base-dir .evidencespine --thread-id my_debate
```

Chat messages are stored as events whose payload carries only `chat` text —
they never become facts and never pollute briefs. Summary events carry
`chat_summary` and `chat_summarized_through` metadata.

## Concurrency and storage notes

- Publishing is locked; each room's `chat_seq` is monotonic and gaps are not
  reused.
- One store per room/thread is the intended shape. Two chat processes sharing
  one room will interleave messages (both seed from existing rows); use
  distinct rooms for parallel debates.
- Model failures are fail-open for the room (the message is skipped and the
  failure counted) but a sustained streak terminates the run as
  `backend_failure` rather than pretending the debate continued.
