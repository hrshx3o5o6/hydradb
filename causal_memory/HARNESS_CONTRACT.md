# Harness Event-Tagging Contract

HydraDNA's causal-discovery pass (`CausalMemory._discover_in_window`) makes
one silent, load-bearing assumption: `event_type` is accurate. Two rules in
that function depend on it directly:

1. **Action-action suppression** — an edge is never proposed between two
   events both tagged `event_type="action"` (mechanical tool-call ladders
   are not causal chains). If a harness tags a genuine decision or user
   request as `"action"`, that edge silently never gets proposed, and
   nobody is told why.
2. **Confidence-band routing** (see `discovery_core.py`) treats `"action"`
   pairs more skeptically than `"fact"`/`"observation"` pairs when routing
   borderline-confidence candidates to extra verification.

There is no validation catching a mistagged event — it just quietly
produces a worse graph. This document is the contract a harness adapter
must follow so discovery quality doesn't depend on each integrator
guessing.

## The three `event_type` values

| value | means | discovery treats it as |
|---|---|---|
| `"fact"` | A stated truth about the world/user that can be superseded later (`OVERWRITES`) | Eligible source or target for any edge type, including `OVERWRITES` |
| `"action"` | A mechanical step the agent itself performed (ran a tool, edited a file, called an API) | Never linked to another `"action"` — must have a `"fact"`/`"observation"` on at least one end of the edge |
| `"observation"` | A raw, unprocessed thing that happened (a message arrived, a value was read) — the default from `observe()` | Treated like `"fact"` for suppression purposes, but not assumed durable/current the way a `"fact"` is |

## Coding-agent harness (e.g. Claude Code style hooks/tool events)

| harness event | `event_type` | why |
|---|---|---|
| User sends a request/instruction | `"fact"` | Durable intent — the thing later actions are *caused by* |
| Agent states a decision/plan ("I'll refactor X because Y") | `"fact"` | A decision is a fact about agent state, not a mechanical step — it's exactly the kind of node that should CAUSE the actions that follow |
| Tool call: file edit, bash command, test run | `"action"` | Mechanical execution step |
| Tool result: test passed/failed, command output | `"observation"` | Raw outcome, not yet judged |
| Agent's own summary/conclusion drawn from a result | `"fact"` | Interpreted state, eligible to be a cause of subsequent actions |
| Error/exception surfaced | `"observation"` | Same as tool result |

Consecutive `bash: run tests` → `bash: git commit` are both `"action"` —
suppressed by design; the correct causal edge is `"decision to ship" →
"run tests"` and `"tests passed" → "git commit"`, both of which have a
non-action endpoint.

## Conversational harness (LoCoMo-style multi-turn dialogue)

| harness event | `event_type` | why |
|---|---|---|
| Each dialogue turn (either speaker) | `"observation"` | Raw utterance — nothing has been distilled from it yet |
| A stated preference/fact a speaker asserts about themselves | `"fact"` | Should participate in `OVERWRITES` if a later turn contradicts it |
| A tool/function the conversational agent invokes on the user's behalf | `"action"` | Same mechanical-step rule as the coding harness |

Plain chat turns default to `"observation"` — `CausalMemory.observe()`
already defaults there, so a conversational adapter usually does not need
to override `event_type` per turn, only per distilled fact.

## What breaks if you get this wrong

Tag every event `"observation"` (the safe default) and you lose nothing
from suppression (nothing is `"action"`) but you also lose the durable/
supersede-tracking benefit of `"fact"` — `find_current_fact` and
`OVERWRITES` become no-ops for that harness. Tag everything `"action"`
(common mistake when adapting a tool-heavy harness naively) and discovery
silently proposes almost nothing, because nearly every pair gets
suppressed — this looks like "the LLM isn't finding causal links" when
it's actually the deterministic filter doing exactly what it's told.

## Adapters

Two reference adapters ship in `causal_memory/adapters/`:

- `conversation_adapter.py` — plain multi-turn dialogue (LoCoMo shape) →
  `observe()` calls, one per turn, `event_type="observation"`.
- `claude_code_adapter.py` — Claude-Code-style hook payloads (user
  prompt, tool pre/post-call, tool result) → `observe()` calls following
  the coding-agent table above.

Both are intentionally small — the contract is the part meant to be
copied into a new adapter, not the code.
