"""
Reference adapter: Claude-Code-style hook payloads -> CausalMemory events.

Claude Code (and similar coding-agent harnesses) fire hook events with a
`hook_event_name` and a JSON payload. This adapter maps the common ones to
observe() calls with the event_type the discovery pipeline expects, per
HARNESS_CONTRACT.md:

  UserPromptSubmit      -> "fact"        (durable intent)
  PreToolUse/PostToolUse -> "action"     (mechanical step)
  PostToolUse (result)   -> "observation" (raw outcome, stored alongside)
  Stop / agent summary   -> "fact"        (interpreted state)

Unknown hook names fall back to "observation" (the safe default from
HARNESS_CONTRACT.md: never mistag something as "action" when unsure, since
that risks silently suppressing real causal edges).
"""

import time
from typing import Any, Dict, Optional

from ..memory import CausalMemory
from ..schema import Event

_ACTION_HOOKS = {"PreToolUse", "PostToolUse"}
_FACT_HOOKS = {"UserPromptSubmit", "Stop", "SubagentStop"}


def _describe_tool_call(payload: Dict[str, Any]) -> str:
    tool = payload.get("tool_name", "tool")
    tool_input = payload.get("tool_input", {})
    if isinstance(tool_input, dict):
        summary = ", ".join(f"{k}={v!r}" for k, v in list(tool_input.items())[:3])
    else:
        summary = str(tool_input)[:200]
    return f"tool: {tool}({summary})"


def _describe_tool_result(payload: Dict[str, Any]) -> str:
    tool = payload.get("tool_name", "tool")
    result = payload.get("tool_response") or payload.get("tool_output") or ""
    text = str(result)
    if len(text) > 300:
        text = text[:300] + "..."
    return f"result: {tool} -> {text}"


def ingest_hook_event(
    memory: CausalMemory,
    hook_event_name: str,
    payload: Dict[str, Any],
    session_id: str,
    timestamp: Optional[int] = None,
) -> Optional[Event]:
    """
    Map one Claude-Code-style hook event to a single observe() call.

    Returns None for hook events that carry no causally-relevant content
    (e.g. a bare SessionStart with no payload text).
    """
    ts = timestamp if timestamp is not None else int(time.time())

    # observe() hardcodes event_type="observation" and can't carry
    # "fact"/"action" — use add_event() directly so this table actually
    # takes effect (see event_type_for_hook()).
    if hook_event_name == "UserPromptSubmit":
        text = payload.get("prompt", "").strip()
        if not text:
            return None
        return memory.add_event(text=f"user: {text}", session_id=session_id,
                                 event_type="fact", topic="request", timestamp=ts)

    if hook_event_name == "PreToolUse":
        return memory.add_event(text=_describe_tool_call(payload),
                                 session_id=session_id, event_type="action",
                                 topic="tool_call", timestamp=ts)

    if hook_event_name == "PostToolUse":
        # PostToolUse carries both the call and its result; store the result
        # as an observation (the raw outcome), distinct from the "action"
        # PreToolUse already logged for harnesses that fire both hooks. A
        # harness that only fires PostToolUse should treat this as its sole
        # record of the call and may want "action" instead — see
        # HARNESS_CONTRACT.md.
        return memory.add_event(text=_describe_tool_result(payload),
                                 session_id=session_id, event_type="observation",
                                 topic="tool_result", timestamp=ts)

    if hook_event_name in ("Stop", "SubagentStop"):
        summary = payload.get("summary") or payload.get("message") or ""
        if not summary:
            return None
        return memory.add_event(text=f"decision: {summary}", session_id=session_id,
                                 event_type="fact", topic="decision", timestamp=ts)

    # Unknown hook: fall back to a plain observation rather than guessing
    # "action" and risking silent action-action suppression.
    text = payload.get("message") or payload.get("summary") or str(payload)[:200]
    if not text:
        return None
    return memory.add_event(text=text, session_id=session_id,
                             event_type="observation", topic=hook_event_name,
                             timestamp=ts)


def event_type_for_hook(hook_event_name: str) -> str:
    """Expose the event_type mapping table for callers that build Events
    directly instead of going through ingest_hook_event()."""
    if hook_event_name in _ACTION_HOOKS:
        return "action"
    if hook_event_name in _FACT_HOOKS:
        return "fact"
    return "observation"
