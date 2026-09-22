"""
Reference adapter: plain multi-turn dialogue -> CausalMemory events.

Matches the LoCoMo conversation shape (speaker_a/speaker_b, session_N +
session_N_date_time, each turn {speaker, dia_id, text}), but any dialogue
with (speaker, text, session_id) per turn fits the same call shape.

Per HARNESS_CONTRACT.md: plain chat turns are tagged event_type="observation"
(the CausalMemory.observe() default) since nothing has been distilled from
the raw utterance yet.
"""

import re
import time
from typing import Any, Dict, List, Optional

from ..memory import CausalMemory
from ..schema import Event

_DATE_RE = re.compile(
    r"(\d{1,2}):(\d{2})\s*(am|pm)\s*on\s*(\d{1,2})\s+(\w+),?\s*(\d{4})",
    re.IGNORECASE,
)
_MONTHS = {
    m: i
    for i, m in enumerate(
        [
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ],
        start=1,
    )
}


def _sanitize(text: str) -> str:
    """Adapter-layer workaround for a real bug in schema.py's Event.props():
    it interpolates `text` into an unescaped double-quoted Cypher string
    literal, so any `"` in the source text breaks the query (confirmed live
    against LoCoMo conv-26 session 6, turn D6:10: 'I loved reading
    "Charlotte\'s Web" as a kid...' -> HTTP 400). The real fix belongs in
    schema.py (out of scope for this adapter); this is a narrow, lossy
    stand-in so benchmark ingestion doesn't silently drop turns.
    """
    return text.replace('"', "'")


def parse_session_datetime(text: str) -> int:
    """Best-effort parse of LoCoMo's 'H:MM am/pm on D Month, YYYY' -> unix ts.

    Falls back to time.time() (still monotonic-enough for ordering within a
    session) if the format doesn't match.
    """
    m = _DATE_RE.search(text)
    if not m:
        return int(time.time())
    hour, minute, ampm, day, month_name, year = m.groups()
    hour = int(hour) % 12
    if ampm.lower() == "pm":
        hour += 12
    month = _MONTHS.get(month_name.lower())
    if not month:
        return int(time.time())
    import calendar

    try:
        return calendar.timegm(
            (int(year), month, int(day), hour, int(minute), 0, 0, 0, 0)
        )
    except ValueError:
        return int(time.time())


def ingest_conversation(
    memory: CausalMemory,
    conversation: Dict[str, Any],
    session_prefix: str,
    autocausal: bool = False,
) -> List[Event]:
    """
    Ingest a LoCoMo-shaped conversation dict into CausalMemory.

    `conversation` has speaker_a/speaker_b plus session_N/session_N_date_time
    pairs for N = 1, 2, 3, .... Each turn is stored as one observation event;
    session_id is f"{session_prefix}-s{N}" so sessions stay distinguishable
    inside one CausalMemory graph. Timestamps are derived from the session
    date/time plus the turn's position, so ordering within and across
    sessions round-trips through the graph's real timestamp field (needed
    for extract_causality's cause-must-precede-effect check).

    autocausal=False by default: benchmark callers run extract_causality()
    themselves (batched across the whole conversation) rather than per-turn,
    which would re-run discovery on an ever-growing window.
    """
    events: List[Event] = []
    session_idx = 1
    while f"session_{session_idx}" in conversation:
        turns = conversation[f"session_{session_idx}"]
        date_key = f"session_{session_idx}_date_time"
        base_ts = parse_session_datetime(conversation.get(date_key, ""))
        session_id = f"{session_prefix}-s{session_idx}"
        for turn_offset, turn in enumerate(turns):
            ts = base_ts + turn_offset  # preserve within-session order
            ev = memory.observe(
                text=_sanitize(f"{turn['speaker']}: {turn['text']}"),
                session_id=session_id,
                topic=turn.get("dia_id"),
                timestamp=ts,
            )
            events.append(ev)
        session_idx += 1

    if autocausal and len(events) >= 2:
        memory.extract_causality(window_events=len(events))

    return events


def conversation_to_transcript(conversation: Dict[str, Any]) -> str:
    """Flatten a LoCoMo-shaped conversation into one plain-text transcript
    (session headers + speaker turns), for baselines that want raw text
    instead of graph ingestion (e.g. a raw-context-stuffing comparator)."""
    lines: List[str] = []
    session_idx = 1
    while f"session_{session_idx}" in conversation:
        turns = conversation[f"session_{session_idx}"]
        date_key = f"session_{session_idx}_date_time"
        lines.append(f"--- Session {session_idx} ({conversation.get(date_key, '')}) ---")
        for turn in turns:
            lines.append(f"{turn['speaker']}: {turn['text']}")
        session_idx += 1
    return "\n".join(lines)
