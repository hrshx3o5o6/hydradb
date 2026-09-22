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
    "Charlotte\'s Web" as a kid...' -> HTTP 400), and a raw newline breaks it
    the same way (confirmed live: session-level chunks, which join turns
    with '\\n', 400 on the very first ingest call). The real fix belongs in
    schema.py (out of scope for this adapter); this is a narrow, lossy
    stand-in so benchmark ingestion doesn't silently drop content.
    """
    return text.replace('"', "'").replace("\n", " | ").replace("\\", "/")


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


# The live query engine hard-caps total query length: binary-searched live
# against this exact MERGE-with-event-props query shape, a plain-ASCII text
# literal of 864 chars succeeds and 865 fails (HTTP 400, OpenCypher parse
# error - the lexer appears to truncate its input buffer mid-literal rather
# than reject the query cleanly, so a too-long string produces a confusing
# "invalid input '<mid-word-letter>'" error, not a length error). 600 leaves
# real margin for the surrounding query template + session/topic fields.
MAX_EVENT_TEXT_CHARS = 650  # _sanitize()'s '\n'->' | ' expansion eats into
# the 864-char measured boundary, so this leaves real margin below it.


def session_chunks(conversation: Dict[str, Any], max_chars: int = MAX_EVENT_TEXT_CHARS) -> List[Dict[str, Any]]:
    """Chunks of up to `max_chars` of dialogue text per session (a session
    that fits in one chunk gets one; a longer session splits into several,
    always on turn boundaries): {session_idx, date_time, text}. Coarser than
    per-turn - used by both the hydradna and vector_rag benchmark strategies
    so chunking granularity is apples-to-apples between them, and because
    per-turn chunking (419 turns -> 419 embed calls / 419 graph events ->
    ~35 discovery LLM calls per conversation) trips Gemini free-tier rate
    limits (measured live: 429 Too Many Requests mid-run)."""
    chunks = []
    session_idx = 1
    while f"session_{session_idx}" in conversation:
        turns = conversation[f"session_{session_idx}"]
        date_key = f"session_{session_idx}_date_time"
        date_time = conversation.get(date_key, "")
        cur_lines: List[str] = []
        cur_len = 0
        for t in turns:
            line = f"{t['speaker']}: {t['text']}"
            if cur_lines and cur_len + len(line) + 1 > max_chars:
                chunks.append({
                    "session_idx": session_idx, "date_time": date_time,
                    "text": "\n".join(cur_lines),
                })
                cur_lines, cur_len = [], 0
            cur_lines.append(line)
            cur_len += len(line) + 1
        if cur_lines:
            chunks.append({
                "session_idx": session_idx, "date_time": date_time,
                "text": "\n".join(cur_lines),
            })
        session_idx += 1
    return chunks


def ingest_conversation(
    memory: CausalMemory,
    conversation: Dict[str, Any],
    session_prefix: str,
    autocausal: bool = False,
    granularity: str = "turn",
) -> List[Event]:
    """
    Ingest a LoCoMo-shaped conversation dict into CausalMemory.

    `conversation` has speaker_a/speaker_b plus session_N/session_N_date_time
    pairs for N = 1, 2, 3, .... session_id is f"{session_prefix}-s{N}" so
    sessions stay distinguishable inside one CausalMemory graph. Timestamps
    are derived from the session date/time plus position, so ordering round-
    trips through the graph's real timestamp field (needed for
    extract_causality's cause-must-precede-effect check).

    granularity="turn" (default): one event per dialogue turn.
    granularity="session": one event per whole session (see session_chunks) -
    far fewer events/discovery-LLM-calls, needed to stay under Gemini's
    free-tier rate limit on longer conversations (see session_chunks).

    autocausal=False by default: benchmark callers run extract_causality()
    themselves (batched across the whole conversation) rather than per-turn,
    which would re-run discovery on an ever-growing window.
    """
    events: List[Event] = []

    if granularity == "session":
        for chunk_offset, chunk in enumerate(session_chunks(conversation)):
            # chunk_offset (not session_idx alone) keeps timestamps strictly
            # increasing across multiple chunks of the same long session -
            # needed for extract_causality's cause-must-precede-effect check.
            ts = parse_session_datetime(chunk["date_time"]) + chunk_offset
            ev = memory.observe(
                text=_sanitize(chunk["text"]),
                session_id=f"{session_prefix}-s{chunk['session_idx']}",
                topic=f"session_{chunk['session_idx']}",
                timestamp=ts,
            )
            events.append(ev)
    else:
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
