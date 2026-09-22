"""
Deterministic, LLM-free causal-discovery building blocks shared by BOTH
entry points into edge discovery:

  - memory.py's CausalMemory._discover_in_window() -- the SDK path, which
    calls its own LLM to propose edges, then filters/gates them with these
    functions before writing.
  - mcp_server.py's propose_edges_window/link tools -- the MCP path, where
    the HOST harness's own model proposes edges (mcp_server.py never calls
    an LLM itself, by design -- see its module docstring). These functions
    let the MCP path apply the SAME candidate-quality filtering and
    edge-quality checks without needing an LLM of its own, so a host using
    MCP doesn't get systematically worse graphs than the SDK.

Nothing in this module makes a network or LLM call. That's the point: it's
the cheap, deterministic layer that runs before (candidate generation) and
after (edge validation) the one place an LLM judgment is actually needed.
"""

import re
from typing import List, Literal, Sequence


# --------------------------------------------------------------- candidates

def temporal_windows(events: Sequence, size: int = 18, overlap: int = 6) -> List[list]:
    """
    Split a time-sorted event list into overlapping windows.

    A single large digest makes an LLM (or a host model doing the same job
    over MCP) under-propose and miss chain links; overlapping windows keep
    each digest small while still letting adjacent-window causal pairs be
    seen together at least once.
    """
    events = list(events)
    if len(events) <= size:
        return [events] if events else []
    windows = []
    step = size - overlap
    for start in range(0, len(events), step):
        win = events[start: start + size]
        if len(win) >= 2:
            windows.append(win)
        if start + size >= len(events):
            break
    return windows


def dedup_by_text(events: Sequence) -> list:
    """
    Drop duplicate-text events (same session, same text): repeated mechanical
    actions (a re-run tool call, a re-logged line) carry no causal signal and
    make a proposer chain near-identical texts together. Keeps first
    occurrence, preserves order.
    """
    seen = set()
    uniq = []
    for e in events:
        key = (e.session_id, e.text)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(e)
    return uniq


def is_action_action_pair(src_event_type: str, dst_event_type: str) -> bool:
    """
    True when both ends are mechanical actions (tool calls, bash steps).
    Agent tool-action chains are noise, not memory -- a real causal signal
    is a decision/request causing an action, or an action causing a result,
    not one action mechanically following another.
    """
    return src_event_type == "action" and dst_event_type == "action"


# ------------------------------------------------------------ edge quality

_STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "did", "do", "does", "why",
    "what", "when", "who", "how", "to", "of", "in", "on", "it", "for",
    "and", "or", "but", "their", "user", "from", "with", "this", "that",
    "then", "so", "as", "at", "by", "be", "been", "has", "have", "had",
}


def _words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if w not in _STOPWORDS and len(w) > 2}


def overlap_score(src_text: str, mechanism: str, dst_text: str) -> float:
    """
    Deterministic proxy for "does this proposed edge have any textual/
    entity grounding, beyond the proposer's say-so and temporal order?"

    Score = fraction of (source-event + mechanism) content words that also
    appear in the target event's text. An LLM confidently proposing A->B
    with a plausible-sounding mechanism but ZERO shared vocabulary with B is
    the classic hallucinated-link failure mode (this is a cheap proxy for
    the "kernel obstruction" pattern -- a global judgment call with no
    grounding check); temporal precedence alone (~0.40 precision in prior
    measurement) is not enough of a filter on its own.

    Not a semantic/embedding check -- deliberately cheap and dependency-free.
    Range [0, 1]; 0 when either side has no scorable content words.
    """
    src_words = _words(src_text) | _words(mechanism)
    dst_words = _words(dst_text)
    if not src_words or not dst_words:
        return 0.0
    shared = src_words & dst_words
    return len(shared) / len(dst_words)


OVERLAP_REJECT_THRESHOLD = 0.05  # near-zero shared vocabulary -> reject


def passes_overlap_check(src_text: str, mechanism: str, dst_text: str) -> bool:
    """Convenience wrapper: does this candidate clear the deterministic bar?"""
    return overlap_score(src_text, mechanism, dst_text) >= OVERLAP_REJECT_THRESHOLD


# --------------------------------------------------------- confidence gate

ConfidenceBand = Literal["reject", "verify", "accept"]

REJECT_BELOW = 0.70
VERIFY_BELOW = 0.85


def route_confidence(confidence: float) -> ConfidenceBand:
    """
    Classify a candidate edge's confidence into a routing decision:
      < 0.70          -> reject outright (tightened from the old blanket 0.8
                         cutoff; the band below replaces a single threshold)
      0.70 - 0.85     -> route to a narrow, separate interventional
                         verification call (see memory.py's
                         _verify_necessity) -- NOT the same global-window
                         call that proposed the edge, since a single
                         judgment call on a complex batch is exactly the
                         failure mode this is trying to avoid
      > 0.85          -> accept directly, no extra LLM call

    Deliberately narrow banding: applying verification to EVERY edge (full
    multi-agent debate) measured 16-25x the token cost for a comparable
    accuracy gain over a single targeted check on the uncertain band -- and
    burning tokens on edges the proposer was already confident about would
    undercut the token-bottleneck bet this whole project is testing.
    """
    if confidence < REJECT_BELOW:
        return "reject"
    if confidence < VERIFY_BELOW:
        return "verify"
    return "accept"
