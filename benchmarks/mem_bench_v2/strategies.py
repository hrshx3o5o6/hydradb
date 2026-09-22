"""
Three retrieval strategies compared on the same LoCoMo QA, same answer model,
tokens metered per query via the API's own usage field.

  raw_context - dump the full conversation transcript into context every query
  vector_rag  - real Gemini-embedding cosine-similarity top-k retrieval
  hydradna    - ingest into HydraDNA's causal graph, retrieve the causal
                subgraph relevant to the question

Each strategy exposes a common shape: a one-time `setup(conversation)` that
returns (state, one_time_usage), and an `answer(state, question)` that
returns (answer_text, per_query_usage, seconds).
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_client import Usage, chat, embed, throttle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from causal_memory.memory import CausalMemory  # noqa: E402
from causal_memory.llm import LLM  # noqa: E402
from causal_memory.adapters.conversation_adapter import (  # noqa: E402
    ingest_conversation,
    conversation_to_transcript,
    session_chunks,
)

ANSWER_SYSTEM = (
    "Answer the question using ONLY the provided context. Be concise - a "
    "short phrase or sentence, not a paragraph. If the context doesn't say, "
    "answer 'unknown'."
)


# --------------------------------------------------------------------- raw_context

def raw_context_setup(conversation: Dict[str, Any]) -> Tuple[str, Usage]:
    transcript = conversation_to_transcript(conversation)
    return transcript, Usage()  # no one-time cost - nothing precomputed


def raw_context_answer(transcript: str, question: str):
    user = f"Conversation transcript:\n{transcript}\n\nQuestion: {question}"
    r = chat(ANSWER_SYSTEM, user)
    return r.text, r.usage, r.seconds


# ----------------------------------------------------------------------- vector_rag

@dataclass
class VectorIndex:
    chunks: List[str]
    vectors: List[List[float]]


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def vector_rag_setup(conversation: Dict[str, Any]) -> Tuple[VectorIndex, Usage]:
    # Session-level chunks (not per-turn): per-turn chunking on a 400+ turn
    # conversation means 400+ embed calls, which trips Gemini's free-tier
    # rate limit (measured live: 429 mid-run). Session-level also matches
    # the granularity hydradna_setup uses below, keeping the two retrieval
    # strategies' chunking apples-to-apples.
    chunks = [
        f"[session {c['session_idx']}] {c['text']}" for c in session_chunks(conversation)
    ]
    vectors, usage = embed(chunks)
    return VectorIndex(chunks=chunks, vectors=vectors), usage


def vector_rag_answer(index: VectorIndex, question: str, top_k: int = 3):
    (qvec,), qusage = embed([question])
    scored = sorted(
        range(len(index.chunks)),
        key=lambda i: _cosine(qvec, index.vectors[i]),
        reverse=True,
    )[:top_k]
    context = "\n".join(index.chunks[i] for i in sorted(scored))
    user = f"Retrieved context:\n{context}\n\nQuestion: {question}"
    r = chat(ANSWER_SYSTEM, user)
    # Per-query cost = the embedding call for the question + the answer
    # call's context/completion tokens. Question-embedding cost is small but
    # real and recurring (every query re-embeds the question) - counted here,
    # not in the one-time setup cost.
    combined = Usage(
        prompt_tokens=r.usage.prompt_tokens + qusage.prompt_tokens,
        completion_tokens=r.usage.completion_tokens,
    )
    return r.text, combined, r.seconds


# ------------------------------------------------------------------------ hydradna

class TrackedLLM(LLM):
    """Wraps causal_memory.llm.LLM to capture usage from every call instead
    of discarding it. Uses Gemini (the base class's default when
    GEMINI_API_KEY is set) - OpenAI has zero API credits in this
    environment (confirmed live: 429 insufficient_quota), so Gemini is the
    one working provider and is what all three strategies use here,
    keeping the token comparison single-provider/single-tokenizer."""

    def __init__(self):
        super().__init__()
        self.calls: List[Usage] = []
        # Both gemini-2.5-flash and gemini-2.5-flash-lite are capped at 20
        # generateContent requests/DAY each on this key's free tier
        # (confirmed live, both exhausted) - gemma-4-26b-a4b-it is a
        # separate, working quota bucket (confirmed live).
        self.model = self.model or "gemma-4-26b-a4b-it"

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        # Override the base class entirely: LLM.complete() retries Gemini
        # only twice with a flat 2s sleep, then falls through to OpenAI -
        # which has zero credits in this environment (confirmed live: 429
        # insufficient_quota) and crashes uncaught. Real Gemini free-tier
        # rate limiting was hit live during this benchmark (429 mid-run on
        # a discovery call), so this needs real backoff, not a fallback to
        # a dead provider.
        import time

        last_exc = None
        for attempt in range(8):
            try:
                return self._gemini(system, user, json_mode)
            except Exception as e:  # noqa: BLE001 - benchmark retry
                last_exc = e
                wait = min(60.0, 15.0 * (attempt + 1)) if "429" in str(e) else min(20.0, 2.0 * (2**attempt))
                time.sleep(wait)
        raise RuntimeError(f"TrackedLLM.complete failed after 8 attempts: {last_exc}")

    def _gemini(self, system: str, user: str, json_mode: bool) -> str:
        import json as _json
        import urllib.request

        throttle()  # shared client-side pacing - see llm_client.throttle()
        model = self.model or "gemma-4-26b-a4b-it"  # __init__ always sets self.model; this is a defensive fallback only
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self.gemini_key}"
        )
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
        }
        if json_mode:
            payload["generationConfig"] = {
                "responseMimeType": "application/json",
                "temperature": 0.2,
            }
        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # gemma-4-26b-a4b-it burns a lot of "thinking" tokens even on
        # simple prompts (measured live: 227 thoughtsTokenCount on a
        # 2-token trivial prompt) - discovery prompts are much longer, and
        # 60s timed out live. 150s gives real headroom.
        with urllib.request.urlopen(req, timeout=150) as resp:
            body = _json.loads(resp.read().decode())
        um = body.get("usageMetadata", {})
        self.calls.append(
            Usage(
                prompt_tokens=um.get("promptTokenCount", 0),
                completion_tokens=um.get("candidatesTokenCount", 0)
                + um.get("thoughtsTokenCount", 0),
            )
        )
        parts = body["candidates"][0]["content"]["parts"]
        # gemma-4-26b-a4b-it emits a {"thought": true} reasoning part before
        # the real answer (confirmed live) - critical here since this feeds
        # extract_json() for causal-discovery edges; unfiltered reasoning
        # text would break JSON parsing, not just pollute prose.
        return "".join(p.get("text", "") for p in parts if not p.get("thought"))

    def total_usage(self) -> Usage:
        return Usage(
            prompt_tokens=sum(c.prompt_tokens for c in self.calls),
            completion_tokens=sum(c.completion_tokens for c in self.calls),
        )


@dataclass
class HydraDNAState:
    memory: CausalMemory
    conv_prefix: str
    llm: TrackedLLM


def hydradna_setup(
    conversation: Dict[str, Any], conv_prefix: str, url: str, token: str, admin_url: str
) -> Tuple[HydraDNAState, Usage]:
    tracked = TrackedLLM()
    mem = CausalMemory(url=url, auth_token=token, admin_url=admin_url, llm=tracked)
    mem.reset()
    # granularity="session" (not "turn"): see vector_rag_setup's comment -
    # per-turn ingestion means ~35 discovery-window LLM calls per
    # conversation, which trips Gemini's free-tier rate limit.
    events = ingest_conversation(
        mem, conversation, session_prefix=conv_prefix, autocausal=False,
        granularity="session",
    )
    n_edges = mem.extract_causality(window_events=len(events))
    usage = tracked.total_usage()
    return HydraDNAState(memory=mem, conv_prefix=conv_prefix, llm=tracked), usage


def hydradna_answer(state: HydraDNAState, question: str):
    """Retrieve the causally-relevant subgraph (not the whole graph) and
    hand it to the answer model - the actual thing being tested."""
    import time

    t0 = time.monotonic()
    target_id = state.memory._match_event_by_keywords(question)
    context_lines: List[str] = []
    if target_id is not None:
        base = state.memory.get_event(int(target_id))
        if base:
            context_lines.append(f"event: {base.text}")
        causes = state.memory.find_causes(int(target_id), max_hops=4)
        effects = state.memory.find_all_effects(int(target_id), max_hops=2)
        for p in causes[:3]:
            for i, ev in enumerate(p.events):
                line = f"  <- {ev.text}"
                if i < len(p.relations):
                    r = p.relations[i]
                    if r.mechanism:
                        line += f" ({r.mechanism})"
                context_lines.append(line)
        for p in effects[:2]:
            for ev in p.events[1:]:
                context_lines.append(f"  -> {ev.text}")
    if not context_lines:
        # Fall back to keyword search over stored events (still a small
        # retrieved slice, not the full graph).
        hits = state.memory.search(question, limit=6)
        context_lines = [f"event: {h.text}" for h in hits]

    context = "\n".join(context_lines) if context_lines else "(no relevant memory found)"
    user = f"Retrieved causal context:\n{context}\n\nQuestion: {question}"
    r = chat(ANSWER_SYSTEM, user)
    return r.text, r.usage, time.monotonic() - t0
