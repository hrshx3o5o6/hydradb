"""
Shared, token-metered Gemini client for the v2 benchmark.

Originally targeted OpenAI (gpt-4o-mini + text-embedding-3-small) for a
single-provider, single-tokenizer comparison - but the OPENAI_API_KEY in
this environment has zero credits (confirmed live: 429
insufficient_quota/credit_balance_exhausted on the very first call).
GEMINI_API_KEY works and has quota, so everything - answer generation,
judge, embeddings, and HydraDNA's own causal-discovery LLM calls - runs on
Gemini instead. Still a single provider/tokenizer across all three
strategies, so the per-token comparison stays fair.

Chat token counts come from the API's own usageMetadata field. Gemini's
embedContent/batchEmbedContent responses do NOT return a token count, so
embedding token cost is measured via a separate countTokens call on the
same input text (still a real API-reported count, not a guess).
"""

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple

API_KEY = os.environ["GEMINI_API_KEY"]
# Both gemini-2.5-flash AND gemini-2.5-flash-lite's free tiers are capped at
# 20 generateContent requests/DAY EACH (confirmed live via the API's own
# error body: quotaId "GenerateRequestsPerDayPerProjectPerModel-FreeTier",
# quotaValue 20 for both models) - both exhausted by debugging before the
# smoke test finished. gemma-4-26b-a4b-it is on a separate quota pool
# (confirmed live: works after both gemini models returned 429) - a real,
# capable instruction-tuned model, just a different free-tier bucket.
ANSWER_MODEL = os.environ.get("BENCH_ANSWER_MODEL", "gemma-4-26b-a4b-it")
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "gemma-4-26b-a4b-it")
EMBED_MODEL = os.environ.get("BENCH_EMBED_MODEL", "gemini-embedding-001")
BASE = "https://generativelanguage.googleapis.com/v1beta/models"


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMResult:
    text: str
    usage: Usage
    seconds: float


import threading

_throttle_lock = threading.Lock()
_last_call_at = [0.0]
# Client-side pacing: a burst of embed/discovery/answer calls tripped
# Gemini's per-minute rate limit mid-run even with reactive 429 backoff
# (measured live: 132-chunk embed batch + discovery calls -> 429 that
# outlasted a 6-attempt/~190s backoff). Spacing calls out front-loads the
# wait instead of colliding with the limit and retrying blind.
MIN_CALL_INTERVAL = 2.5  # seconds, ~24 req/min - flash-lite has separate,
# less constrained quota than flash's 20/day (see ANSWER_MODEL comment above)


def throttle() -> None:
    with _throttle_lock:
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call_at[0])
        if wait > 0:
            time.sleep(wait)
        _last_call_at[0] = time.monotonic()


def _post(url: str, payload: dict, retries: int, timeout: int = 150) -> dict:
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        throttle()
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:  # noqa: BLE001 - benchmark retry, not library code
            last_exc = e
            msg = str(e)
            wait = min(60.0, 15.0 * (attempt + 1)) if "429" in msg else min(30.0, 2.0 * (2**attempt))
            time.sleep(wait)
    raise RuntimeError(f"request to {url} failed after {retries} attempts: {last_exc}")


def chat(system: str, user: str, model: str = ANSWER_MODEL, retries: int = 6) -> LLMResult:
    t0 = time.monotonic()
    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": user}]}],
        "generationConfig": {"temperature": 0.0},
    }
    body = _post(f"{BASE}/{model}:generateContent?key={API_KEY}", payload, retries)
    candidates = body.get("candidates", [])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        # gemma-4-26b-a4b-it returns a separate {"thought": true} part
        # carrying its chain-of-thought BEFORE the real answer part
        # (confirmed live) - joining all parts blindly would prepend
        # reasoning text to every answer. Skip thought parts.
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
    um = body.get("usageMetadata", {})
    usage = Usage(
        prompt_tokens=um.get("promptTokenCount", 0),
        # candidatesTokenCount excludes Gemini 2.5's internal "thinking"
        # tokens (thoughtsTokenCount) - those are real, billed generation
        # cost but not part of the visible completion, so track them
        # combined here since they're a genuine per-query cost.
        completion_tokens=um.get("candidatesTokenCount", 0) + um.get("thoughtsTokenCount", 0),
    )
    return LLMResult(text=text, usage=usage, seconds=time.monotonic() - t0)


def count_tokens(text: str, model: str = ANSWER_MODEL, retries: int = 6) -> int:
    body = _post(
        f"{BASE}/{model}:countTokens?key={API_KEY}",
        {"contents": [{"parts": [{"text": text}]}]},
        retries,
        timeout=30,
    )
    return body.get("totalTokens", 0)


_st_model = None


def _local_embedder():
    """Lazily load a local sentence-transformers model.

    Originally this called Gemini's gemini-embedding-001 batchEmbedContents.
    Both external embedding options are exhausted in this environment:
    OpenAI has zero API credits (confirmed live, first call), and Gemini's
    embed_content_free_tier_requests quota (1000/day/project/model) is also
    exhausted (confirmed live via the API's own error body - a fresh
    embedContent call still returns 429/RESOURCE_EXHAUSTED even after
    waiting past the quoted retryDelay, so it's a real daily cap, not a
    transient rate limit). Local embeddings avoid the dependency entirely -
    still genuine dense-vector cosine retrieval, a materially harder
    baseline than BM25's lexical scoring, just computed on-device instead
    of through an API.
    """
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer

        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def embed(texts: List[str], retries: int = 6, batch_size: int = 20) -> Tuple[List[List[float]], Usage]:
    """Embed texts locally (all-MiniLM-L6-v2, see _local_embedder). Token
    cost is measured with this model's OWN tokenizer (not Gemini's) - a
    real count for the model actually doing the work, but not directly
    comparable 1:1 to the Gemini-tokenizer counts used for raw_context/
    hydradna's answer-generation costs or vector_rag's own per-query answer
    cost. Reported separately as one_time_setup_tokens for exactly this
    reason - see run_benchmark.py's report.md caveat."""
    model = _local_embedder()
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()
    tok = model.tokenizer
    total_tokens = sum(len(tok.encode(t, add_special_tokens=True)) for t in texts)
    return vectors, Usage(prompt_tokens=total_tokens, completion_tokens=0)
