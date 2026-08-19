"""HydraDNA adapter for mem-bench.

HydraDNA is the causal memory layer built on HydraDB. Events are stored as
(Session)-[:HAS_EVENT]->(Event) vertices; causal discovery proposes CAUSES
edges between temporally ordered observations (see causal_memory).

Retrieval strategy exposed to the harness:
  1. lexical substring search over stored events (HydraDNA ``search``)
  2. word-overlap scoring (the keyword resolution HydraDNA's ``why()`` uses
     as its fallback)
  3. causal expansion: events reached via ``find_causes``/``find_effects`` on
     any hit are surfaced as neighbors (structural retrieval)

document_id round-trips through the adapter's text->document map because the
current Event schema does not carry a document_id property and the query
engine rejects property filters beyond equality.

Env vars:
    HYDRADB_URL          base URL of the HydraDB HTTP API (default
                         http://localhost:8443)
    HYDRADB_AUTH_TOKEN   bearer token (default local-dev token)
    HYDRA_CAUSAL         "1" to run causal discovery during ingest
                         (default: enabled)
"""

from __future__ import annotations

import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Sequence

from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.types import IngestItem, RecallQuery, RecallResult

logger = logging.getLogger(__name__)

STOPWORDS = {
    "the", "a", "an", "is", "was", "were", "did", "do", "does", "why", "what",
    "when", "who", "how", "to", "of", "in", "on", "it", "for", "and", "or",
    "but", "their", "user", "from", "with", "that", "this", "which", "about",
    "you", "they", "he", "she", "we", "i",
}

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")


def _safe(text: str) -> str:
    """Make a string safe to inline into a Cypher double-quoted literal."""
    text = _CONTROL_RE.sub(" ", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _iso_to_unix(iso: str | None) -> int:
    if not iso:
        return int(time.time())
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except ValueError:
        return int(time.time())


# The HydraDB query engine rejects statements whose string literals exceed a
# hard ~880-BYTE (UTF-8) cap; the parser errors mid-token once the literal is
# too large. Byte budget 700 keeps unicode-heavy text (multi-byte glyphs like
# math-italic/√/⎡ are 3-4 bytes) comfortably under the cap. This mirrors how a
# real memory layer chunks long transcripts anyway.
CHUNK_SIZE = 700


def _chunks(text: str, size: int = CHUNK_SIZE) -> list[str]:
    chunks: list[str] = []
    cur = ""
    cur_bytes = 0
    for ch in text:
        b = len(ch.encode("utf-8"))
        if cur and cur_bytes + b > size:
            chunks.append(cur)
            cur = ""
            cur_bytes = 0
        cur += ch
        cur_bytes += b
    if cur:
        chunks.append(cur)
    return chunks


# Cap the event set fed to causal discovery so per-sample latency stays
# bounded on long sessions (each LLM window call is seconds). Evenly sample
# from the full ordered list to preserve temporal spread.
MAX_DISCOVERY_EVENTS = 40


def _cap_events(stored: list[tuple[str, int, str]], cap: int = MAX_DISCOVERY_EVENTS) -> list[tuple[str, int, str]]:
    if len(stored) <= cap:
        return stored
    step = (len(stored) - 1) / (cap - 1)
    return [stored[round(i * step)] for i in range(cap)]


class HydraDNAAdapter(BaseAdapter):
    """mem-bench adapter backed by the HydraDNA causal memory layer."""

    def __init__(self, **kwargs: Any) -> None:
        self._url = kwargs.pop("url", None) or os.environ.get(
            "HYDRADB_URL", "http://localhost:8443"
        )
        self._auth_token = kwargs.pop("auth_token", None) or os.environ.get(
            "HYDRADB_AUTH_TOKEN", "local-dev-auth-token-32-characters-long"
        )
        self._causal = bool(int(os.environ.get("HYDRA_CAUSAL", "1")))
        self._llm_model = kwargs.pop("llm_model", None)
        self._memory: Any = None
        self._docs: dict[str, dict[str, tuple[str, str]]] = {}
        # Conversation caching: LoCoMo re-ingests the SAME ~20-session
        # conversation for every QA sample (1986 samples, 196k chunk writes
        # naive). When enabled, samples whose document_ids share a "conv-N_"
        # prefix store their events ONCE under a stable conversation namespace
        # and reuse it across samples, cutting writes ~158x and keeping the
        # node graph tiny (no delete churn -> no engine degradation).
        self._cache_conv = bool(kwargs.pop("cache_conversations", False))
        self._conv_ns: str | None = None  # storage namespace of live conversation
        self._ns_to_conv: dict[str, str] = {}  # sample ns -> conv storage ns
        # storage-ns -> document_id -> full original text. Kept so recall can
        # hand the judge WHOLE sessions (not 700-byte slices) as context:
        # retrieval metrics match on document_id, but qa_accuracy depends on
        # the context text the LLM actually reads. Chunk slices were losing
        # sentences and surrounding context, so hydradna retrieved the right
        # session but answered worse than BM25 (which returns whole sessions).
        self._sessions: dict[str, dict[str, str]] = {}

    # ------------------------------------------------------------------ lazy

    def _get_memory(self) -> Any:
        if self._memory is None:
            from causal_memory import CausalMemory

            self._memory = CausalMemory(
                url=self._url,
                auth_token=self._auth_token,
                llm=self._make_llm() if self._causal else None,
            )
        return self._memory

    def _make_llm(self) -> Any:
        from causal_memory.llm import LLM

        try:
            return LLM(model=self._llm_model) if self._llm_model else LLM()
        except Exception:
            logger.warning("hydradna: no LLM credentials; causal discovery disabled")
            return None

    def _conv_storage_ns(self, items: Sequence[IngestItem]) -> str | None:
        """Derive a stable conversation storage namespace from document_ids.

        LoCoMo document ids look like ``conv-26_session_1``; all QA samples of
        one conversation share the same prefix, so we can store the
        conversation once and reuse it across samples.
        """
        if not self._cache_conv:
            return None
        for item in items:
            d = item.document_id or ""
            m = re.match(r"^([A-Za-z0-9_-]+)_session_\d+$", d)
            if m:
                return "lc_" + m.group(1)
        return None

    # --------------------------------------------------------------- interface

    def ingest(self, items: Sequence[IngestItem], *, namespace: str = "default") -> None:
        conv = self._conv_storage_ns(items)
        if conv:
            self._ns_to_conv[namespace] = conv
            if self._conv_ns == conv:
                return  # conversation already stored; reuse it
            if self._conv_ns is not None:
                try:
                    self._delete_namespace(self._get_memory(), self._conv_ns)
                except Exception:
                    logger.warning("hydradna: dropping cached conv %s failed", self._conv_ns, exc_info=True)
                self._docs.pop(self._conv_ns, None)
                self._sessions.pop(self._conv_ns, None)
            self._conv_ns = conv
            namespace = conv

        memory = self._get_memory()
        session_texts = self._sessions.setdefault(namespace, {})
        for item in items:
            if item.document_id:
                session_texts[item.document_id] = item.content
        bucket = self._docs.setdefault(namespace, {})
        stored: list[tuple[str, int, str]] = []  # (safe_text, ts, doc_id)
        for item in items:
            base_ts = _iso_to_unix(item.timestamp)
            for idx, part in enumerate(_chunks(item.content)):
                ts = base_ts + idx * 60  # preserve in-session ordering
                text = _safe(part)
                bucket[text] = (item.document_id, part)
                memory.add_event(
                    text=text,
                    session_id=namespace,
                    event_type="observation",
                    topic=item.metadata.get("session_id") if item.metadata else None,
                    timestamp=ts,
                    metadata={"document_id": item.document_id},
                )
                stored.append((text, ts, item.document_id))

        # Causal discovery is run over THIS sample's events only. The live
        # causal-memory plugin shares the same graph (its events carry newer
        # timestamps), so extract_causality()'s recent-events window would miss
        # the old-dated benchmark events. Drive _discover_in_window directly.
        if self._causal and len(stored) >= 2:
            try:
                from causal_memory.schema import Event

                sampled = _cap_events(stored)
                events = [
                    Event(
                        id=Event.derive_id(namespace, text, ts),
                        text=text,
                        timestamp=ts,
                        session_id=namespace,
                        event_type="observation",
                        topic=item.metadata.get("session_id") if item.metadata else None,
                    )
                    for text, ts, _doc in sampled
                ]
                n = memory._discover_in_window(events)
                logger.info("hydradna: causal discovery wrote %d edge(s)", n)
            except Exception:
                logger.warning(
                    "hydradna: causal discovery skipped for %s", namespace, exc_info=True
                )

    def recall(self, query: RecallQuery, *, namespace: str = "default") -> list[RecallResult]:
        memory = self._get_memory()
        ns = self._ns_to_conv.get(namespace) or namespace
        bucket = self._docs.get(ns, {})
        hits: dict[str, RecallResult] = {}

        def bump(doc_id: str, content: str, score: float) -> None:
            cur = hits.get(doc_id)
            if cur is None or score > cur.score:
                hits[doc_id] = RecallResult(
                    document_id=doc_id, content=content, score=score
                )

        # 1. substring hits
        for event in memory.search(query.query, limit=max(query.top_k, 5)):
            lookup = bucket.get(event.text)
            if lookup is None:
                continue
            doc_id, content = lookup
            bump(doc_id, content, 1.0)
            # 3. causal neighbors of each hit (structural retrieval)
            try:
                for path in memory.find_causes(event.id, max_hops=2):
                    for ev in path.events:
                        if (found := bucket.get(ev.text)) is not None:
                            bump(found[0], found[1], 0.9)
                for path in memory.find_effects(event.id, max_hops=2):
                    for ev in path.events:
                        if (found := bucket.get(ev.text)) is not None:
                            bump(found[0], found[1], 0.9)
            except Exception:
                logger.debug("hydradna: causal expansion failed", exc_info=True)

        # 2. word-overlap fallback
        qwords = set(re.findall(r"[a-z0-9]+", query.query.lower())) - STOPWORDS
        for text, (doc_id, content) in bucket.items():
            ewords = set(re.findall(r"[a-z0-9]+", text.lower()))
            overlap = len(qwords & ewords)
            if overlap > 0:
                bump(doc_id, content, float(overlap))

        ranked = sorted(hits.values(), key=lambda r: r.score, reverse=True)
        # Context packaging: hand the judge WHOLE sessions, not chunk slices.
        # hits is already keyed by document_id (one entry per session), so just
        # swap the sliced content for the full session text.
        session_texts = self._sessions.get(ns, {})
        for r in ranked:
            full = session_texts.get(r.document_id)
            if full:
                r.content = full
        return ranked[: query.top_k]

    def cleanup(self, *, namespace: str = "default") -> None:
        if self._cache_conv and namespace in self._ns_to_conv:
            # Data lives in the shared conversation store; other samples of
            # the same conversation still need it. Just release the mapping.
            self._ns_to_conv.pop(namespace, None)
            self._docs.pop(namespace, None)
            return
        self._docs.pop(namespace, None)
        self._sessions.pop(namespace, None)
        try:
            memory = self._get_memory()
            ns = namespace.replace('"', '\\"')
            self._delete_namespace(memory, ns)
        except Exception:
            logger.warning("hydradna: cleanup failed for %s", namespace, exc_info=True)

    def _delete_namespace(self, memory: Any, ns: str) -> None:
        """Delete a namespace's events + session anchor.

        The engine's per-query deadline is 30s; on a grown graph a single
        DETACH DELETE over the pattern can exceed it (HTTP 408/429) and the
        leftover events accumulate into a runaway. Do a bounded pattern delete
        first; on failure fall back to deleting events by id in small batches,
        each op comfortably under the deadline.
        """
        try:
            memory.client.execute(
                f'MATCH (s:Session {{session_id: "{ns}"}})-[:HAS_EVENT]->(e:Event) '
                "DETACH DELETE e, s",
                timeout_ms=30_000,
            )
            return
        except Exception:
            logger.debug(
                "hydradna: pattern delete failed for %s; batching by id", ns, exc_info=True
            )
        while True:
            rows = memory.client.execute(
                f'MATCH (s:Session {{session_id: "{ns}"}})-[:HAS_EVENT]->(e:Event) '
                "RETURN e.id LIMIT 200",
                timeout_ms=30_000,
            )
            ids = [int(r["e.id"]) for r in rows]
            if not ids:
                break
            for eid in ids:
                memory.client.execute(
                    f"MATCH (e:Event {{id: {eid}}}) DETACH DELETE e", timeout_ms=30_000
                )
        memory.client.execute(
            f'MATCH (s:Session {{session_id: "{ns}"}}) DETACH DELETE s', timeout_ms=30_000
        )

    # ------------------------------------------------------------------ meta

    @property
    def name(self) -> str:
        return "HydraDNA"

    @property
    def capabilities(self) -> set[str]:
        return {"graph", "temporal", "causal"}