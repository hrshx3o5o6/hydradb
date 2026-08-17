"""
Causal Memory Layer - Main interface for causal memory operations.

Graph model (matches HydraDB query-engine constraints):
  (s:Session {id: <hash(session_id)>})-[:HAS_EVENT]->(e:Event {id: <int>, ...})
  (e1)-[:CAUSES {confidence, mechanism}]->(e2)
  (e2)-[:OVERWRITES]->(e1)          # e2 supersedes e1

Every vertex is created as part of an edge pattern (the engine rejects
lone-vertex CREATE). Session anchors group events; causal edges link them.
Event ids are deterministic hashes of (session, text, timestamp) so a harness
re-logging the same fact resolves to the same vertex.
"""

import time
from typing import Optional, List, Dict, Any

from .client import HydraDBClient
from .llm import LLM, LLMError, extract_json
from .schema import Event, CausalRelation, CausalPath, hash_id
from .tracing import Trace


class CausalMemory:
    """
    Causal memory layer for AI agents.

    Stores cause→effect relationships, not just facts.
    Enables "why" queries by tracing causal chains.
    """

    def __init__(
        self,
        url: str = "http://localhost:18443",
        auth_token: str = "local-dev-auth-token-32-characters-long",
        namespace: str = "default",
        cell_id: str = "cell-0",
        admin_url: Optional[str] = None,
        llm: Optional[LLM] = None,
    ):
        self.client = HydraDBClient(url, auth_token, namespace, cell_id, admin_url)
        self._llm = llm

    def _get_llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    # ------------------------------------------------------------------ writes

    def add_event(
        self,
        text: str,
        session_id: str,
        event_type: str = "fact",
        topic: Optional[str] = None,
        timestamp: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Event:
        """
        Add an event to the causal memory graph.

        The event is created as (s:Session)-[:HAS_EVENT]->(e:Event) so the
        engine accepts it; the session anchor is (re)used idempotently.
        """
        timestamp = timestamp if timestamp is not None else int(time.time())
        session_root = hash_id(f"session:{session_id}")
        event_id = Event.derive_id(session_id, text, timestamp)

        event = Event(
            id=event_id,
            text=text,
            timestamp=timestamp,
            session_id=session_id,
            event_type=event_type,
            topic=topic,
            metadata=metadata or {},
        )

        query = (
            f"MERGE (s:Session {{id: {session_root}, session_id: \"{session_id}\"}})"
            f"-[:HAS_EVENT]->(e:Event {{{event.props()}}})"
        )
        self.client.execute(query)
        return event

    def add_causal_relation(
        self,
        source_id: int,
        target_id: int,
        relation_type: str = "CAUSES",
        confidence: float = 1.0,
        mechanism: Optional[str] = None,
        evidence: Optional[List[str]] = None,
    ) -> CausalRelation:
        """
        Add (or merge, idempotently) a causal relationship between events.
        """
        relation = CausalRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            mechanism=mechanism,
            timestamp=int(time.time()),
            evidence=evidence or [],
        )

        query = (
            f"MERGE (a:Event {{id: {relation.source_id}}})"
            f"-[r:{relation.relation_type} {{{relation.props()}}}]"
            f"->(b:Event {{id: {relation.target_id}}})"
        )
        self.client.execute(query)
        return relation

    def add_overwrite(
        self,
        old_event_id: int,
        new_event_id: int,
        reason: Optional[str] = None,
    ) -> CausalRelation:
        """
        Mark that a new event overwrites/supersedes an old event.
        """
        return self.add_causal_relation(
            source_id=new_event_id,
            target_id=old_event_id,
            relation_type="OVERWRITES",
            confidence=1.0,
            mechanism=reason,
        )

    # ------------------------------------------------------------------- reads

    def find_causal_path(
        self,
        source_id: int,
        target_id: int,
        max_hops: int = 5,
        path_count: int = 5,
    ) -> List[CausalPath]:
        """
        Find causal paths between two events via algo.SPpaths.
        """
        query = (
            "CALL algo.SPpaths({"
            f"sourceNode: {source_id}, "
            f"targetNode: {target_id}, "
            "relTypes: ['CAUSES', 'ENABLES'], "
            f"maxLen: {max_hops}, "
            "relDirection: 'outgoing', "
            f"pathCount: {path_count}"
            "}) YIELD path, pathWeight, pathCost "
            "RETURN path, pathWeight, pathCost"
        )

        results = self.client.execute(query)
        paths = []
        for row in results:
            paths.append(self._path_from_row(row))
        return paths

    def find_all_effects(
        self,
        source_id: int,
        max_hops: int = 3,
        path_count: int = 20,
    ) -> List[CausalPath]:
        """
        Find all causal paths starting from an event via algo.SSpaths.
        """
        query = (
            "CALL algo.SSpaths({"
            f"sourceNode: {source_id}, "
            "relTypes: ['CAUSES', 'ENABLES'], "
            f"maxLen: {max_hops}, "
            f"pathCount: {path_count}"
            "}) YIELD path "
            "RETURN path"
        )

        results = self.client.execute(query)
        paths = []
        for row in results:
            paths.append(self._path_from_row(row))
        return paths

    def find_causes(self, event_id: int, max_hops: int = 3) -> List[CausalPath]:
        """
        Find causal paths leading INTO an event (its ancestry).

        Uses SSpaths (single-source) with relDirection 'incoming'; SPpaths
        with source==target yields a zero-length path.
        """
        query = (
            "CALL algo.SSpaths({"
            f"sourceNode: {event_id}, "
            "relTypes: ['CAUSES', 'ENABLES'], "
            f"maxLen: {max_hops}, "
            "relDirection: 'incoming', "
            "pathCount: 20"
            "}) YIELD path "
            "RETURN path"
        )

        results = self.client.execute(query)
        paths = []
        for row in results:
            paths.append(self._path_from_row(row))
        return paths

    def find_events_by_topic(self, topic: str, limit: int = 20) -> List[Event]:
        """
        Find events on a topic (WHERE-only predicate; engine rejects CONTAINS).
        """
        query = (
            'MATCH (e:Event) WHERE e.topic = "' + topic + '" '
            "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic "
            f"ORDER BY e.timestamp DESC LIMIT {limit}"
        )
        return [self._event_from_row(row) for row in self.client.execute(query)]

    def find_current_fact(self, topic: str) -> Optional[Event]:
        """
        Find the most recent event on a topic (proxy for "current" fact).
        """
        events = self.find_events_by_topic(topic, limit=1)
        return events[0] if events else None

    def get_event(self, event_id: int) -> Optional[Event]:
        """Fetch a single event by id."""
        query = (
            f"MATCH (e:Event {{id: {event_id}}}) "
            "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic"
        )
        rows = self.client.execute(query)
        if not rows:
            return None
        return self._event_from_row(rows[0])

    def search(self, text: str, limit: int = 10) -> List[Event]:
        """
        Search events by substring of text.

        The engine rejects CONTAINS in WHERE, so fetch recent events and
        filter client-side.
        """
        trace = Trace("search", text)
        s1 = trace.stage("fetch_events",
                         "MATCH (e:Event) RETURN e.* ORDER BY e.timestamp DESC")
        query = (
            'MATCH (e:Event) '
            "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic "
            "ORDER BY e.timestamp DESC"
        )
        events = [self._event_from_row(row) for row in self.client.execute(query)]
        s1.detail = f"engine returned {len(events)} event row(s)"
        s1.finish()
        s2 = trace.stage("filter", f"client-side substring '{text}' (CONTAINS rejected by engine)")
        needle = text.lower()
        hits = [e for e in events if needle in e.text.lower()][:limit]
        s2.detail = f"{len(hits)} match(es) after substring filter"
        s2.finish()
        trace.result = f"{len(hits)} match(es)"
        trace.emit()
        return hits

    # --------------------------------------------------------------- helpers

    def _path_from_row(self, row: Dict[str, Any]) -> CausalPath:
        path = row.get("path", {})
        events = []
        for n in path.get("nodes", []):
            props = n.get("properties", {})
            events.append(Event(
                id=props.get("id", n.get("id", 0)),
                text=props.get("text", ""),
                timestamp=int(props.get("timestamp", 0)),
                session_id=props.get("session_id", ""),
                event_type=props.get("type", "fact"),
                topic=props.get("topic"),
            ))
        relations = []
        for r in path.get("relationships", []):
            rprops = r.get("properties", {})
            relations.append(CausalRelation(
                source_id=r.get("src", 0),
                target_id=r.get("dst", 0),
                relation_type=r.get("type", "CAUSES"),
                confidence=float(rprops.get("confidence", 1.0)),
                mechanism=rprops.get("mechanism"),
                timestamp=int(rprops.get("timestamp", 0)) if rprops.get("timestamp") else None,
            ))
        return CausalPath(
            events=events,
            relations=relations,
            path_weight=float(row.get("pathWeight", 0.0) or 0.0),
            path_cost=float(row.get("pathCost", 0.0) or 0.0),
        )

    def _event_from_row(self, row: Dict[str, Any]) -> Event:
        return Event(
            id=int(row.get("e.id") or 0),
            text=row.get("e.text") or "",
            timestamp=int(row.get("e.timestamp") or 0),
            session_id=row.get("e.session_id") or "",
            event_type=row.get("e.type") or "fact",
            topic=row.get("e.topic"),
        )

    def why(self, question: str) -> str:
        """
        Answer a "why" question in natural language.

        Pipeline:
          1. LLM extracts which event(s) the question targets (from stored events).
          2. find_causes() pulls the causal ancestry of the target.
          3. The path (events + mechanisms + confidences) is handed to the LLM
             to narrate: "X happened because Y, which followed Z..."

        Returns a plain-sentence answer.
        """
        trace = Trace("why", question)
        llm = self._get_llm()
        s1 = trace.stage("build_digest", "fetch recent events for LLM resolution")
        events = self._recent_events(limit=10000)
        event_digest = "\n".join(
            f"- id {e.id}: {e.text} (session {e.session_id}, topic {e.topic})"
            for e in events[-250:]  # newest 250 for LLM digestion
        )
        s1.detail = f"digested {min(len(events), 250)} event(s)"
        s1.finish()

        # 1. Resolve the question to a target event id.
        s2 = trace.stage("resolve_target", "LLM maps question -> event id")
        target_json = llm.complete(
            system=(
                "You map natural-language questions to stored event ids. "
                "Respond with JSON only: {\"event_id\": <int or null>}."
            ),
            user=f"Events in memory:\n{event_digest}\n\n"
                 f'Question: "{question}"\nWhich event id is the question asking about? '
                 "If none fits, return null.",
            json_mode=True,
        )
        try:
            target = extract_json(target_json).get("event_id")
        except LLMError:
            target = None
        if not target:
            s2.detail = "LLM returned null -> keyword fallback"
            s2.finish()
            s2b = trace.stage("keyword_fallback",
                              "word-overlap scoring across stored events")
            target = self._match_event_by_keywords(question)
            s2b.detail = f"resolved to event {target}" if target else "no keyword match"
            s2b.finish()
        else:
            s2.detail = f"resolved to event {target}"
            s2.finish()

        if not target:
            trace.result = "no matching event"
            trace.emit()
            return f'I don\'t have a memory entry that matches "{question}".'

        # 2. Pull causal ancestry.
        s3 = trace.stage("traverse_graph",
                         "CALL algo.SSpaths relDirection=incoming maxLen=4")
        paths = self.find_causes(int(target), max_hops=4)
        s3.detail = f"engine returned {len(paths)} causal path(s)"
        s3.finish()
        if not paths:
            base: Optional[Event] = self.get_event(int(target))
            label = base.text if base else str(target)
            trace.result = "no causal chain recorded"
            trace.emit()
            return f'No causal chain recorded yet for "{label}". ' \
                   "Log more observations and run extract_causality() to discover one."

        # 3. Narrate the most complete (longest) causal chain.
        s4 = trace.stage("narrate", "LLM turns longest causal chain into prose")
        best = max(paths, key=lambda p: len(p.events))
        chain_lines = []
        for i, ev in enumerate(best.events):
            chain_lines.append(f"event {ev.id} @ t={ev.timestamp}: {ev.text}")
            if i < len(best.relations) and best.relations[i].relation_type == "CAUSES":
                r = best.relations[i]
                mech = f' — "{r.mechanism}"' if r.mechanism else ""
                chain_lines.append(f"    <- CAUSED BY (conf {r.confidence:.2f}){mech}")
        chain_text = "\n".join(chain_lines)
        s4.detail = f"chain of {len(best.events)} event(s), {len(best.relations)} relation(s)"
        s4.finish()

        answer = llm.complete(
            system=(
                "You explain causal chains found in an agent's memory graph. "
                "Answer the question in 2-4 clear sentences. Start directly with "
                "the explanation; do not mention event ids or graph internals."
            ),
            user=f"Causal chain leading to the target event:\n{chain_text}\n\n"
                 f'Original question: "{question}"',
        )
        trace.result = answer.strip()[:120]
        trace.emit()
        return answer.strip()

    def extract_causality(self, window_events: Optional[int] = 12,
                          session_id: Optional[str] = None) -> int:
        """
        LLM proposes CAUSES edges between recent events (causal discovery).

        Batches the most recent events in timestamp order (optionally scoped
        to one session) and asks the LLM to propose cause-effect pairs.

        Returns the number of edges written.
        """
        llm = self._get_llm()
        events = self._recent_events(limit=window_events)
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        events.sort(key=lambda e: e.timestamp)
        if len(events) < 2:
            return 0

        # Large streams are digested in overlapping temporal windows; a single
        # 80+ event digest makes the LLM under-propose and miss chain links.
        windows = self._temporal_windows(events)
        total = 0
        for win in windows:
            total += self._discover_in_window(win)
        return total

    def _temporal_windows(self, events: List[Event], size: int = 18, overlap: int = 6) -> List[List[Event]]:
        """Split a sorted event list into overlapping windows."""
        if len(events) <= size:
            return [events]
        windows = []
        step = size - overlap
        for start in range(0, len(events), step):
            win = events[start : start + size]
            if len(win) >= 2:
                windows.append(win)
            if start + size >= len(events):
                break
        return windows

    def _discover_in_window(self, events: List[Event]) -> int:
        """Run LLM causal discovery over one window. Returns edges written."""
        llm = self._get_llm()
        # Drop duplicate-text events: repeated mechanical actions (same tool
        # invocation re-run, same log line) carry no causal signal and make the
        # LLM chain near-identical texts together. Keep the first occurrence.
        seen_texts = set()
        uniq = []
        for e in events:
            key = (e.session_id, e.text)
            if key in seen_texts:
                continue
            seen_texts.add(key)
            uniq.append(e)
        if len(uniq) < 2:
            return 0
        events = uniq
        digest_lines = []
        for i, e in enumerate(events):
            digest_lines.append(
                f"[{i}] t={e.timestamp} session={e.session_id} "
                f"topic={e.topic} text={e.text}"
            )
        digest = "\n".join(digest_lines)
        edges_json = llm.complete(
            system=(
                "You are a causal-discovery assistant for an agent memory graph. "
                "Given a time-ordered list of observations, propose every "
                "defensible CAUSE->EFFECT pair. The events are indexed [0], [1], "
                "... with increasing time. Rules:\n"
                "- A cause index must be LESS than its effect index (cause "
                "happens earlier).\n"
                "- Only propose genuine pairs with a clear mechanism; state it.\n"
                "- If A->B and B->C are both real, emit both edges.\n"
                "- Confidence reflects causal strength, not just temporal "
                "proximity. Prefer a few strong edges over many weak ones.\n"
                "- NEVER link two events with identical or near-identical text "
                "(a repeated action is not a cause of another repeated "
                "action).\n"
                "- Agent tool actions (text starts with 'tool:') are mechanical "
                "steps. Do NOT chain consecutive tool actions into a causal "
                "ladder unless one is the DIRECT, specific result of the other. "
                "Prefer linking a user request or decision to the actions that "
                "follow it.\n"
                "- Prefer linking distinct-meaning events: a decision or user "
                "request CAUSES the action that follows it; an action CAUSES "
                "the result that directly derives from it.\n"
                "- Return JSON ONLY: {\"edges\": [{\"source_idx\": <int>, "
                "\"target_idx\": <int>, \"confidence\": 0.0-1.0, "
                "\"mechanism\": \"<why>\"}]}. Use the index numbers, never "
                "event ids."
            ),
            user=digest,
            json_mode=True,
        )
        try:
            data = extract_json(edges_json)
        except LLMError:
            return 0
        edges = data.get("edges", []) if isinstance(data, dict) else []
        written = 0
        for edge in edges:
            try:
                src_idx = int(edge.get("source_idx"))
                dst_idx = int(edge.get("target_idx"))
                # validate indices and temporal ordering
                if src_idx == dst_idx:
                    continue
                if not (0 <= src_idx < len(events) and 0 <= dst_idx < len(events)):
                    continue
                if events[src_idx].timestamp > events[dst_idx].timestamp:
                    continue
                src = events[src_idx]
                dst = events[dst_idx]
                # Mechanical tool->tool chains are noise, not memory. Only keep
                # action edges when a non-action event (user request, decision,
                # result) is on one end. User->action and decision->result are
                # the real signal; bash->bash ladders are not.
                if (src.event_type == "action" and dst.event_type == "action"):
                    continue
                conf = float(edge.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                if conf < 0.8:
                    continue  # weak/hallucinated links are noise, not memory
                mech = edge.get("mechanism") or None
                self.add_causal_relation(
                    source_id=src.id,
                    target_id=dst.id,
                    relation_type="CAUSES",
                    confidence=conf,
                    mechanism=mech,
                )
                written += 1
            except (TypeError, ValueError):
                continue
        return written

    def reset(self) -> None:
        """Wipe all events, sessions, and causal edges (demo cleanup)."""
        self.client.execute("MATCH (e:Event) DETACH DELETE e")
        self.client.execute("MATCH (s:Session) DETACH DELETE s")

    def observe(self, text: str, session_id: str, topic: Optional[str] = None,
                timestamp: Optional[int] = None) -> Event:
        """
        Log a raw observation. Call this from a harness for every event.

        Optionally triggers extract_causality() after the batch when
        autocausal=True (see add_conversation).
        """
        return self.add_event(
            text=text,
            session_id=session_id,
            event_type="observation",
            topic=topic,
            timestamp=timestamp,
        )

    def _match_event_by_keywords(self, question: str) -> Optional[int]:
        """
        Fallback target resolution: score stored events by word overlap with
        the question (stopwords removed), pick the best match.
        """
        import re

        stopwords = {
            "the", "a", "an", "is", "was", "were", "did", "do", "does", "why",
            "what", "when", "who", "how", "to", "of", "in", "on", "it", "for",
            "and", "or", "but", "their", "user", "from", "with",
        }
        qwords = {
            w for w in re.findall(r"[a-z0-9]+", question.lower())
            if w not in stopwords
        }
        best_id = None
        best_score = 0
        best_timestamp = -1
        for e in self._recent_events(limit=10000):
            ewords = set(re.findall(r"[a-z0-9]+", e.text.lower()))
            score = len(qwords & ewords)
            if score > best_score or (score == best_score and e.timestamp > best_timestamp):
                best_score = score
                best_id = e.id
                best_timestamp = e.timestamp
        return best_id if best_score > 0 else None

    def _recent_events(self, limit: int = 50) -> List[Event]:
        """Fetch most recent Event nodes, oldest first within the window."""
        query = (
            "MATCH (e:Event) "
            "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic "
            "ORDER BY e.timestamp DESC"
        )
        events = [self._event_from_row(row) for row in self.client.execute(query)]
        return list(reversed(events[:limit]))

    def add_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
        autocausal: bool = True,
    ) -> List[Event]:
        """
        Store each message as an observation event, then propose causal edges.
        """
        events = []
        for msg in messages:
            event = self.observe(
                text=msg.get("content", ""),
                session_id=session_id,
                topic=msg.get("topic"),
            )
            events.append(event)

        if autocausal and len(events) >= 2:
            try:
                n = self.extract_causality(window_events=len(events))
                print(f"  [causal-discovery] proposed {n} causal edge(s)")
            except LLMError as e:
                print(f"  [causal-discovery] skipped: {e}")

        return events