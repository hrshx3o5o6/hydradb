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
from .schema import Event, CausalRelation, CausalPath, hash_id


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
    ):
        self.client = HydraDBClient(url, auth_token, namespace, cell_id, admin_url)

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
        """
        query = (
            "CALL algo.SPpaths({"
            f"sourceNode: {event_id}, "
            f"targetNode: {event_id}, "
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
        query = (
            'MATCH (e:Event) '
            "RETURN e.id, e.text, e.timestamp, e.session_id, e.type, e.topic "
            "ORDER BY e.timestamp DESC"
        )
        events = [self._event_from_row(row) for row in self.client.execute(query)]
        needle = text.lower()
        return [e for e in events if needle in e.text.lower()][:limit]

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
        Answer a "why" question by tracing causal chains.
        """
        # TODO: LLM integration to parse question -> target event, then explain.
        return "Why query not yet implemented. Requires LLM integration."

    def add_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: str,
    ) -> List[Event]:
        """
        Store each message as an observation event (LLM extraction TODO).
        """
        events = []
        for msg in messages:
            event = self.add_event(
                text=msg.get("content", ""),
                session_id=session_id,
                event_type="observation",
                metadata={"role": msg.get("role", "user")},
            )
            events.append(event)
        return events