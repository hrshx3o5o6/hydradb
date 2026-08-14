"""
Causal Memory Layer - Main interface for causal memory operations.
"""

import uuid
import time
from typing import Optional, List, Dict, Any, Tuple

from .client import HydraDBClient
from .schema import Event, CausalRelation, CausalPath


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
        cell_id: str = "cell-0"
    ):
        self.client = HydraDBClient(url, auth_token, namespace, cell_id)
    
    def add_event(
        self,
        text: str,
        session_id: str,
        event_type: str = "fact",
        topic: Optional[str] = None,
        timestamp: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Event:
        """
        Add an event to the causal memory graph.
        
        Args:
            text: Event description
            session_id: Session identifier
            event_type: Type of event (fact, action, observation)
            topic: Topic/category for the event
            timestamp: Unix timestamp (defaults to current time)
            metadata: Additional metadata
        
        Returns:
            Created Event object
        """
        event_id = str(uuid.uuid4())[:8]
        timestamp = timestamp or int(time.time())
        
        event = Event(
            id=event_id,
            text=text,
            timestamp=timestamp,
            session_id=session_id,
            event_type=event_type,
            topic=topic,
            metadata=metadata or {}
        )
        
        self.client.execute(event.to_cypher())
        return event
    
    def add_causal_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "CAUSES",
        confidence: float = 1.0,
        mechanism: Optional[str] = None,
        evidence: Optional[List[str]] = None
    ) -> CausalRelation:
        """
        Add a causal relationship between events.
        
        Args:
            source_id: Source event ID
            target_id: Target event ID
            relation_type: CAUSES, OVERWRITES, CONFLICTS, ENABLES
            confidence: Confidence score (0.0-1.0)
            mechanism: Description of causal mechanism
            evidence: List of evidence sources
        
        Returns:
            Created CausalRelation object
        """
        relation = CausalRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            mechanism=mechanism,
            timestamp=int(time.time()),
            evidence=evidence or []
        )
        
        self.client.execute(relation.to_cypher())
        return relation
    
    def add_overwrite(
        self,
        old_event_id: str,
        new_event_id: str,
        reason: Optional[str] = None
    ) -> CausalRelation:
        """
        Mark that a new event overwrites/supersedes an old event.
        
        Args:
            old_event_id: Old event being superseded
            new_event_id: New event that supersedes
            reason: Reason for the overwrite
        
        Returns:
            Created OVERWRITES relation
        """
        return self.add_causal_relation(
            source_id=new_event_id,
            target_id=old_event_id,
            relation_type="OVERWRITES",
            confidence=1.0,
            mechanism=reason
        )
    
    def find_causal_path(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 5
    ) -> List[CausalPath]:
        """
        Find causal paths between two events.
        
        Args:
            source_id: Source event ID
            target_id: Target event ID
            max_hops: Maximum path length
        
        Returns:
            List of CausalPath objects
        """
        query = f"""
        CALL algo.SPpaths({{
            sourceNode: '{source_id}',
            targetNode: '{target_id}',
            relTypes: ['CAUSES', 'ENABLES'],
            maxLen: {max_hops},
            relDirection: 'outgoing',
            pathCount: 5
        }}) YIELD path, pathWeight, pathCost
        RETURN path, pathWeight, pathCost
        """
        
        results = self.client.execute(query)
        paths = []
        
        for row in results:
            path_data = row.get("path", {})
            paths.append(CausalPath(
                events=[],  # TODO: Parse nodes from path
                relations=[],  # TODO: Parse relationships from path
                path_weight=row.get("pathWeight", 0.0),
                path_cost=row.get("pathCost", 0.0)
            ))
        
        return paths
    
    def find_all_effects(
        self,
        source_id: str,
        max_hops: int = 3
    ) -> List[CausalPath]:
        """
        Find all causal effects of an event.
        
        Args:
            source_id: Source event ID
            max_hops: Maximum path length
        
        Returns:
            List of CausalPath objects
        """
        query = f"""
        CALL algo.SSpaths({{
            sourceNode: '{source_id}',
            relTypes: ['CAUSES', 'ENABLES'],
            maxLen: {max_hops},
            pathCount: 20
        }}) YIELD path
        RETURN path
        """
        
        results = self.client.execute(query)
        paths = []
        
        for row in results:
            paths.append(CausalPath(events=[], relations=[]))
        
        return paths
    
    def find_current_fact(self, topic: str) -> Optional[Event]:
        """
        Find the current (non-overwritten) fact on a topic.
        
        Args:
            topic: Topic to search for
        
        Returns:
            Current Event or None
        """
        query = f"""
        MATCH (e:Event {{topic: "{topic}"}})
        WHERE NOT (e)<-[:OVERWRITES]-()
        RETURN e
        """
        
        results = self.client.execute(query)
        if results:
            row = results[0]["e"]
            return Event(
                id=row.get("id", ""),
                text=row.get("text", ""),
                timestamp=row.get("timestamp", 0),
                session_id=row.get("session_id", ""),
                event_type=row.get("type", "fact"),
                topic=row.get("topic")
            )
        return None
    
    def why(self, question: str) -> str:
        """
        Answer a "why" question by tracing causal chains.
        
        Args:
            question: Natural language "why" question
        
        Returns:
            Natural language explanation
        """
        # TODO: Implement LLM integration to:
        # 1. Parse question to identify target event
        # 2. Find causal path to target
        # 3. Generate natural language explanation
        
        return "Why query not yet implemented. Requires LLM integration."
    
    def search(self, query: str) -> List[Dict]:
        """
        Search for events matching a query.
        
        Args:
            query: Search query (simple text match for now)
        
        Returns:
            List of matching events
        """
        cypher = f"""
        MATCH (e:Event)
        WHERE e.text CONTAINS "{query}"
        RETURN e
        ORDER BY e.timestamp DESC
        LIMIT 10
        """
        
        results = self.client.execute(cypher)
        events = []
        
        for row in results:
            e = row.get("e", {})
            events.append({
                "id": e.get("id", ""),
                "text": e.get("text", ""),
                "timestamp": e.get("timestamp", 0),
                "session_id": e.get("session_id", ""),
                "topic": e.get("topic")
            })
        
        return events
    
    def add_conversation(
        self,
        messages: List[Dict[str, str]],
        session_id: str
    ) -> List[Event]:
        """
        Add a conversation and extract causal facts.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            session_id: Session identifier
        
        Returns:
            List of extracted Event objects
        """
        # TODO: Implement LLM-based causal extraction
        # For now, just store each message as an event
        
        events = []
        for msg in messages:
            event = self.add_event(
                text=msg.get("content", ""),
                session_id=session_id,
                event_type="observation",
                metadata={"role": msg.get("role", "user")}
            )
            events.append(event)
        
        return events
