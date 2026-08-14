"""
Data models for causal memory graph.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime


@dataclass
class Event:
    """An event/fact in the causal memory graph."""
    
    id: str
    text: str
    timestamp: int  # Unix timestamp
    session_id: str
    event_type: str = "fact"  # fact, action, observation
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_cypher(self) -> str:
        """Generate Cypher CREATE statement for this event."""
        props = [
            f'id: "{self.id}"',
            f'text: "{self.text}"',
            f'timestamp: {self.timestamp}',
            f'session_id: "{self.session_id}"',
            f'type: "{self.event_type}"'
        ]
        
        if self.topic:
            props.append(f'topic: "{self.topic}"')
        
        for key, value in self.metadata.items():
            if isinstance(value, str):
                props.append(f'{key}: "{value}"')
            else:
                props.append(f'{key}: {value}')
        
        props_str = ", ".join(props)
        return f'CREATE (e:Event {{{props_str}}})'


@dataclass
class CausalRelation:
    """A causal relationship between events."""
    
    source_id: str
    target_id: str
    relation_type: str = "CAUSES"  # CAUSES, OVERWRITES, CONFLICTS, ENABLES
    confidence: float = 1.0
    mechanism: Optional[str] = None
    timestamp: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_cypher(self) -> str:
        """Generate Cypher MERGE statement for this relation."""
        props = [f'confidence: {self.confidence}']
        
        if self.mechanism:
            props.append(f'mechanism: "{self.mechanism}"')
        
        if self.timestamp:
            props.append(f'timestamp: {self.timestamp}')
        
        if self.evidence:
            evidence_str = ", ".join(f'"{e}"' for e in self.evidence)
            props.append(f'evidence: [{evidence_str}]')
        
        for key, value in self.metadata.items():
            if isinstance(value, str):
                props.append(f'{key}: "{value}"')
            else:
                props.append(f'{key}: {value}')
        
        props_str = ", ".join(props)
        return f'''
MATCH (a:Event {{id: "{self.source_id}"}}), (b:Event {{id: "{self.target_id}"}})
MERGE (a)-[r:{self.relation_type} {{{props_str}}}]->(b)
RETURN r
'''


@dataclass
class CausalPath:
    """A causal path through the memory graph."""
    
    events: List[Event]
    relations: List[CausalRelation]
    path_weight: float = 0.0
    path_cost: float = 0.0
    
    def to_narrative(self) -> str:
        """Convert causal path to natural language narrative."""
        if not self.events:
            return "No causal path found."
        
        narrative_parts = []
        for i, event in enumerate(self.events):
            narrative_parts.append(event.text)
            if i < len(self.relations):
                rel = self.relations[i]
                if rel.relation_type == "CAUSES":
                    narrative_parts.append("→")
                elif rel.relation_type == "OVERWRITES":
                    narrative_parts.append("(superseded)")
                elif rel.relation_type == "ENABLES":
                    narrative_parts.append("→ enabled →")
        
        return " ".join(narrative_parts)
