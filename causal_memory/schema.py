"""
Data models for causal memory graph.

HydraDB's query engine contract (verified against live node):
- Node ids MUST be integers.
- CREATE requires an edge pattern (no lone-vertex CREATE).
- Reusing an existing vertex id as an endpoint preserves its metadata (upsert-like).
- RETURN supports only <binding>.<property> or count(*).
- WHERE supports boolean combinations of property comparisons only.
- Path procedures use CALL algo.SPpaths / SSpaths / MSpaths ... YIELD ... RETURN.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


def hash_id(seed: str) -> int:
    """Deterministic integer id from a string (FNV-1a, 63-bit)."""
    h = 0x811C9DC5
    for b in seed.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFFFFFFFFFF
    return h & 0x7FFFFFFFFFFFFFFF


@dataclass
class Event:
    """An event/fact in the causal memory graph."""

    id: int
    text: str
    timestamp: int  # Unix timestamp
    session_id: str
    event_type: str = "fact"  # fact, action, observation
    topic: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def derive_id(session_id: str, text: str, timestamp: int) -> int:
        """Deterministic integer event id from content, safe to recompute."""
        return hash_id(f"{session_id}|{text}|{timestamp}")

    def props(self) -> str:
        """Property map string for use inside a node pattern."""
        props = [
            f'id: {self.id}',
            f'text: "{self.text}"',
            f'timestamp: {self.timestamp}',
            f'session_id: "{self.session_id}"',
            f'type: "{self.event_type}"',
        ]

        if self.topic:
            props.append(f'topic: "{self.topic}"')

        for key, value in self.metadata.items():
            if isinstance(value, str):
                props.append(f'{key}: "{value}"')
            else:
                props.append(f'{key}: {value}')

        return ", ".join(props)


@dataclass
class CausalRelation:
    """A causal relationship between events."""

    source_id: int
    target_id: int
    relation_type: str = "CAUSES"  # CAUSES, OVERWRITES, CONFLICTS, ENABLES
    confidence: float = 1.0
    mechanism: Optional[str] = None
    timestamp: Optional[int] = None
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def props(self) -> str:
        """Edge property map string."""
        props = [f"confidence: {self.confidence}"]

        if self.mechanism:
            props.append(f'mechanism: "{self.mechanism}"')

        if self.timestamp:
            props.append(f"timestamp: {self.timestamp}")

        if self.evidence:
            evidence_str = ", ".join(f'"{e}"' for e in self.evidence)
            props.append(f"evidence: [{evidence_str}]")

        for key, value in self.metadata.items():
            if isinstance(value, str):
                props.append(f'{key}: "{value}"')
            else:
                props.append(f'{key}: {value}')

        return ", ".join(props)


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