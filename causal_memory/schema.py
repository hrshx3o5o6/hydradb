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

import json
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

    def cypher_props(self, prefix: str = "e") -> "tuple[str, Dict[str, Any]]":
        """
        Property map fragment using $-parameters (engine-verified: opencypher.rs
        threads `parameters: &BTreeMap<String, VertexPropertyValue>` through
        lower_create_mutations/lower_simple_merge, and the HTTP client's
        `parameters` body field maps 1:1 onto it). Property VALUES never enter
        the query text, so arbitrary event text (quotes, backslashes,
        newlines) is safe.

        Returns (fragment, params) where fragment goes inside `{...}` in the
        pattern and params merges into the query's top-level parameters dict.
        Prefix must be unique per node/edge referenced in a single query.
        """
        fields = {
            "id": self.id,
            "text": self.text,
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "type": self.event_type,
        }
        if self.topic:
            fields["topic"] = self.topic
        fields.update(self.metadata)

        parts = []
        params: Dict[str, Any] = {}
        for key, value in fields.items():
            pname = f"{prefix}_{key}"
            parts.append(f"{key}: ${pname}")
            params[pname] = value
        return ", ".join(parts), params


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
    # Provenance: one entry per agent/session that proposed this edge, e.g.
    # {"session_id": "...", "confidence": 0.9, "mechanism": "...", "timestamp": 123}.
    # Corroboration (multiple agreeing entries) may raise aggregate confidence;
    # a CONFLICTS edge on the same pair means it must NOT be raised further.
    proposed_by: List[Dict[str, Any]] = field(default_factory=list)
    conflict_count: int = 0
    # Counterfactual necessity: True if a narrow interventional check judged
    # the target would NOT plausibly have occurred without the source
    # (confidence-band routing gate; see discovery_core.route_confidence).
    necessity: Optional[bool] = None

    def cypher_props(self, prefix: str = "r") -> "tuple[str, Dict[str, Any]]":
        """Edge property map fragment using $-parameters. See Event.cypher_props."""
        fields: Dict[str, Any] = {"confidence": self.confidence}
        if self.mechanism:
            fields["mechanism"] = self.mechanism
        if self.timestamp:
            fields["timestamp"] = self.timestamp
        if self.evidence:
            # Stored vertex/edge properties are scalar-only (engine's
            # VertexPropertyValue: Integer/SignedInteger/Bool/Float/String —
            # verified in src/core/model.rs, no List/Map variant). Encode as
            # JSON text rather than a Cypher list literal, which is a query
            # PARAMETER type but not a storable PROPERTY type.
            fields["evidence"] = json.dumps(list(self.evidence))
        if self.proposed_by:
            fields["proposed_by"] = json.dumps(self.proposed_by)
        if self.conflict_count:
            fields["conflict_count"] = self.conflict_count
        if self.necessity is not None:
            fields["necessity"] = self.necessity
        fields.update(self.metadata)

        parts = []
        params: Dict[str, Any] = {}
        for key, value in fields.items():
            pname = f"{prefix}_{key}"
            parts.append(f"{key}: ${pname}")
            params[pname] = value
        return ", ".join(parts), params


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