"""
Regression tests for parameterized Cypher (item 1) and CONFLICTS/provenance
(item 4). Requires a live HydraDB node -- point HYDRADB_URL at it (defaults
to the docker-compose / local dev address on 18444).

Run:
    HYDRADB_URL=http://localhost:18444 pytest test_parameterization_and_conflicts.py -v
"""

import os
import pytest

from causal_memory.memory import CausalMemory

HYDRADB_URL = os.environ.get("HYDRADB_URL", "http://localhost:18444")


@pytest.fixture
def mem():
    m = CausalMemory(url=HYDRADB_URL)
    m.reset()
    yield m
    m.reset()


def test_adversarial_text_round_trips_without_breaking_the_query(mem):
    """A f-string-built query would have broken on the embedded quote/backslash."""
    text = 'User said: "I moved to SF" and typed C:\\path\\with\\backslashes\nand a newline'
    e = mem.add_event(text=text, session_id="s1", topic="location")
    got = mem.get_event(e.id)
    assert got is not None
    assert got.text == text


def test_corroborating_proposals_raise_confidence(mem):
    e1 = mem.add_event(text="A happened", session_id="s1")
    e2 = mem.add_event(text="B happened", session_id="s1")
    r1 = mem.add_causal_relation(e1.id, e2.id, confidence=0.8, mechanism="m1", proposer="agent-1")
    assert r1.confidence == 0.8
    assert len(r1.proposed_by) == 1

    r2 = mem.add_causal_relation(e1.id, e2.id, confidence=0.85, mechanism="m2", proposer="agent-2")
    assert r2.confidence > r1.confidence
    assert len(r2.proposed_by) == 2
    assert r2.conflict_count == 0


def test_conflicting_direction_writes_conflicts_edge_and_freezes_confidence(mem):
    e1 = mem.add_event(text="A happened", session_id="s1")
    e2 = mem.add_event(text="B happened", session_id="s1")

    r1 = mem.add_causal_relation(e1.id, e2.id, confidence=0.9, mechanism="A causes B", proposer="agent-1")
    r_conflict = mem.add_causal_relation(e2.id, e1.id, confidence=0.9, mechanism="B causes A", proposer="agent-2")

    assert r_conflict.conflict_count == 1
    conflict_edge = mem._get_relation(e2.id, e1.id, "CONFLICTS")
    assert conflict_edge is not None
    assert conflict_edge.conflict_count == 1

    # A third agreeing proposal on the original direction must NOT raise
    # confidence while the conflict stands.
    r1_again = mem.add_causal_relation(e1.id, e2.id, confidence=0.97, mechanism="A causes B again", proposer="agent-3")
    assert r1_again.confidence == r1.confidence
    assert r1_again.conflict_count == 1


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
