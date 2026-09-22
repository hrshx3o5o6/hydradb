"""
Deterministic (mocked-LLM) test that the confidence-band gate (item 8)
actually routes through _verify_necessity for the 0.70-0.85 band, and
skips it entirely outside that band -- doesn't rely on a live LLM call
happening to land in the band to exercise the code path.

Run:
    HYDRADB_URL=http://localhost:18444 pytest test_confidence_band_gate.py -v
"""

import json
import os
import time
from unittest.mock import patch

import pytest

from causal_memory.memory import CausalMemory

HYDRADB_URL = os.environ.get("HYDRADB_URL", "http://localhost:18444")


@pytest.fixture
def mem():
    m = CausalMemory(url=HYDRADB_URL)
    m.reset()
    yield m
    m.reset()


def _fake_llm_proposing(confidence: float):
    """A stand-in LLM whose .complete() always proposes one edge [0]->[1]
    at the given confidence, with a mechanism that will pass the overlap
    check (shares vocabulary with the target event text)."""
    class _FakeLLM:
        def complete(self, system, user, json_mode=False):
            return json.dumps({
                "edges": [{
                    "source_idx": 0, "target_idx": 1,
                    "confidence": confidence,
                    "mechanism": "decision to migrate database triggered export",
                }]
            })
    return _FakeLLM()


def test_verify_band_calls_necessity_check_and_a_no_raises_confidence(mem):
    t0 = int(time.time()) - 100
    e0 = mem.add_event(text="User decided to migrate the database", session_id="s1", timestamp=t0)
    e1 = mem.add_event(text="Exported the database schema for migration", session_id="s1", timestamp=t0 + 5)

    mem._llm = _fake_llm_proposing(0.75)  # squarely in the "verify" band

    with patch.object(CausalMemory, "_verify_necessity", return_value=True) as mock_verify:
        written = mem._discover_in_window([e0, e1], proposer="test")
        assert mock_verify.called, "0.70-0.85 confidence must route through _verify_necessity"
    assert written == 1

    rel = mem._get_relation(e0.id, e1.id, "CAUSES")
    assert rel is not None
    assert rel.confidence >= 0.85, "a 'necessary' verdict should raise confidence into the accept range"


def test_verify_band_a_yes_verdict_rejects_the_edge(mem):
    t0 = int(time.time()) - 100
    e0 = mem.add_event(text="User decided to migrate the database", session_id="s1", timestamp=t0)
    e1 = mem.add_event(text="Exported the database schema for migration", session_id="s1", timestamp=t0 + 5)

    mem._llm = _fake_llm_proposing(0.75)

    with patch.object(CausalMemory, "_verify_necessity", return_value=False):
        written = mem._discover_in_window([e0, e1], proposer="test")
    assert written == 0
    assert mem._get_relation(e0.id, e1.id, "CAUSES") is None


def test_accept_band_skips_the_necessity_check_entirely(mem):
    t0 = int(time.time()) - 100
    e0 = mem.add_event(text="User decided to migrate the database", session_id="s1", timestamp=t0)
    e1 = mem.add_event(text="Exported the database schema for migration", session_id="s1", timestamp=t0 + 5)

    mem._llm = _fake_llm_proposing(0.95)  # above the verify band

    with patch.object(CausalMemory, "_verify_necessity") as mock_verify:
        written = mem._discover_in_window([e0, e1], proposer="test")
        assert not mock_verify.called, "high-confidence edges must not spend an extra LLM call"
    assert written == 1


def test_reject_band_never_writes_and_skips_the_check(mem):
    t0 = int(time.time()) - 100
    e0 = mem.add_event(text="User decided to migrate the database", session_id="s1", timestamp=t0)
    e1 = mem.add_event(text="Exported the database schema for migration", session_id="s1", timestamp=t0 + 5)

    mem._llm = _fake_llm_proposing(0.5)  # below the reject threshold

    with patch.object(CausalMemory, "_verify_necessity") as mock_verify:
        written = mem._discover_in_window([e0, e1], proposer="test")
        assert not mock_verify.called
    assert written == 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
