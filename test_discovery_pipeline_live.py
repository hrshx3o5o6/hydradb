"""
Live integration test for the discovery pipeline (items 3/5/8): real LLM
calls, real HydraDB node. Confirms the deterministic overlap check +
confidence-band gate actually change what gets written, not just that the
pure functions in discovery_core behave in isolation.

Run:
    HYDRADB_URL=http://localhost:18444 pytest test_discovery_pipeline_live.py -v -s
Requires GEMINI_API_KEY or OPENAI_API_KEY.
"""

import os
import time
import pytest

from causal_memory.memory import CausalMemory

HYDRADB_URL = os.environ.get("HYDRADB_URL", "http://localhost:18444")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="no LLM credentials in env",
)


@pytest.fixture
def mem():
    m = CausalMemory(url=HYDRADB_URL)
    m.reset()
    yield m
    m.reset()


def test_extract_causality_chains_related_events_and_ignores_unrelated_one(mem):
    t0 = int(time.time()) - 1000
    mem.observe("User decided to migrate the database to Postgres", session_id="s1", timestamp=t0)
    mem.observe("Exported the current schema for the Postgres migration", session_id="s1", timestamp=t0 + 10)
    mem.observe("Ran the schema import into the new Postgres instance", session_id="s1", timestamp=t0 + 20)
    mem.observe("Verified row counts matched between old and new database", session_id="s1", timestamp=t0 + 30)
    mem.observe("User asked what the weather is like today", session_id="s1", timestamp=t0 + 40)

    written = mem.extract_causality(window_events=10, session_id="s1")
    assert written >= 2, "expected the migration chain to produce at least 2 edges"

    rows = mem.client.execute(
        "MATCH (a:Event)-[r:CAUSES]->(b:Event) RETURN a.id, b.id, r.confidence, r.proposed_by"
    )
    assert len(rows) == written
    for row in rows:
        # every written edge has provenance recorded from this discovery run
        assert row["r.proposed_by"]
        assert "discovery:s1" in row["r.proposed_by"]

    events = mem._recent_events(limit=10)
    weather = next(e for e in events if "weather" in e.text)
    touches_weather = any(
        row["a.id"] == weather.id or row["b.id"] == weather.id for row in rows
    )
    assert not touches_weather, "unrelated weather event should not be chained in"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
