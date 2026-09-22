"""
Regression tests for the graph-backed discovery watermark (item 6):
sharded by session, survives process restart, no local-file race.

Run:
    HYDRADB_URL=http://localhost:18444 pytest test_edge_discovery_daemon.py -v
"""

import os
import time
import pytest

from causal_memory.memory import CausalMemory
from causal_memory.edge_discovery import run_once

HYDRADB_URL = os.environ.get("HYDRADB_URL", "http://localhost:18444")

pytestmark = pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="no LLM credentials in env",
)


@pytest.fixture
def mem():
    m = CausalMemory(url=HYDRADB_URL)
    m.reset()
    m.client.execute("MATCH (w:DiscoveryWatermark) DETACH DELETE w")
    yield m
    m.reset()
    m.client.execute("MATCH (w:DiscoveryWatermark) DETACH DELETE w")


def test_watermark_survives_a_fresh_causalmemory_instance(mem):
    """Simulates a daemon restart: a brand-new CausalMemory (new client, same
    node) must read back the same watermark, not start over."""
    mem.set_watermark("s1", 12345)
    fresh = CausalMemory(url=HYDRADB_URL)
    assert fresh.get_watermark("s1") == 12345


def test_two_sessions_do_not_clobber_each_others_watermark(mem):
    t0 = int(time.time()) - 500
    mem.observe("decision A for session one", session_id="daemon-s1", timestamp=t0)
    mem.observe("consequence A for session one", session_id="daemon-s1", timestamp=t0 + 5)
    mem.observe("decision B for session two", session_id="daemon-s2", timestamp=t0 + 1)
    mem.observe("consequence B for session two", session_id="daemon-s2", timestamp=t0 + 6)

    run_once(mem, window=10)

    wm1 = mem.get_watermark("daemon-s1")
    wm2 = mem.get_watermark("daemon-s2")
    assert wm1 is not None and wm2 is not None
    assert wm1 != wm2 or wm1 == wm2  # both set independently; just assert both exist
    assert wm1 == t0 + 5
    assert wm2 == t0 + 6

    # A second pass with no new events for either session should be a no-op
    # (watermarks already at the newest timestamp for each).
    written_again = run_once(mem, window=10)
    assert written_again == 0
    assert mem.get_watermark("daemon-s1") == wm1
    assert mem.get_watermark("daemon-s2") == wm2


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
