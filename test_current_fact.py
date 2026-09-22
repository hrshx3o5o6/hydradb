"""
Regression test for find_current_fact (item 2): it must walk OVERWRITES
chains, not just pick the newest same-topic event by timestamp.

Run:
    HYDRADB_URL=http://localhost:18444 pytest test_current_fact.py -v
"""

import os
import time
import pytest

from causal_memory.memory import CausalMemory

HYDRADB_URL = os.environ.get("HYDRADB_URL", "http://localhost:18444")


@pytest.fixture
def mem():
    m = CausalMemory(url=HYDRADB_URL)
    m.reset()
    yield m
    m.reset()


def test_current_fact_follows_overwrite_not_just_recency(mem):
    t0 = int(time.time()) - 100
    a = mem.add_event(text="lives in SF", session_id="s1", topic="location", timestamp=t0)
    b = mem.add_event(text="lives in NYC", session_id="s1", topic="location", timestamp=t0 + 10)
    mem.add_overwrite(old_event_id=a.id, new_event_id=b.id, reason="moved")

    current = mem.find_current_fact("location")
    assert current is not None
    assert current.id == b.id


def test_current_fact_ignores_a_superseded_event_even_if_it_is_re_logged_later(mem):
    """Adversarial ordering: a stale duplicate of the overwritten fact gets
    logged with a timestamp AFTER its own successor exists. It's still the
    overwritten one -- recency alone must not win."""
    t0 = int(time.time()) - 100
    a = mem.add_event(text="lives in SF", session_id="s1", topic="location", timestamp=t0)
    b = mem.add_event(text="lives in NYC", session_id="s1", topic="location", timestamp=t0 + 10)
    mem.add_overwrite(old_event_id=a.id, new_event_id=b.id, reason="moved")

    stale_dup = mem.add_event(text="lives in SF (dup)", session_id="s1", topic="location", timestamp=t0 + 20)
    mem.add_overwrite(old_event_id=stale_dup.id, new_event_id=b.id, reason="dup also overwritten")

    current = mem.find_current_fact("location")
    assert current is not None
    assert current.id == b.id


def test_current_fact_returns_none_for_unknown_topic(mem):
    assert mem.find_current_fact("nonexistent-topic-xyz") is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
