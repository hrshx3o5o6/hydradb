"""
Unit tests for discovery_core.py -- pure functions, no live node or LLM
required. Run: pytest test_discovery_core.py -v
"""

from dataclasses import dataclass

import pytest

from causal_memory import discovery_core as dc


@dataclass
class _E:
    session_id: str
    text: str
    event_type: str = "fact"


# --------------------------------------------------------------- windowing

def test_temporal_windows_returns_single_window_when_small():
    events = [_E("s", f"e{i}") for i in range(5)]
    windows = dc.temporal_windows(events, size=18, overlap=6)
    assert windows == [events]


def test_temporal_windows_splits_and_overlaps_when_large():
    events = [_E("s", f"e{i}") for i in range(40)]
    windows = dc.temporal_windows(events, size=18, overlap=6)
    assert len(windows) > 1
    # consecutive windows share the overlap region
    assert windows[0][-6:] == windows[1][:6]
    # every event appears in at least one window
    covered = {id(e) for w in windows for e in w}
    assert covered == {id(e) for e in events}


def test_temporal_windows_empty_input():
    assert dc.temporal_windows([], size=18, overlap=6) == []


# ------------------------------------------------------------------ dedup

def test_dedup_by_text_keeps_first_occurrence_drops_repeats():
    events = [
        _E("s1", "ran tool X"),
        _E("s1", "ran tool X"),  # exact repeat, same session -> dropped
        _E("s1", "ran tool Y"),
        _E("s2", "ran tool X"),  # different session, same text -> kept
    ]
    uniq = dc.dedup_by_text(events)
    assert len(uniq) == 3
    assert uniq[0].text == "ran tool X" and uniq[0].session_id == "s1"
    assert uniq[1].text == "ran tool Y"
    assert uniq[2].session_id == "s2"


# --------------------------------------------------------- action-action

def test_action_action_pair_suppressed():
    assert dc.is_action_action_pair("action", "action") is True


@pytest.mark.parametrize("src,dst", [("fact", "action"), ("action", "fact"), ("fact", "fact")])
def test_non_action_action_pair_not_suppressed(src, dst):
    assert dc.is_action_action_pair(src, dst) is False


# ------------------------------------------------------------- overlap

def test_overlap_score_high_for_grounded_causal_pair():
    src = "User decided to migrate the database to Postgres"
    mech = "migration decision triggers the schema export step"
    dst = "Exported the database schema for Postgres migration"
    score = dc.overlap_score(src, mech, dst)
    assert score > 0.3
    assert dc.passes_overlap_check(src, mech, dst)


def test_overlap_score_low_for_hallucinated_unrelated_pair():
    src = "User asked what the weather is like today"
    mech = "curiosity about weather causes a mood change"
    dst = "Deployed the new authentication service to production"
    score = dc.overlap_score(src, mech, dst)
    assert score < dc.OVERLAP_REJECT_THRESHOLD
    assert not dc.passes_overlap_check(src, mech, dst)


def test_overlap_score_zero_when_either_side_empty():
    assert dc.overlap_score("", "", "some text here") == 0.0
    assert dc.overlap_score("some text here", "", "") == 0.0


# --------------------------------------------------------- confidence gate

@pytest.mark.parametrize("conf,expected", [
    (0.0, "reject"),
    (0.69, "reject"),
    (0.70, "verify"),
    (0.77, "verify"),
    (0.849, "verify"),
    (0.85, "accept"),
    (1.0, "accept"),
])
def test_route_confidence_bands(conf, expected):
    assert dc.route_confidence(conf) == expected


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
