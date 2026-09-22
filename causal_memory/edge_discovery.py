"""
Automated causal-edge discovery for the live causal-memory graph.

The plugin records events (user messages, tool actions) but does not link
them: edges are *discovered* here. This daemon watches the graph for new
events and, whenever new nodes appear, runs LLM causal discovery over the
recent window. No dependency found is a fine outcome — nothing is written
and we wait for the next new event.

Mode:
    --watch  poll forever (default), running discovery each time new events land
    --once   single pass, then exit (for loop-tick integration)

Edge writes are idempotent (MATCH+SET on repeat proposals, see
CausalMemory.add_causal_relation), so re-digesting overlapping windows is
safe. Discovery state (last-processed timestamp) is a graph-backed
watermark, ONE PER SESSION (CausalMemory.get_watermark/set_watermark) --
not a local JSON file. A local file breaks the moment more than one daemon
process (or agent session) writes events concurrently: two processes race
on the same file, and neither's watermark reflects what the other already
processed. Sharding by session_id also means one session's backlog doesn't
block another's from being picked up.

Run:
    source .venv/bin/activate
    HYDRADB_URL=http://localhost:18444 python -m causal_memory.edge_discovery --watch
"""

import argparse
import os
import sys
import time
from collections import defaultdict

from causal_memory.memory import CausalMemory

DEFAULT_URL = "http://localhost:18444"
POLL_SECONDS = 5
DEFAULT_WINDOW = 30          # events fed to the LLM per discovery pass, per session


def run_once(mem: CausalMemory, window: int) -> int:
    """
    One discovery pass: group recent events by session, and for each session
    with events newer than its graph-backed watermark, run discovery scoped
    to that session and advance its watermark to the newest timestamp seen.

    Returns total edges written across all sessions.
    """
    try:
        events = mem._recent_events(limit=max(window * 4, 200))
    except Exception as e:
        print("[edge_discovery] read failed: %s" % e, file=sys.stderr)
        return 0

    by_session = defaultdict(list)
    for e in events:
        by_session[e.session_id].append(e)

    total_written = 0
    for session_id, session_events in by_session.items():
        watermark = mem.get_watermark(session_id)
        new_events = [e for e in session_events if watermark is None or e.timestamp > watermark]
        if not new_events:
            continue

        print("[edge_discovery] session=%s: %d new event(s), running causal "
              "discovery (window=%d)…" % (session_id, len(new_events), window), flush=True)
        try:
            written = mem.extract_causality(window_events=window, session_id=session_id)
        except Exception as e:
            print("[edge_discovery] discovery failed for session=%s: %s" % (session_id, e), file=sys.stderr)
            written = 0

        newest = max(e.timestamp for e in session_events)
        mem.set_watermark(session_id, newest)
        total_written += written
        print("[edge_discovery] session=%s: %d edge(s) written" % (session_id, written), flush=True)

    return total_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="poll forever")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--url", default=os.environ.get("HYDRADB_URL", DEFAULT_URL))
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = ap.parse_args()

    mem = CausalMemory(url=args.url)

    while True:
        run_once(mem, args.window)
        if args.once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
