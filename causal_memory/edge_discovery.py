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

Edge writes are idempotent (MERGE), so re-digesting overlapping windows is
safe. Discovery state (seen event ids) is persisted to a small JSON file so
restarting the daemon does not re-digest the whole history.

Run:
    source .venv/bin/activate
    HYDRADB_URL=http://localhost:8443 python -m causal_memory.edge_discovery --watch
"""

import argparse
import json
import os
import sys
import time

from causal_memory.memory import CausalMemory

DEFAULT_URL = "http://localhost:8443"
STATE_FILE = os.environ.get("EDGE_DISCOVERY_STATE", "/tmp/sgk-edge-discovery.json")
POLL_SECONDS = 5
DEFAULT_WINDOW = 30          # events fed to the LLM per discovery pass
MAX_KNOWN_IDS = 5000


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f).get("known_ids", [])
    except (OSError, ValueError):
        return []


def save_state(known_ids):
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"known_ids": known_ids[-MAX_KNOWN_IDS:]}, f)
    os.replace(tmp, STATE_FILE)


def run_once(mem, known_ids, window):
    """One discovery pass over events newer than the watermark. Returns stats."""
    try:
        events = mem._recent_events(limit=window)
    except Exception as e:
        print("[edge_discovery] read failed: %s" % e, file=sys.stderr)
        return known_ids, 0

    known = set(known_ids)
    new_events = [e for e in events if e.id not in known]
    if not new_events:
        return known_ids, 0

    ids = known_ids + [e.id for e in new_events]
    print("[edge_discovery] %d new event(s), running causal discovery "
          "(window=%d)…" % (len(new_events), window), flush=True)
    try:
        written = mem.extract_causality(window_events=window)
    except Exception as e:
        print("[edge_discovery] discovery failed: %s" % e, file=sys.stderr)
        written = 0
    print("[edge_discovery] %d edge(s) written" % written, flush=True)
    return ids, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true", help="poll forever")
    ap.add_argument("--once", action="store_true", help="single pass then exit")
    ap.add_argument("--url", default=os.environ.get("HYDRADB_URL", DEFAULT_URL))
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = ap.parse_args()

    mem = CausalMemory(url=args.url)
    known_ids = load_state()

    while True:
        known_ids, _ = run_once(mem, known_ids, args.window)
        save_state(known_ids)
        if args.once:
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
