#!/usr/bin/env python3
"""
Live harness demo: developer works in an agent harness, observations stream
into HydraDB, causal edges are discovered, and "why" answers come back in
sentences.

Run:
    source .venv/bin/activate
    python harness_demo.py

Prereqs: HydraDB running on :8443 (docker), GEMINI_API_KEY set.
"""

import os
import sys
import time

from causal_memory import CausalMemory


def section(title):
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def main():
    token = os.environ.get("HYDRADB_TOKEN", "local-dev-auth-token-32-characters-long")
    mem = CausalMemory(
        url="http://localhost:8443",
        auth_token=token,
        admin_url="http://localhost:9090",
    )

    if not (mem.client.ready_check() or mem.client.health_check()):
        print("HydraDB not reachable. Start it first (docker).")
        sys.exit(1)
    print("HydraDB reachable. LLM:", "GEMINI" if os.environ.get("GEMINI_API_KEY") else "OPENAI")

    if "--reset" in sys.argv:
        mem.reset()
        print("Wiped all events + causal edges for a clean run.")

    session = f"harness-{int(time.time())}"

    section("PHASE 1 — developer works in a harness; observations stream in")
    story = [
        ("User opened the app for the first time", "onboarding"),
        ("User saw the empty dashboard", "onboarding"),
        ("User connected their GitHub account", "setup"),
        ("User imported their first repo", "setup"),
        ("User ran a search query", "usage"),
        ("User reported the search felt slow", "feedback"),
        ("User opened the docs page on indexing", "usage"),
        ("User asked why search is slow", "feedback"),
    ]
    events = []
    now = int(time.time())
    for i, (text, topic) in enumerate(story):
        e = mem.observe(text, session_id=session, topic=topic, timestamp=now + i)
        events.append(e)
        print(f"  observed #{i}: {text}")

    section("PHASE 2 — causal discovery: LLM proposes cause->effect edges")
    n = mem.extract_causality(session_id=session)
    print(f"  proposed {n} causal edge(s)")

    section("PHASE 3 — 'why' questions answered from the discovered graph")
    questions = [
        "Why is search slow?",
        "Why did the user connect their GitHub account?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        print("A:", mem.why(q))

    section("DONE")
    print(f"Events + causal edges persisted in HydraDB under session {session}")
    print("Re-run with a fresh session to see discovery build a new chain.")


if __name__ == "__main__":
    main()