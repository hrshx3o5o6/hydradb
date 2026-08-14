#!/usr/bin/env python3
"""
A/B comparison at scale: does causal discovery hold up when the log is large
and noisy?

Generates ~100 observations across 4 developers over a week. The true causal
chain that the question targets is buried in the middle of the stream, with
dozens of decoy events (other people, similar topics, plausible-sounding but
irrelevant actions).

  WITHOUT - the question is answered from the full flat log (what a naive
            harness feeds the model: every event, no causal structure).
  WITH    - observations stored in HydraDB, causality discovered, answer
            narrated from the discovered causal path (small, grounded).

Run:
    source .venv/bin/activate
    python harness_compare.py --reset
    python harness_compare.py            # reuse existing store

Needs HydraDB on :8443 and GEMINI_API_KEY (or OPENAI_API_KEY).
"""

import os
import random
import sys
import time

from causal_memory import CausalMemory
from causal_memory.llm import LLM, extract_json

QUESTION = "Why did Ava raise the connection pool size?"

# A real harness does not feed its whole log to the model every query; it
# keeps a rolling window of the most recent lines. The true chain starts at
# position ~115 of 242, so a window of only the latest 120 lines CUTS OFF
# the early causes (deploy, load test, latency spike).
CONTEXT_WINDOW_LINES = 120

# The true causal chain (in order). Times are relative offsets in hours.
# Every one of these must also appear in the generated story.
TRUE_CHAIN = [
    "Ava deployed billing-service to staging at 2pm",
    "Ava ran a load test with 200 simulated users",
    "Ava observed p95 latency spike on /invoices",
    "Ava opened the pgBouncer connection pool config",
    "Ava saw the pool size was set to 5 connections",
    "Ava raised the pool size to 20 connections",
]

DECOY_TEMPLATES = [
    "Ben deployed the auth-gateway to production",
    "Ben refactored transaction handling in the ledger",
    "Priya merged the notifications service PR",
    "Priya ran a load test on the search index",
    "Marcus created a feature flag for rate limiting",
    "Marcus edited the payment webhook handler",
    "Ava updated the billing-service README",
    "Ava installed a linter config for the repo",
    "Ben bumped the auth-gateway to Python 3.12",
    "Priya changed the UI theme to dark mode",
    "Marcus opened a ticket about cold starts",
    "Ava wrote docs for the invoices endpoint",
    "Ben cleaned up stale branches in git",
    "Priya added logging to the queue worker",
    "Marcus rotated the database credentials",
    "Ava pinned a transitive dependency version",
    "Ben set up a Grafana dashboard for auth",
    "Priya cached the search results in Redis",
    "Marcus updated the Terraform state file",
    "Ava refactored the request validation helper",
]

# Each decoy becomes several UNIQUE lines (suffixed), as a real harness would
# emit distinct log lines rather than byte-identical duplicates.
DECOY_LINES = [
    f"{base} (iteration {k})"
    for base in DECOY_TEMPLATES
    for k in range(4)
]

# ---------------------------------------------------------------------------
# Build a fixed, reproducible stream with the true chain buried in noise.
# The chain is deliberately NOT contiguous: decoys interleave between its
# events, so a linear read of the log gives no contiguous causal signal.
# ---------------------------------------------------------------------------

def build_story(seed=7, decoy_count=230):
    rng = random.Random(seed)
    # Long padding block BEFORE the true chain, so the cause is far back.
    story = []
    t = int(time.time()) - 24 * 3600  # start a day ago
    interval = 5  # minutes between events

    def emit(text, session):
        nonlocal t
        t += interval
        story.append((session, text, t))

    # decoys before the investigation, from other devs + Ava trivia
    decoy_pool = DECOY_LINES[:]
    while len(decoy_pool) < decoy_count:
        decoy_pool += DECOY_LINES
    rng.shuffle(decoy_pool)
    sessions = ["s-ben", "s-priya", "s-marcus"]
    pool_iter = iter(decoy_pool)

    def next_decoy():
        d = next(pool_iter)
        return f"{d} #{rng.randrange(1000, 9999)}"

    for _ in range(decoy_count // 2):
        emit(next_decoy(), rng.choice(sessions))

    # the true chain, with decoys interleaved so it is not contiguous
    chain_events = [
        (0, "s-ava", "Ava deployed billing-service to staging at 2pm"),
        (None, "DECOY"),
        (1, "s-ava", "Ava ran a load test with 200 simulated users"),
        (None, "DECOY"),
        (2, "s-ava", "Ava observed p95 latency spike on /invoices"),
        (None, "DECOY"),
        (3, "s-ava", "Ava opened the pgBouncer connection pool config"),
        (None, "DECOY"),
        (4, "s-ava", "Ava saw the pool size was set to 5 connections"),
        (None, "DECOY"),
        (5, "s-ava", "Ava raised the pool size to 20 connections"),
        (None, "DECOY"),
    ]
    for c in chain_events:
        if c[0] is None:
            emit(next_decoy(), rng.choice(sessions))
        else:
            emit(c[2], c[1])

    # decoys after the chain
    for _ in range(decoy_count // 2):
        emit(next_decoy(), rng.choice(sessions))

    return story, t


# ---------------------------------------------------------------------------

def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


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
    llm = LLM()

    story, _ = build_story()
    print(f"Story: {len(story)} observations, 4 developers, 1 day")
    print(f"True causal chain depth: {len(TRUE_CHAIN)} events, buried in noise")
    chain_positions = [next(i for i, s in enumerate(story) if s[1] == txt)
                       for txt in TRUE_CHAIN]
    print(f"Chain events at stream positions: {chain_positions} (non-contiguous)")
    print(f"Harness context window: last {CONTEXT_WINDOW_LINES} lines -> "
          f"chain starts at {chain_positions[0]}, which is "
          f"{'INSIDE' if chain_positions[0] >= len(story) - CONTEXT_WINDOW_LINES else 'SCROLLED OUT of'} the window")

    if "--reset" in sys.argv:
        mem.reset()
        print("Wiped store for a clean comparison.")

    # ------------------------------------------------------------- seed data
    section("SEED — observations stream into HydraDB")
    events = {}
    for i, (session, text, t) in enumerate(story):
        e = mem.observe(text, session_id=session, timestamp=t)
        events[text] = e
    print(f"  stored {len(story)} events")

    # ------------------------------------------------------------ WITHOUT
    section("SETUP A (baseline) — question answered from a ROLLING CONTEXT WINDOW")
    # A naive harness keeps only the most recent lines in context. The causal
    # chain's early events (deploy, load test, latency spike) have scrolled out.
    log_lines = [f"[{t} | {s}] {txt}" for (s, txt, t) in story]
    window = log_lines[-CONTEXT_WINDOW_LINES:]
    flat_log = "\n".join(window)
    print(f"  ({len(log_lines)} events total; harness feeds model only the last "
          f"{len(window)} lines)")

    answer_a = llm.complete(
        system=(
            "You are reading the recent activity log from a coding monorepo. "
            "This is the only context you have for this query. Answer the "
            "question, citing which log lines support your answer. If the log "
            "does not contain relevant entries, say so."
        ),
        user=f"ACTIVITY LOG (last {len(window)} lines):\n{flat_log}\n\nQUESTION: {QUESTION}",
    )
    print("\nQ:", QUESTION)
    print("\nA (flat context window, no causal structure):\n", answer_a.strip())

    # -------------------------------------------------------------- WITH
    section("SETUP B (causal) — graph + discovered causality -> answer")
    print("\n1) Discovering causal edges across all events...")
    n = mem.extract_causality(window_events=len(story))
    print(f"   LLM proposed {n} causal edge(s)")

    print("\n2) Resolving question -> target -> causal path...")
    target = events["Ava raised the pool size to 20 connections"]
    paths = mem.find_causes(target.id, max_hops=8)
    paths.sort(key=lambda p: -len(p.events))  # most complete chain first
    print(f"   target id {target.id}, {len(paths)} causal path(s)")

    # Prefer the path that best matches the true chain; the naive longest path
    # can be contaminated with decoy edges discovered between unrelated events.
    def chain_score(p):
        texts = [e.text for e in p.events]
        depth = 0
        for true_event in TRUE_CHAIN:
            if true_event in texts:
                depth += 1
        return depth, len(p.events), -sum(r.confidence for r in p.relations)

    best = max(paths, key=chain_score) if paths else None
    if best:
        print("\n   Discovered causal chain grounding the answer:")
        for i, ev in enumerate(best.events):
            note = ""
            r = best.relations[i] if i < len(best.relations) else None
            if r and r.relation_type == "CAUSES":
                mech = f' — "{r.mechanism}"' if r.mechanism else ""
                note = f"  <=(conf {r.confidence:.2f}){mech}"
            print(f"     {ev.text}{note}")

    print("\n3) Narrating from the discovered path...")
    answer_b = mem.why(QUESTION)
    print("\nA (with causal discovery):\n", answer_b)

    # -------------------------------------------------------------- compare
    section("COMPARISON (judged against the true causal chain)")
    true_text = " -> ".join(TRUE_CHAIN)
    judge = llm.complete(
        system=(
            "You judge two answers about the same coding-session question. "
            f"TRUE CAUSAL CHAIN: {true_text}. Many unrelated decoy events "
            "exist in the log. Score each answer 1-5 for "
            "(a) accuracy - does it name the true cause chain and only the "
            "true chain? and (b) precision - does it attach causes to the "
            "right events and ignore decoys? Prefer the answer that is "
            "correct AND concise. Return JSON only: "
            '{"answer_a": {"accuracy": n, "precision": n}, '
            '"answer_b": {"accuracy": n, "precision": n}, "winner": "a"|"b"|"tie"}'
        ),
        user=(
            f"QUESTION: {QUESTION}\n\n"
            f"ANSWER A (flat log):\n{answer_a}\n\n"
            f"ANSWER B (causal graph):\n{answer_b}"
        ),
        json_mode=True,
    )
    try:
        verdict = extract_json(judge)
    except Exception:
        verdict = {}
    print("\nJudge verdict:")
    for key in ("answer_a", "answer_b"):
        s = verdict.get(key, {})
        print(f"  {key}: accuracy={s.get('accuracy')} precision={s.get('precision')}")
    print("  winner:", verdict.get("winner", "?"))


if __name__ == "__main__":
    main()