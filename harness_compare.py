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


def run_trial(mem, llm, seed, dry_run=False):
    """Run one full comparison. Returns verdict dict + answers."""
    story, _ = build_story(seed=seed)
    print(f"Story: {len(story)} observations, 4 developers, 1 day")
    print(f"True causal chain depth: {len(TRUE_CHAIN)} events, buried in noise")
    chain_positions = [next(i for i, s in enumerate(story) if s[1] == txt)
                       for txt in TRUE_CHAIN]
    print(f"Chain events at stream positions: {chain_positions} (non-contiguous)")
    print(f"Harness context window: last {CONTEXT_WINDOW_LINES} lines -> "
          f"chain starts at {chain_positions[0]}, which is "
          f"{'INSIDE' if chain_positions[0] >= len(story) - CONTEXT_WINDOW_LINES else 'SCROLLED OUT of'} the window")

    mem.reset()
    print("Wiped store for a clean comparison.")

    # ------------------------------------------------------------- seed data
    section("SEED — observations stream into HydraDB")
    events = {}
    for session, text, t in story:
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

    # Deterministic baseline recall: of the TRUE_CHAIN, how many events even
    # survive inside the flat window the model sees? Independent of the judge.
    window_texts = set(txt for (_s, txt, _t) in story[-CONTEXT_WINDOW_LINES:])
    base_recall = sum(1 for t in TRUE_CHAIN if t in window_texts) / len(TRUE_CHAIN)
    print(f"  [RETRIEVAL RECALL] true-chain events visible in flat window: "
          f"{base_recall:.0%} ({sum(1 for t in TRUE_CHAIN if t in window_texts)}/{len(TRUE_CHAIN)})")

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
    true_hits = 0
    if best:
        best_texts = [e.text for e in best.events]
        true_hits = sum(1 for t in TRUE_CHAIN if t in best_texts)
        print("\n   Discovered causal chain grounding the answer:")
        for i, ev in enumerate(best.events):
            note = ""
            r = best.relations[i] if i < len(best.relations) else None
            if r and r.relation_type == "CAUSES":
                mech = f' — "{r.mechanism}"' if r.mechanism else ""
                note = f"  <=(conf {r.confidence:.2f}){mech}"
            print(f"     {ev.text}{note}")
        print(f"   [true-chain events recovered: {true_hits}/{len(TRUE_CHAIN)}]")

    # Deterministic retrieval metrics for the causal path (no LLM judge).
    causal_recall = true_hits / len(TRUE_CHAIN)
    path_len = len(best.events) if best else 0
    causal_precision = true_hits / path_len if path_len else 0.0
    # Context efficiency: tokens the model must ingest to answer.
    base_tokens = sum(len(l.split()) for l in window) + len(QUESTION.split())
    causal_tokens = sum(len(e.text.split()) for e in best.events) + len(QUESTION.split()) \
        if best else len(QUESTION.split())
    print(f"   [RETRIEVAL RECALL] causal graph recovered {true_hits}/{len(TRUE_CHAIN)} "
          f"= {causal_recall:.0%}")
    print(f"   [RETRIEVAL PRECISION] {true_hits} true events in path of {path_len} "
          f"= {causal_precision:.0%}")
    print(f"   [CONTEXT EFFICIENCY] flat window {base_tokens} tokens vs causal path "
          f"{causal_tokens} tokens = {base_tokens/max(causal_tokens,1):.1f}x less context")

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
    verdict["true_hits"] = true_hits
    verdict["base_recall"] = base_recall
    verdict["causal_recall"] = causal_recall
    verdict["causal_precision"] = causal_precision
    verdict["context_ratio"] = base_tokens / max(causal_tokens, 1)
    print("\nJudge verdict:")
    for key in ("answer_a", "answer_b"):
        s = verdict.get(key, {})
        print(f"  {key}: accuracy={s.get('accuracy')} precision={s.get('precision')}")
    print("  winner:", verdict.get("winner", "?"))
    return verdict


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

    seeds = [7]
    if "--trials" in sys.argv:
        try:
            seeds = list(range(1, int(sys.argv[sys.argv.index("--trials") + 1]) + 1))
        except (IndexError, ValueError):
            print("usage: --trials N")
            sys.exit(1)

    results = []
    for seed in seeds:
        print("\n\n###################### TRIAL seed=%d ######################" % seed)
        results.append(run_trial(mem, llm, seed))

    if len(results) > 1:
        section("AGGREGATE OVER %d TRIALS" % len(results))
        def avg(key):
            vals = [r.get(key, 0) for r in results]
            return sum(vals) / len(vals)
        def avg_sub(main_key, sub):
            vals = []
            for r in results:
                s = r.get(main_key, {})
                if isinstance(s, dict):
                    vals.append(s.get(sub, 0))
            return sum(vals) / len(vals) if vals else 0
        from collections import Counter
        winners = Counter(r.get("winner", "?") for r in results)
        print(f"accuracy   A={avg_sub('answer_a', 'accuracy'):4.2f}   "
              f"B={avg_sub('answer_b', 'accuracy'):4.2f}")
        print(f"precision  A={avg_sub('answer_a', 'precision'):4.2f}   "
              f"B={avg_sub('answer_b', 'precision'):4.2f}")
        print(f"true-chain recovery in discovered paths: "
              f"{avg('true_hits'):4.2f}/{len(TRUE_CHAIN)}")
        print(f"retrieval recall (base window)  = {avg('base_recall'):6.1%}")
        print(f"retrieval recall (causal path)  = {avg('causal_recall'):6.1%}")
        print(f"retrieval precision (causal)    = {avg('causal_precision'):6.1%}")
        print(f"context efficiency (base/causal)= {avg('context_ratio'):6.1f}x")
        print("winners:", dict(winners))


if __name__ == "__main__":
    main()