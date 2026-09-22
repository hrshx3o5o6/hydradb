"""
Token-metered three-way retrieval benchmark: raw_context vs vector_rag
(real dense-vector cosine retrieval (local sentence-transformers - see llm_client.py for why Gemini/OpenAI embeddings were unavailable) - the harder baseline than the
earlier BM25 run) vs hydradna (causal subgraph retrieval), on real LoCoMo
QA.

The previous benchmark in this repo (benchmarks/results/locomo_subset_fix/)
only compared hydradna against BM25 + no-memory, and never measured tokens -
so the actual "does a causal graph save tokens vs a real RAG system"
question this project is built on had never been measured, anywhere (per
this project's own research phase). This is that measurement.

The `mem_bench` package the old benchmark depended on is gone (editable
install pointing at a since-cleaned-up temp directory) - this benchmark is
built fresh and does not depend on it.

Usage:
  GEMINI_API_KEY=... python run_benchmark.py \
      --hydradb-url http://localhost:28444 --admin-url http://localhost:29090 \
      --per-conv-n 25 --seed 42 --output ../results/locomo_v2_token_metered
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import load_conversations, build_manifest  # noqa: E402
from judge import judge  # noqa: E402
from llm_client import Usage  # noqa: E402
import strategies  # noqa: E402

logger = logging.getLogger("run_benchmark")

TOKEN = "local-dev-auth-token-32-characters-long"


def run(args) -> None:
    convs = load_conversations(args.dataset)
    subset_convs = convs[: args.n_conversations]
    built = build_manifest(subset_convs, args.per_conv_n, args.seed)
    samples = built["samples"]
    manifest = built["manifest"]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("subset: %d samples across %d conversations", len(samples), len(subset_convs))

    conv_by_id = {c["sample_id"]: c for c in subset_convs}
    strategy_names = args.strategies.split(",")

    all_results = {name: [] for name in strategy_names}
    one_time_usage = {name: Usage() for name in strategy_names}
    setup_seconds = {name: 0.0 for name in strategy_names}
    judge_usage = Usage()

    for conv_id, sample in conv_by_id.items():
        conv = sample["conversation"]  # sample also carries qa/event_summary/etc - not needed here
        conv_samples = [s for s in samples if s.conv_id == conv_id]
        logger.info("=== conversation %s: %d questions ===", conv_id, len(conv_samples))

        states = {}
        if "raw_context" in strategy_names:
            t0 = time.monotonic()
            state, u = strategies.raw_context_setup(conv)
            states["raw_context"] = state
            one_time_usage["raw_context"].prompt_tokens += u.prompt_tokens
            one_time_usage["raw_context"].completion_tokens += u.completion_tokens
            setup_seconds["raw_context"] += time.monotonic() - t0
            logger.info("  raw_context: transcript %d chars", len(state))

        if "vector_rag" in strategy_names:
            t0 = time.monotonic()
            state, u = strategies.vector_rag_setup(conv)
            states["vector_rag"] = state
            one_time_usage["vector_rag"].prompt_tokens += u.prompt_tokens
            one_time_usage["vector_rag"].completion_tokens += u.completion_tokens
            setup_seconds["vector_rag"] += time.monotonic() - t0
            logger.info("  vector_rag: embedded %d chunks (%d tokens)", len(state.chunks), u.prompt_tokens)

        if "hydradna" in strategy_names:
            t0 = time.monotonic()
            state, u = strategies.hydradna_setup(
                conv, conv_prefix=conv_id, url=args.hydradb_url,
                token=TOKEN, admin_url=args.admin_url,
            )
            states["hydradna"] = state
            one_time_usage["hydradna"].prompt_tokens += u.prompt_tokens
            one_time_usage["hydradna"].completion_tokens += u.completion_tokens
            setup_seconds["hydradna"] += time.monotonic() - t0
            logger.info("  hydradna: discovery cost %d prompt + %d completion tokens", u.prompt_tokens, u.completion_tokens)

        for si, s in enumerate(conv_samples):
            for name in strategy_names:
                try:
                    if name == "raw_context":
                        ans, usage, secs = strategies.raw_context_answer(states["raw_context"], s.question)
                    elif name == "vector_rag":
                        ans, usage, secs = strategies.vector_rag_answer(states["vector_rag"], s.question)
                    elif name == "hydradna":
                        ans, usage, secs = strategies.hydradna_answer(states["hydradna"], s.question)
                    else:
                        raise ValueError(name)
                    correct, jusage = judge(s.question, s.answer, ans)
                    judge_usage.prompt_tokens += jusage.prompt_tokens
                    judge_usage.completion_tokens += jusage.completion_tokens
                except Exception as e:
                    logger.warning("  [%s] q%d failed: %s", name, si, e)
                    ans, usage, secs, correct = "", Usage(), 0.0, False

                all_results[name].append({
                    "conv_id": conv_id, "qa_idx": s.qa_idx, "category": s.category,
                    "question": s.question, "gold": s.answer, "answer": ans,
                    "correct": correct, "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens, "seconds": secs,
                })
                logger.info("  [%s] q%d %s (%d+%d tok, %.2fs)", name, si,
                            "OK" if correct else "MISS", usage.prompt_tokens,
                            usage.completion_tokens, secs)
                with (out_dir / f"{name}.progress.jsonl").open("a") as f:
                    f.write(json.dumps(all_results[name][-1]) + "\n")

    # ---- aggregate
    comparison = {"strategies": {}, "judge_total_tokens": asdict(judge_usage)}
    for name in strategy_names:
        rs = all_results[name]
        n = len(rs)
        acc = sum(1 for r in rs if r["correct"]) / n if n else 0.0
        mean_prompt = sum(r["prompt_tokens"] for r in rs) / n if n else 0.0
        mean_completion = sum(r["completion_tokens"] for r in rs) / n if n else 0.0
        mean_seconds = sum(r["seconds"] for r in rs) / n if n else 0.0
        by_cat = {}
        cats = sorted({r["category"] for r in rs})
        for c in cats:
            crs = [r for r in rs if r["category"] == c]
            by_cat[str(c)] = sum(1 for r in crs if r["correct"]) / len(crs) if crs else 0.0
        comparison["strategies"][name] = {
            "n_queries": n,
            "qa_accuracy": acc,
            "qa_accuracy_by_category": by_cat,
            "mean_per_query_prompt_tokens": mean_prompt,
            "mean_per_query_completion_tokens": mean_completion,
            "mean_per_query_total_tokens": mean_prompt + mean_completion,
            "mean_per_query_seconds": mean_seconds,
            "one_time_setup_tokens": asdict(one_time_usage[name]),
            "one_time_setup_seconds": setup_seconds[name],
        }
        (out_dir / f"{name}.summary.json").write_text(
            json.dumps(comparison["strategies"][name], indent=2)
        )

    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    logger.info("done. results in %s", out_dir)
    print(json.dumps(comparison, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="/tmp/mb_data/locomo10.json")
    ap.add_argument("--n-conversations", type=int, default=2)
    ap.add_argument("--per-conv-n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strategies", default="raw_context,vector_rag,hydradna")
    ap.add_argument("--hydradb-url", default="http://localhost:28444")
    ap.add_argument("--admin-url", default="http://localhost:29090")
    ap.add_argument("--output", default=str(Path(__file__).resolve().parents[1] / "results" / "locomo_v2_token_metered"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
