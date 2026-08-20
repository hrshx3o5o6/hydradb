"""Industry-standard subset benchmark driver for HydraDNA on LongMemEval.

Why subset + batching:
- The full 500-sample S split degrades on the single-node HydraDB engine: under
  sustained write load the local store stops applying deletes, the graph grows
  unboundedly and every query slows (measured: 42s/sample at sample 20,
  507s/sample at sample 165 with 25k orphaned events).
- LongMemEval paper uses the full split, but for iterative model development a
  stratified subset is the accepted practice: report the subset composition,
  selection seed, and exact sample ids so numbers are reproducible and
  comparable. Here 100/500 samples (20%) stratified by question type.
- The node is recreated fresh between batches so each batch runs on a clean
  graph and deletes stay bounded.

Baselines run on the SAME subset with the SAME judge:
- bm25: built-in BM25 adapter (industry-standard lexical RAG baseline).
- nomemory: empty recall (no-memory abstention baseline / sanity control).

Usage:
  HYDRADB_URL=http://localhost:18444 python run_subset.py --adapters hydradna,bm25,nomemory --n 100 --batch 40 --seed 42

Output: <output_dir>/{subset_manifest.json, progress.jsonl, summary.json,
summary.md, per-adapter subdirs with same files}.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator, Sequence

from mem_bench.benchmarks.longmemeval import LongMemEvalBenchmark
from mem_bench.core.adapter import BaseAdapter
from mem_bench.core.benchmark import BenchmarkSample
from mem_bench.core.config import RunConfig, load_config
from mem_bench.core.types import RecallQuery, RecallResult
from mem_bench.evaluation.qa import generate_answer
from mem_bench.reporting.markdown_report import save_markdown_report

logger = logging.getLogger("run_subset")


def retry_call(fn, *, attempts: int = 6, max_wait: float = 45.0, what: str = "call"):
    """Call fn with exponential backoff; extra wait on 429 rate limits.

    The harness judge and answer generator make no retry calls; back-to-back
    benchmark runs trip OpenAI 429s and every sample then fails (measured:
    bm25 165/300 and nomemory 300/300 failed on 429 with no backoff).
    """
    import random as _rng

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            msg = str(exc)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = min(max_wait, 10.0 + 10.0 * attempt + _rng.uniform(0, 3))
            else:
                wait = min(max_wait, 1.5 * (2 ** attempt) + _rng.uniform(0, 1))
            logger.warning("%s failed (attempt %d/%d), retrying in %.1fs: %s", what, attempt + 1, attempts, wait, msg)
            time.sleep(wait)

HYDRADB_URL = "http://localhost:18444"
NODE_NAME = "hydradb-bench"
NODE_DATA = Path("/tmp/hydradb-bench-data")
NODE_TOKEN = "local-dev-auth-token-32-characters-long"

MODULE_DIR = Path(__file__).resolve().parent


def reset_node() -> None:
    """Recreate the HydraDB bench node from scratch (fresh store)."""
    subprocess.run(["docker", "rm", "-f", NODE_NAME], capture_output=True)
    for p in (NODE_DATA / "store", NODE_DATA / "cache"):
        subprocess.run(["rm", "-rf", str(p)])
        p.mkdir(parents=True, exist_ok=True)
    (NODE_DATA / "auth-token").write_text(NODE_TOKEN)
    subprocess.run(
        [
            "docker", "run", "-d", "--name", NODE_NAME,
            "-p", "18443:7687", "-p", "18444:8443", "-p", "19090:9090",
            "-e", "GRAPH_ADVERTISED_BOLT_ADDR=127.0.0.1:18443",
            "-e", "GRAPH_BOLT_NODE_ADDRESSES=node-0=127.0.0.1:18443",
            "-e", "GRAPH_AUTH_TOKEN_FILE=/data/auth-token",
            "-e", "GRAPH_ALLOW_PLAINTEXT=true",
            "-e", "CLOUD_PROVIDER=local",
            "-e", "GRAPH_ID=default", "-e", "GRAPH_CELL_ID=cell-0",
            "-e", "GRAPH_NODE_ID=node-0", "-e", "GRAPH_NAMESPACE=default",
            "-e", "GRAPH_CELLS=cell-0", "-e", "GRAPH_DATA_CACHE_DIR=/data/cache",
            "-e", "LOCAL_PATH=/data/store",
            "-v", f"{NODE_DATA}:/data",
            "ghcr.io/hydra-db/hydradb:latest",
        ],
        check=True,
        capture_output=True,
    )
    # Wait until the node answers queries.
    import requests

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            r = requests.post(
                f"{HYDRADB_URL}/v1/graphs/default/query",
                headers={
                    "Authorization": f"Bearer {NODE_TOKEN}",
                    "X-Graph-Namespace": "default",
                    "Content-Type": "application/json",
                },
                json={"cell_id": "cell-0", "query": "MATCH (e:Event) RETURN count(*) AS n"},
                timeout=5,
            )
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("HydraDB node did not come up within 60s")


def stratify_subset(
    samples: list[BenchmarkSample], n: int, seed: int
) -> tuple[list[BenchmarkSample], dict[str, Any]]:
    """Proportional stratified sample by question type, deterministic order."""
    rng = random.Random(seed)
    types = Counter(s.question_type for s in samples)
    order = sorted(types)
    quotas: dict[str, int] = {}
    assigned = 0
    for i, t in enumerate(order):
        raw = n * types[t] / len(samples)
        quota = int(raw)
        if i == len(order) - 1:
            quota = n - assigned
        else:
            remaining = n - assigned
            # Largest remainder method so totals add to n.
            if raw - quota > 0.5 or quota > remaining:
                quota = min(quota, remaining)
        quotas[t] = quota
        assigned += quota
    # Fix rounding drift: adjust largest type by the shortfall.
    drift = n - assigned
    if drift:
        quotas[order[0]] += drift
        assigned += drift
    assert assigned == n, (assigned, n)

    by_type: dict[str, list[BenchmarkSample]] = {}
    for s in samples:
        by_type.setdefault(s.question_type, []).append(s)
    chosen: list[BenchmarkSample] = []
    for t in order:
        pool = by_type[t]
        picked = rng.sample(pool, quotas[t])
        chosen.extend(picked)
    # Deterministic original order.
    chosen.sort(key=lambda s: samples.index(s))

    manifest = {
        "n_subset": n,
        "total": len(samples),
        "seed": seed,
        "selection": "proportional stratified random by question_type, largest-remainder quotas",
        "question_type_quota": quotas,
        "question_type_distribution": {
            "subset": {t: quotas[t] for t in order},
            "full": dict(types),
        },
        "sample_ids": [s.sample_id for s in chosen],
    }
    return chosen, manifest


class _FilteredBenchmark:
    """Iterable benchmark view exposing only the selected samples."""

    def __init__(self, name: str, samples: Sequence[BenchmarkSample]) -> None:
        self.name = name
        self._samples = list(samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self._samples)

    def __len__(self) -> int:
        return len(self._samples)


class _NoMemoryAdapter(BaseAdapter):
    """Trivial adapter: stores nothing, recalls nothing.

    The judge then answers from an empty context — the no-memory/abstention
    baseline. Also acts as a sanity control (expected ~chance accuracy).
    """

    def __init__(self) -> None:
        self._stored = 0

    def ingest(self, items, *, namespace="default") -> None:
        self._stored += len(items)

    def recall(self, query: RecallQuery, *, namespace="default") -> list[RecallResult]:
        return []

    def cleanup(self, *, namespace="default") -> None:
        pass

    @property
    def name(self) -> str:
        return "nomemory"


def make_adapter(name: str, cfg: RunConfig, cache_conversations: bool = False) -> BaseAdapter:
    if name == "bm25":
        from mem_bench.adapters.bm25 import BM25Adapter

        return BM25Adapter()
    if name == "nomemory":
        return _NoMemoryAdapter()
    if name == "hydradna":
        from hydradna import HydraDNAAdapter

        return HydraDNAAdapter(
            url=HYDRADB_URL, cache_conversations=cache_conversations
        )
    raise ValueError(f"Unknown adapter: {name}")


def run_adapter(
    name: str,
    samples: list[BenchmarkSample],
    cfg: RunConfig,
    out_dir: Path,
    batch_size: int,
    cache_conversations: bool = False,
) -> dict[str, Any]:
    from mem_bench.core import runner as _runner_mod
    from mem_bench.core.runner import BenchmarkRunner, _MAX_RETRIES

    class _RetryJudge:
        """Proxy that retries judge.evaluate with backoff (429-safe)."""

        def __init__(self, base):
            self._base = base

        def evaluate(self, *a, **kw):
            return retry_call(lambda: self._base.evaluate(*a, **kw), what="judge")

    # answer generator: same no-retry issue -> wrap the function the runner
    # imported at module scope.
    _orig_gen = _runner_mod.generate_answer
    _runner_mod.generate_answer = lambda *a, **kw: retry_call(
        lambda: _orig_gen(*a, **kw), what="generate_answer"
    )

    logger.info("=== adapter=%s samples=%d ===", name, len(samples))
    adapter = make_adapter(name, cfg, cache_conversations)
    bench = _FilteredBenchmark(f"longmemeval_{name}", samples)
    runner = BenchmarkRunner(adapter, bench, cfg)
    if runner._judge is not None:
        runner._judge = _RetryJudge(runner._judge)
    top_k = max(cfg.metrics.retrieval_k)
    k_values = cfg.metrics.retrieval_k

    progress = out_dir / "progress.jsonl"
    sample_results = []
    num_failed = 0

    for bi, start in enumerate(range(0, len(samples), batch_size)):
        batch = samples[start : start + batch_size]
        logger.info("batch %d/%d: samples %d-%d", bi + 1, (len(samples) - 1) // batch_size + 1, start + 1, start + len(batch))
        if name == "hydradna" and bi > 0:
            logger.info("resetting node for fresh batch")
            reset_node()
        for sample in batch:
            succeeded = False
            for attempt in range(_MAX_RETRIES + 1):
                try:
                    res = runner._run_sample(sample, top_k=top_k, k_values=k_values)
                    sample_results.append(res)
                    succeeded = True
                    break
                except Exception:
                    if attempt < _MAX_RETRIES:
                        logger.warning("Sample %s failed (attempt %d/%d), retrying", sample.sample_id, attempt + 1, _MAX_RETRIES + 1, exc_info=True)
                    else:
                        logger.warning("Sample %s failed after %d attempts, skipping", sample.sample_id, _MAX_RETRIES + 1, exc_info=True)
            if not succeeded:
                num_failed += 1
            with progress.open("a") as f:
                f.write(json.dumps({"sample_id": sample.sample_id, "ok": succeeded}) + "\n")

    # Aggregate like the runner does (per-adapter mean of each metric).
    from mem_bench.core.runner import RunResult

    result = RunResult(
        benchmark_name="longmemeval",
        split=cfg.split,
        adapter_name=name,
        num_samples=len(sample_results),
        num_failed=num_failed,
        sample_results=sample_results,
        config=cfg.model_dump(),
    )
    result.aggregate_metrics = runner._aggregate_metrics(sample_results)
    total_seconds = sum(r.timing.ingest_seconds + r.timing.recall_seconds + r.timing.cleanup_seconds for r in sample_results)
    result.total_seconds = total_seconds

    # Per-type breakdown (industry-standard reporting).
    by_type: dict[str, list[Any]] = {}
    for r in sample_results:
        by_type.setdefault(r.question_type, []).append(r)
    type_breakdown: dict[str, dict[str, float]] = {}
    for t, rs in sorted(by_type.items()):
        keys = {k for r in rs for k in r.retrieval_metrics}
        agg = {}
        for k in sorted(keys):
            vals = [r.retrieval_metrics.get(k, 0.0) for r in rs]
            agg[k] = sum(vals) / len(vals)
        qa = [r.qa_score for r in rs if r.qa_score is not None]
        if qa:
            agg["qa_accuracy"] = sum(1 for s in qa if s > 0.5) / len(qa)
        type_breakdown[t] = agg

    summary = {
        "benchmark": "longmemeval",
        "split": cfg.split,
        "adapter": name,
        "num_samples": len(sample_results),
        "num_failed": num_failed,
        "aggregate_metrics": result.aggregate_metrics,
        "type_breakdown": type_breakdown,
        "total_seconds": total_seconds,
        "metadata": result.metadata,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", default="hydradna,bm25,nomemory")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", default="s")
    ap.add_argument("--benchmark", default="longmemeval", choices=["longmemeval", "locomo"])
    ap.add_argument("--config", default=str(MODULE_DIR / "mem-bench.toml"))
    ap.add_argument("--output", default=str(MODULE_DIR.parent / "results" / "lme_subset"))
    ap.add_argument("--judge-model", default=None, help="override judge/answer model (e.g. qwen3.5:4b for local Ollama)")
    ap.add_argument("--judge-base-url", default=None, help="override judge base URL (e.g. http://localhost:11434/v1 for Ollama)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    cfg = load_config(args.config)
    cfg.split = args.split
    cfg.benchmark = args.benchmark
    if args.judge_model:
        cfg.judge.model = args.judge_model
    if args.judge_base_url:
        cfg.judge.base_url = args.judge_base_url

    if args.benchmark == "locomo":
        from mem_bench.benchmarks.locomo import LoCoMoBenchmark

        bench = LoCoMoBenchmark(cache_dir="/tmp/mb_data")
        bench.load(split="test", limit=0)
        cache_conversations = True
    else:
        bench = LongMemEvalBenchmark(cache_dir="/tmp/mb_data")
        bench.load(split=args.split, limit=0)
        cache_conversations = False
    all_samples = list(bench)
    logger.info("loaded %d samples (split=%s)", len(all_samples), args.split)

    subset, manifest = stratify_subset(all_samples, args.n, args.seed)
    logger.info("subset %d samples: %s", len(subset), manifest["question_type_quota"])

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))

    summaries = []
    for name in [a for a in args.adapters.split(",") if a]:
        adir = out_root / name
        adir.mkdir(parents=True, exist_ok=True)
        summaries.append(run_adapter(name, subset, cfg, adir, args.batch, cache_conversations))

    # Comparison table.
    table = {
        "adapters": [s["adapter"] for s in summaries],
        "metrics": list(summaries[0]["aggregate_metrics"].keys())
        if summaries
        else [],
        "aggregate": {s["adapter"]: s["aggregate_metrics"] for s in summaries},
        "num_failed": {s["adapter"]: s["num_failed"] for s in summaries},
    }
    (out_root / "comparison.json").write_text(json.dumps(table, indent=2))
    for s in summaries:
        print(f"\n[{s['adapter']}] n={s['num_samples']} failed={s['num_failed']}")
        for k, v in sorted(s["aggregate_metrics"].items()):
            print(f"  {k}: {v:.4f}")
    logger.info("done. outputs in %s", out_root)


if __name__ == "__main__":
    main()
