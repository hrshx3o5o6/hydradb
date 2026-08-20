---
title: Retrieval Benchmarking Research for HydraDNA
status: draft-for-review
date: 2026-08-18
branch: feature/causal-memory-layer
tags:
  - causal-memory
  - benchmarks
  - retrieval
  - evaluation
---

# Retrieval Benchmarking Research — HydraDNA

How to measure HydraDNA's causal memory against mem0, Zep/Graphiti, Letta,
LangMem, and vector RAG — fairly, reproducibly, and on the dimensions that
actually matter for agent memory.

## Executive Summary

- **There is no "causal memory" benchmark.** Existing benchmarks test fact
  recall, temporal reasoning, contradiction handling, and multi-hop retrieval.
  They do **not** test *why* answers. HydraDNA's edge is untested territory —
  the opportunity is to run standard benchmarks for a level playing field, then
  add a **causal capability suite** no one else can run.
- **Standard benchmarks to use:** LongMemEval (ICLR 2025), LoCoMo (ACL 2024),
  BEAM (ICLR 2026), HaluMem (2025). These are the field's consensus.
- **Reuse existing harnesses.** `mem0ai/memory-benchmarks` (Apache-2.0) and
  `rivercrab26/mem-bench` already implement adapters for Mem0, Graphiti, Letta,
  LangMem, BM25 on LongMemEval/LoCoMo/HaluMem. HydraDNA plugs in as one more
  adapter. Do not build an evaluation framework from scratch.
- **The credibility gap is the real win.** Vendors self-report ~93-95% on
  LongMemEval; independent re-runs of the open-source editions land at 32-63%,
  often *below* a no-memory LLM baseline. A reproducible, signed benchmark run
  is itself a differentiator.
- **Where HydraDNA should win:** contradiction resolution (OVERWRITES),
  temporal/causal chains, root-cause attribution, abstention, explainability,
  token efficiency. Where it starts behind: raw recall of isolated facts —
  causal edges add latency and a discovery failure mode.

---

## 1. Benchmark Landscape

Four families, four different questions.

| Family | Representative benchmarks | Answers "how good is the ..." |
|---|---|---|
| Agent memory | LongMemEval, LoCoMo, BEAM, MemBench, HaluMem | system at remembering across sessions |
| GraphRAG | GraphRAG-Bench, BenchmarkQED, RAG-vs-GraphRAG | graph-structured retrieval at answering QA |
| RAG quality | RAGAS, TruLens (metrics), BEIR/MTEB (embeddings) | grounding + retrieval quality of *any* pipeline |
| Causal | SHD/edge-PR (causal-learn), CounterBench, Test-of-Time, CausalBench | causal discovery + reasoning capability |

### 1.1 Agent memory benchmarks (primary target)

| Benchmark | Venue | Size | What it tests |
|---|---|---|---|
| **LoCoMo** | ACL 2024 | 1,540 Qs, ~300 turns/9K tok, ≤35 sessions | single-hop, multi-hop, temporal, open-domain, adversarial |
| **LongMemEval** | ICLR 2025 | 500 Qs; splits S (~115K tok / 40 sessions), M (~500 sessions), Oracle | extraction, multi-session reasoning, temporal, knowledge updates, abstention |
| **LongMemEval-V2** | 2026 | ≤500 web trajectories, 115M tok, multimodal | agentic memory: static/dynamic state, workflows, gotchas, premises. Metric: **LAFS** (latency-accuracy frontier) |
| **BEAM** | ICLR 2026 | 2,000 Qs, 128K→10M tok | 10 abilities incl. contradiction resolution, event ordering, abstention |
| **MemBench** | ACL 2025 | factual + reflective, participation + observation scenarios | effectiveness, efficiency, capacity |
| **HaluMem** | 2025 | 3.5K Qs, ~15K memory points, >1M-tok contexts | **operation-level**: extraction, update, QA hallucination/omission — matches HydraDNA's write path |

**Recommendation for HydraDNA:** run **LongMemEval (S + Oracle)**, **LoCoMo**,
and **HaluMem** as the headline trio. LongMemEval's *knowledge updates* and
*abstention* map directly to OVERWRITES and `why()` confidence. HaluMem's
extraction/update/QA decomposition maps to HydraDNA's event-write / overwrite /
retrieval stages. BEAM at 1M+ tokens is a stretch goal.

**Known pitfalls:** LoCoMo has documented scoring flaws (empty-gold bug,
string-matching bias against paraphrases — Tanguturi 2026). Report
LLM-judge accuracy and retrieval metrics separately, and prefer the corrected
LongMemEval v2025-09 revision.

### 1.2 GraphRAG benchmarks (secondary, informs retrieval design)

- **GraphRAG-Bench** (ICLR 2026) — 1,018 CS-domain Qs, 4 question types.
  Metrics: answer accuracy, **R score**, **AR score**, plus retrieval-side
  **context relevancy** and **evidence recall**. Ships unified eval code across
  LightRAG/HippoRAG2/GraphRAG — reusable as a template for HydraDNA's own
  retrieval-eval scripts.
- **BenchmarkQED** (Microsoft, 2025) — automated QA generation + LLM-judge
  comparisons; LazyGraphRAG beat vector RAG (incl. 1M-token window) on
  comprehensiveness/diversity/empowerment/relevance. Methodology (query classes,
  win-rate aggregation) is directly reusable.
- **CDR-Benchmark** (EMNLP 2025) — conversational data retrieval, 1.6K Qs /
  9.1K conversations; best embedding NDCG@10 ≈ 0.51. Shows plain embedding
  retrieval is weak on conversational data — relevant when HydraDNA compares
  against a naive vector baseline.

### 1.3 RAG quality metrics (RAGAS) — the shared scoring layer

Applied to any memory system's retrieved context + answer. Reference-free
except context recall.

| Metric | Graded side | Failure it flags |
|---|---|---|
| Faithfulness | generator | hallucination beyond retrieved context |
| Answer relevancy | generator | evasive / off-topic answers |
| Context precision | retriever | noise ranked above signal |
| Context recall | retriever | missing needed evidence |
| Entity recall / noise sensitivity | retriever | entity gaps, distractor corruption |

For HydraDNA these reframe as: *given the causal path HydraDNA walked, is the
answer faithful, and did the path contain the evidence?* Context precision on a
path means "root cause ranked near the top", not "relevant chunk near the top".

### 1.4 Causal discovery & reasoning benchmarks (the differentiator)

**Graph-structure metrics** (causal-learn `Evaluations`, ALCM, CauScientist):

- **Edge Precision / Recall / F1** — are discovered CAUSES edges correct?
- **Structural Hamming Distance (SHD)** — edits to turn discovered graph into
  ground truth. **NHD** = normalized.
- **Arrow precision/recall** (directed edges), TENE (missed true edges),
  TERE (reversed edges), AUPRC over confidence thresholds.

**Reasoning benchmarks:**

- **CounterBench** (2025) — 1K formally specified counterfactual Qs over SCMs;
  LLMs near random → tests HydraDNA's `what-if` vision.
- **Test of Time** (2024) — temporal reasoning incl. event ordering; shows
  LLM sensitivity to premise order — motivates graph-side ordering.
- **CausalBench / CaLM** — Pearl's ladder (association → intervention →
  counterfactual) for LLMs.

HydraDNA should synthesize a **small purpose-built causal-memory dataset** with
a *ground-truth event DAG* (see §4.2) and score edge-P/R/F1 + SHD + path
metrics on it. No off-the-shelf benchmark covers this.

---

## 2. Reference Numbers (as of 2026-08)

### 2.1 Self-reported (vendor) — treat with caution

| System | LoCoMo | LongMemEval | Notes |
|---|---|---|---|
| Mem0 (managed) | 92.5 / 66.9-68.5* | 94.4 / 93.4* | temporal reasoning (v3) + token-efficient algorithm |
| Zep / Graphiti | 94.7 (155ms, 5,760 tok) | 90.2 (162ms, 4,408 tok) | temporal knowledge graph; graphiti OSS |
| Letta (MemGPT) | — | — | no published leaderboard numbers |
| LLM baseline (no memory) | 50.4-52.9 | 57.6 | GPT-4o-mini, no memory layer |

### 2.2 Independently verified (benchd.ai harness, 2026-05)

| System | LongMemEval | LoCoMo |
|---|---|---|
| LlamaIndex | 59.0% | 54.8% |
| LangChain | 59.0% | 51.9% |
| LLM baseline (no memory) | 57.6% | 50.4% |
| AutoGPT | 47.4% | — |
| CrewAI | 46.0% | — |
| Mem0 OSS | 32.4% | 0.0% |
| Graphiti / Letta / gbrain | 0.0% | — |

**Implication:** OSS mem0 lands *below* the no-memory LLM baseline on LongMemEval
in an independent run. Self-reported vs independent differs by up to 61 points.
Any credible HydraDNA claim must ship a reproducible run, judge LLM + embedding
pinned, seed recorded, splits named.

---

## 3. Harnesses to Reuse (do not build from scratch)

### 3.1 `rivercrab26/mem-bench` — primary (neutral, pluggable)
- Unified adapter interface: **`ingest()`, `recall()`, `cleanup()`** — 3 methods.
- Adapters exist for: BM25 (built-in), Mem0, Graphiti, LangMem, Letta,
  Hindsight. Benchmarks: LongMemEval, LoCoMo, HaluMem.
- Metrics: `recall_any@k`, `recall_all@k`, `ndcg@k`, `mrr`, LLM-judge QA
  accuracy by question type, abstention accuracy, latency.
- CLI: `mem-bench run --adapter <x> --benchmark longmemeval --split oracle`,
  `mem-bench compare dir1 dir2 --format markdown`.
- **Move:** write a `hydradna` adapter (3 methods) against the existing
  `causal_memory` SDK. Reuses all baselines for free.

### 3.2 `mem0ai/memory-benchmarks` — cross-check (Apache-2.0)
LoCoMo / LongMemEval / BEAM runners, Mem0 Cloud + OSS, docker compose, web UI.
Use to reproduce Mem0's own numbers under identical conditions to HydraDNA.

### 3.3 `benchd.ai/harness` — independent verification
Open source, signed result manifests, full failure traces. Best-practice
reference for reproducibility even if HydraDNA uses mem-bench.

### 3.4 `GraphRAG-Bench` eval scripts + RAGAS
Reuse GraphRAG-Bench's `retrieval_eval` (context relevancy + evidence recall)
and RAGAS for the shared scoring layer (§1.3).

---

## 4. HydraDNA Benchmarking Plan

### 4.1 Layer 1 — Standard agent-memory accuracy (level playing field)

| Benchmark | Split | Primary metrics | Why |
|---|---|---|---|
| LongMemEval | oracle, S | recall_any@k/recall_all@k, ndcg@k, mrr, QA acc, abstention acc | core recall + updates + abstention |
| LoCoMo | full | QA acc (single/multi-hop/temporal/adversarial), retrieval k-metrics | multi-session, temporal |
| HaluMem | medium | extraction acc/recall, update acc, QA hallucination/omission | write-path fidelity |

Config: fixed judge LLM (e.g. GPT-4o-mini) at temperature 0, same embedding
model across all vector-dependent systems, `k ∈ {1,3,5,10}`, latency (p50/p95/
p99) + prompt tokens + total context tokens per question. Record every run in
a JSONL + signed manifest.

**HydraDNA adapter behavior (mem-bench `ingest/recall`):**
- `ingest(conversation)` → replay into `add_event` + edge discovery daemon
  (or offline discovery pass), matching the benchmark's session structure.
- `recall(question)` → `why()` / `search()` path; return retrieved *events*
  as "context" so RAGAS + retrieval metrics can score them like chunks.

### 4.2 Layer 2 — Causal capability suite (the differentiator)

Synthesize a dataset with a **known ground-truth causal DAG** over events:
- ~40-80 conversations × 8-20 sessions, generated from personas + an event
  DAG (LoCoMo-style pipeline), with annotated: true CAUSES edges, true
  OVERWRITES edges, and gold `why` paths.

Metrics:
- **Discovery quality:** edge Precision / Recall / F1, SHD (and NHD) vs truth;
  AUPRC by sweeping discovery confidence threshold.
- **`why()` path quality:** path-accuracy (exact node sequence match), node
  hit@k (does the retrieved path contain root cause in top-k?), path recall
  (fraction of gold chain retrieved).
- **Overwrite resolution:** "what is true now?" accuracy — HydraDNA's
  OVERWRITES vs mem0 temporal scoring vs Zep invalidation.
- **Event ordering:** fraction of pairwise orders preserved vs gold.
- **Abstention:** correct "I don't know" rate when question unanswerable.
- **Counterfactual (stretch):** a small CounterBench-style SCM set mapped onto
  HydraDNA events to test `what-if`.

**Ablations (isolate causal contribution):**
1. HydraDNA causal edges **off** → pure event text search (the vector-ish floor)
2. HydraDNA causal edges **on**, confidence threshold sweep 0.5/0.7/0.8/0.9
3. maxLen sweep 2/4/8 for path traversal
4. hybrid: causal path + top-k semantic (best of both)

### 4.3 Layer 3 — Shared scoring on top of any system's retrieval

Run RAGAS faithfulness, answer relevancy, context precision/recall over the
contexts each system retrieved (paths for HydraDNA, chunks for others). This
lets us *separate retrieval quality from generation quality* and cite one
metric family across all competitors.

### 4.4 Layer 4 — Operational cost frontier

Report latency and token cost alongside every accuracy number (the LAFS idea
from LongMemEval-V2; MemScore-style composite). Graph systems trade accuracy
for latency — HydraDB's GraphBLAS + pinned-snapshot reads should keep
`why()` in single-digit ms at ~1K events.

---

## 5. Competitor Matrix for the Comparison Run

| Adapter | Type | In scope |
|---|---|---|
| BM25 (built-in) | sparse baseline | yes |
| LLM, no memory (built-in) | the honest floor | yes |
| Mem0 OSS | vector + facts | yes |
| Graphiti (Zep) | temporal KG | yes |
| LangMem | LangGraph memory | yes |
| Letta | tiered memory | yes |
| Chroma/OpenAI vector RAG | raw vector | yes (custom adapter) |
| Mem0 managed | self-reported only | no (cloud, paywalled) |

**Fairness rules:** same judge model/temp; same embedding model; identical
splits; same `k`; report all three of accuracy / latency / tokens; pin seeds;
publish run manifests; never cherry-pick the run that "looked best".

---

## 6. Expected Outcomes (hypotheses to falsify)

| Dimension | Expected HydraDNA result |
|---|---|
| Fact recall (LongMemEval IE) | middle of pack; may trail raw vector recall |
| Temporal reasoning | top tier (graph ordering beats flat text) |
| Knowledge updates / contradictions | **top tier** — OVERWRITES is explicit |
| Multi-hop / path retrieval | **top tier** — structural traversal |
| Abstention | top tier — confidence-gated `why` |
| Token efficiency | strong — injects paths, not transcript dumps |
| `why` / root cause (custom suite) | **only system that can run it** |

The honest framing for any published result: *"We match mem0/Zep on recall;
we beat them where causes matter — and we can prove it."*

---

## 7. Execution Plan

1. **Harness (0.5 day):** fork/install `mem-bench`; write `hydradna` adapter;
   smoke on LongMemEval oracle split, limit 10.
2. **Baselines (0.5 day):** run BM25 + no-memory LLM baseline; confirm
   reproduction of benchd's published numbers (sanity check on our harness).
3. **Headline run (1 day):** LongMemEval (oracle, S) + LoCoMo across all
   competitors; capture accuracy + latency + tokens; RAGAS pass.
4. **Causal suite (1-2 days):** build ground-truth-DAG dataset; implement
   edge-P/R/F1, SHD, path metrics, overwrite/ordering/abstention scores.
5. **Ablations (0.5 day):** edges off/on, threshold + maxLen sweeps, hybrid.
6. **Report (0.5 day):** reproducible JSONL + manifest + `benchmark.md`
   (this becomes a page on the landing site: "HydraDNA vs the field").

---

## 8. Risks & Caveats

- **LoCoMo scoring bugs** — prefer corrected splits; report judged accuracy
  and retrieval metrics separately.
- **LLM-as-judge bias** — calibrate judge vs human labels on a 50-sample
  subset; pin judge model; temperature 0.
- **Self-reported ≠ independent** — always label which; target reproducible
  numbers as the product story.
- **Causal ground truth is expensive** — the causal suite dataset is the main
  manual cost; keep DAGs small and generated.
- **No embedding in HydraDNA by default** — recall-only search may underperform
  vector recall; the hybrid (causal + semantic) ablation is the honest
  comparison and likely the production recommendation.

---

## 9. Results Log

### 2026-08-19 — LongMemEval S split, stratified 100-sample subset (seed 42)

**Run protocol (reproducible):** `benchmarks/mem_bench/run_subset.py` —
stratified proportional sampling by question type (largest-remainder quotas),
fixed seed 42, exact sample ids in
`benchmarks/results/lme_subset/subset_manifest.json`. Judge: gpt-4o-mini
(OpenAI, temp 0), retrieval `k ∈ {1,3,5,10}`. HydraDNA ran on a dedicated
single-node HydraDB (port 18444) with node recreation every 40 samples (engine
degrades under sustained writes: deletes silently no-op, graph grows, queries
slow — see `.tnj/learnings/hydradb-engine-write-limits.md`). Adapter:
byte-aware chunking (700-byte budget, engine hard-caps string literals at
~880 UTF-8 bytes, not chars) + causal discovery capped at 40 events/sample.

**Headline comparison** (`benchmarks/results/lme_subset/`):

| metric | hydradna (n=90) | bm25 (n=100) | nomemory (n=100) |
|---|---|---|---|
| qa_accuracy | 0.400 | **0.520** | 0.080 |
| recall_any@1 | 0.722 | 0.750 | 0.000 |
| recall_any@5 | 0.878 | 0.900 | 0.000 |
| recall_all@1 | 0.200 | 0.220 | 0.000 |
| recall_all@5 | 0.700 | 0.740 | 0.000 |
| mrr | 0.796 | 0.812 | 0.000 |
| ndcg@5 | 0.734 | 0.774 | 0.000 |
| failed | 10 | 0 | 0 |

**Verdict: BM25 > HydraDNA on LongMemEval S.** The causal-edge overhead is net
negative on this benchmark — LongMemEval is dominated by cross-session
coreference that lexical overlap already solves. nomemory scores 0.08, so the
judge is discriminating and retrieval is what drives accuracy.

**HydraDNA type breakdown (qa_accuracy):** single-session-user 0.64,
knowledge-update 0.47, temporal-reasoning 0.46, single-session-assistant 0.45,
multi-session 0.23, single-session-preference 0.00. Weakest on
multi-session + preference (causal edges not compensating for multi-hop).

**Failure modes (10/100):** 21 engine HTTP 400s (parser/engine errors), 10
DNS ConnectErrors to the LLM host (transient network window), 3 read
timeouts.

**Next:** LoCoMo (multi-hop + temporal reasoning — the benchmark causal memory
should win), then retrieval tuning driven by the per-type breakdown.

---

### 2026-08-20 — LoCoMo, stratified 300-sample subset (seed 42), 0 failures

Same protocol as above (`run_subset.py --benchmark locomo`), all 10
conversations covered, quotas: open_domain 127, adversarial 67, multi_hop 48,
single_hop 42, temporal 16. **Key change:** conversation caching in the adapter
(`cache_conversations=True`) — each ~20-session conversation is ingested ONCE
and reused across its QA samples (naive re-ingest would be ~196k chunk writes;
cached = ~1.2k). This cut per-sample ingest to ~0s, kept the node at a single
conversation (~150 events), and eliminated the engine-degradation failures.
Full run: hydradna+bm25+nomemory in ~55 min, **0 failed / 300**.

**Headline comparison** (`benchmarks/results/locomo_subset/`):

| metric | hydradna | bm25 | nomemory |
|---|---|---|---|
| qa_accuracy | 0.437 | **0.503** | 0.000 |
| recall_any@1 | **0.577** | 0.487 | 0.000 |
| recall_any@5 | **0.807** | 0.793 | 0.000 |
| recall_all@1 | **0.530** | 0.430 | 0.000 |
| recall_all@5 | **0.710** | 0.677 | 0.000 |
| mrr | **0.684** | 0.613 | 0.000 |
| ndcg@5 | **0.679** | 0.616 | 0.000 |

**HydraDNA vs BM25 by question type (recall_any@1 / qa):**

| type | hydradna | bm25 |
|---|---|---|
| multi_hop | **0.56** / 0.12 | 0.44 / 0.08 |
| temporal | **0.19** / 0.12 | 0.12 / 0.19 |
| adversarial | **0.63** / 0.24 | 0.55 / 0.34 |
| open_domain | **0.72** / 0.72 | 0.55 / 0.80 |
| single_hop | 0.24 / 0.38 | **0.38** / 0.45 |

**Verdict — the causal layer works, the context packaging doesn't (yet):**
HydraDNA **beats BM25 on retrieval on every "causal" type** (multi_hop +12,
temporal +7, adversarial +8, open_domain +17 recall_any@1; mrr +6-9) — the
multi-hop/temporal/adversarial wins predicted in §6 materialized. But its
**qa_accuracy trails BM25 everywhere except multi_hop**, despite retrieving
the right sessions more often. Root cause hypothesis: `recall()` returns
**700-byte chunk slices** as context; BM25 returns **whole sessions**. The LLM
answers better with complete session text. Fix in tuning: return the full
session text (re-joined from chunks) as `RecallResult.content`. The other
anomaly: hydradna loses single_hop retrieval (0.24 vs 0.38) — causal-neighbor
expansion likely dilutes pure lexical scoring on single-session extraction;
worth an ablation (edges off on single_hop).

nomemory = 0.000 confirms the judge is fully retrieval-driven here (LongMemEval
showed 0.08 — LoCoMo's adversarial/empty-gold questions make empty context a
hard abstain).

---

### 2026-08-20 — Positioning vs the field (after LoCoMo)

Absolute qa_accuracy is **mid-pack** vs the independently verified field:
LongMemEval S 0.40 (LlamaIndex 0.59, LLM-baseline 0.576, Mem0 OSS 0.32) and
LoCoMo 0.44 (LlamaIndex 0.548, LLM-baseline 0.504). These are **not directly
comparable** (different judge/subset/harness), but the honest ballpark is
"around the no-memory LLM baseline, below the top independent numbers."

Where HydraDNA is genuinely strong today:
- **Causal retrieval on the claims:** beats BM25 on recall_any@1/mrr/ndcg on
  multi_hop (+12), temporal (+7), adversarial (+8), open_domain (+17).
- **Reliability:** 0/300 failures on LoCoMo (caching + clean node).
- **Latency:** recall ~0.03s/sample; 3-adapter benchmark in ~55 min.

Where it loses:
- **qa_accuracy trails BM25 (0.437 vs 0.503)** — context packaging: the judge
  gets 700-byte chunk slices, BM25 gives whole sessions.
- **single_hop retrieval** (0.24 vs 0.38) — causal expansion dilutes lexical.
- **LongMemEval S qa** (0.40) — coreference-heavy; not causal-memory home turf.

The one-line story: *retrieves better on the benchmarks causal memory claims,
answers worse until the context-packaging fix lands.*

---

### 2026-08-20 — Context-packaging fix: full-session rejoin (VALIDATED)

**Fix** (`hydradna.py`): cache `document_id -> full original session text`
during ingest; in `recall()` keep chunk-level scoring (so retrieval metrics are
unchanged) but swap `RecallResult.content` from the 700-byte slice to the whole
session. The judge now reads complete evidence like BM25, with our better
session ranking on top.

**Re-run, LoCoMo 300-subset (seed 42), same gpt-4o-mini judge**
(`benchmarks/results/locomo_subset_fix/`):

| metric | hydradna before | hydradna after | delta |
|---|---|---|---|
| qa_accuracy | 0.437 | **0.573** | **+13.6** |
| recall_any@1 | 0.577 | 0.577 | 0 (as designed) |
| recall_any@5 | 0.807 | 0.807 | 0 |
| recall_all@1 | 0.530 | 0.530 | 0 |
| mrr | 0.684 | 0.684 | 0 |

0 failed / 300. The retrieval metrics were bit-identical — packaging had zero
effect on retrieval, exactly as designed — and qa_accuracy jumped past BM25's
old 0.503. **HydraDNA now wins both retrieval and QA on LoCoMo.**
Cost tradeoff: context is now ~30-40KB/question (whole sessions) vs ~7KB
(slices); gpt-4o-mini handles it, but tokens/query is a real axis to report.

**Blocker on completing the comparison:** mid-run the OpenAI account hit
`credit_balance_exhausted`, so the bm25 + nomemory legs failed (165/300 and
300/300) and had to be re-run. They are **invalidated** in `locomo_subset_fix/`
and pending a re-run with credits. A retry-with-backoff wrapper (429/credit-safe)
was added to `run_subset.py` so future judge/answer calls retry instead of
silently dropping samples. The `+13.6` headline stands on its own (same judge,
same run, hydradna leg only).

---

### 2026-08-21 — Offline comparison with local judge (credits unavailable)

OpenAI credits are exhausted, so the full 3-adapter LoCoMo 300-subset was
re-run with a **local Ollama judge** (`llama3.2:3b` via OpenAI-compatible
`base_url`, added `--judge-model/--judge-base-url` CLI to `run_subset.py`).
Same subset, same seed 42, same harness; 0 failed / 300 across all adapters
(`benchmarks/results/locomo_subset_local/`):

| metric | hydradna | bm25 | nomemory |
|---|---|---|---|
| qa_accuracy | **0.350** | 0.347 | 0.073 |
| recall_any@1 | **0.577** | 0.487 | 0.000 |
| recall_any@5 | **0.807** | 0.793 | 0.000 |
| recall_all@1 | **0.530** | 0.430 | 0.000 |
| mrr | **0.684** | 0.613 | 0.000 |
| ndcg@5 | **0.679** | 0.616 | 0.000 |

Notes:
- **Retrieval metrics are judge-free** and bit-identical to the gpt-4o-mini
  run: HydraDNA wins recall_any@1 (+9), recall_all@1 (+10), mrr (+7), ndcg@5
  (+6) and recall_any@5 (+1.4). The retrieval claim is fully validated offline.
- qa_accuracy compresses under the weak 3B judge (0.350 vs 0.347 — a near tie;
  gpt-4o-mini showed +13.6). QA differentiation is judge-strength-dependent;
  retrieval advantage is not.
- nomemory qa 0.073 (not 0) confirms the 3B judge occasionally says "yes" to
  empty context — noisier than gpt-4o-mini, but the canary still holds.

---

## References

**Benchmarks:** LoCoMo (arXiv:2402.09727), LongMemEval (arXiv:2410.10813,
ICLR 2025), LongMemEval-V2 (arXiv:2605.12493), BEAM (arXiv:2510.27246, ICLR
2026), MemBench (arXiv:2506.21605, ACL 2025), HaluMem (arXiv:2511.03506),
GraphRAG-Bench (arXiv:2506.05690, ICLR 2026), RAG vs GraphRAG
(arXiv:2502.11371), CDR-Benchmark (EMNLP 2025 industry), CounterBench
(arXiv:2502.11008), Test of Time (arXiv:2406.09170), ALCM (arXiv:2405.01744),
CauScientist (arXiv:2601.13614).

**Harnesses:** mem0ai/memory-benchmarks, rivercrab26/mem-bench,
benchd.ai/harness, GraphRAG-Bench/GraphRAG-Benchmark, RAGAS (docs.ragas.io),
causal-learn `Evaluations`.

**Reference numbers:** mem0.ai/research (2026-08), getzep.com research +
benchd.ai (2026-05).
