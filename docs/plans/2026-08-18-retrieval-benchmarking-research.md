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
