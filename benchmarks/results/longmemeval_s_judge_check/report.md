# Benchmark Report: HydraDNA

| Field | Value |
|-------|-------|
| Benchmark | longmemeval |
| Split | s |
| Samples | 10 |
| Failed | 0 |
| Total Time | 402.1s |

> **NOTE:** This adapter uses fact-extraction mode: it extracts and returns synthesized memories rather than original documents. Document-ID-based retrieval metrics (recall@k, nDCG, MRR) are NOT meaningful for this system. Use QA accuracy as the primary evaluation metric.

## Metrics by Question Type

| Question Type | Count | mrr | ndcg@1 | ndcg@10 | ndcg@3 | ndcg@5 | recall_all@1 | recall_all@10 | recall_all@3 | recall_all@5 | recall_any@1 | recall_any@10 | recall_any@3 | recall_any@5 | qa_accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single-session-user | 10 | 0.8200 | 0.7000 | 0.8649 | 0.8262 | 0.8649 | 0.7000 | 1.0000 | 0.9000 | 1.0000 | 0.7000 | 1.0000 | 0.9000 | 1.0000 | 0.5000 |
| **Overall** | 10 | **0.8200** | **0.7000** | **0.8649** | **0.8262** | **0.8649** | **0.7000** | **1.0000** | **0.9000** | **1.0000** | **0.7000** | **1.0000** | **0.9000** | **1.0000** | **0.5000** |

## Aggregate Metrics

| Metric | Value |
|--------|------:|
| mrr | 0.8200 |
| ndcg@1 | 0.7000 |
| ndcg@10 | 0.8649 |
| ndcg@3 | 0.8262 |
| ndcg@5 | 0.8649 |
| qa_accuracy | 0.5000 |
| recall_all@1 | 0.7000 |
| recall_all@10 | 1.0000 |
| recall_all@3 | 0.9000 |
| recall_all@5 | 1.0000 |
| recall_any@1 | 0.7000 |
| recall_any@10 | 1.0000 |
| recall_any@3 | 0.9000 |
| recall_any@5 | 1.0000 |

## Timing Summary

| Metric | Value (s) |
|--------|----------:|
| mean_cleanup_seconds | 19.4601 |
| mean_ingest_seconds | 18.5918 |
| mean_recall_seconds | 0.1056 |
| total_cleanup_seconds | 194.6007 |
| total_ingest_seconds | 185.9184 |
| total_recall_seconds | 1.0561 |

## Configuration

```json
{
  "benchmark": "longmemeval",
  "split": "s",
  "limit": 10,
  "output_dir": "/Users/harsha/Downloads/ventures_hacks_projects_backup/hydradb/benchmarks/results/longmemeval_s_judge_check",
  "adapter": {
    "name": "hydradna:HydraDNAAdapter",
    "options": {}
  },
  "judge": {
    "enabled": true,
    "model": "gpt-4o-mini",
    "provider": "openai",
    "api_key_env": "OPENAI_API_KEY",
    "base_url": null
  },
  "metrics": {
    "retrieval_k": [
      1,
      3,
      5,
      10
    ],
    "include_latency": true,
    "compute_semantic": false,
    "semantic_retrieval_k": [
      1,
      3,
      5,
      10
    ],
    "semantic_judge_model": "claude-haiku-4-5-20251001"
  },
  "reporting": {
    "formats": [
      "console",
      "json",
      "markdown"
    ]
  }
}
```
