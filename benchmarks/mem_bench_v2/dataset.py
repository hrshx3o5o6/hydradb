"""
LoCoMo dataset loading + stratified subset selection for the v2 (token-metered)
benchmark.

Source: snap-research/locomo, data/locomo10.json (10 conversations). The
original benchmark package this repo previously depended on (`mem_bench`,
PyPI-installed editable from a temp dir) is gone - its temp source directory
was cleaned up by the OS between sessions. This module reads the raw dataset
directly instead.
"""

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_PATH = "/tmp/mb_data/locomo10.json"

# LoCoMo category codes (from the paper): 1=single-hop, 2=multi-hop,
# 3=temporal, 4=open-domain, 5=adversarial.
CATEGORY_NAMES = {
    1: "single_hop",
    2: "multi_hop",
    3: "temporal",
    4: "open_domain",
    5: "adversarial",
}


@dataclass
class QASample:
    conv_id: str
    qa_idx: int
    question: str
    answer: str
    category: int


def load_conversations(path: str = DEFAULT_PATH) -> List[Dict[str, Any]]:
    return json.loads(Path(path).read_text())


def stratified_subset(
    conv: Dict[str, Any], n: int, seed: int
) -> List[QASample]:
    """Proportional stratified sample of a conversation's QA pairs by
    category, largest-remainder rounding so quotas sum exactly to n.
    Mirrors the selection method already used by this repo's LongMemEval
    subset driver (benchmarks/mem_bench/run_subset.py:stratify_subset).
    """
    qa = conv["qa"]
    n = min(n, len(qa))
    rng = random.Random(seed)
    cats = Counter(q.get("category") for q in qa)
    order = sorted(cats)
    quotas: Dict[Any, int] = {}
    assigned = 0
    for i, c in enumerate(order):
        raw = n * cats[c] / len(qa)
        quota = int(raw)
        if i == len(order) - 1:
            quota = n - assigned
        else:
            remaining = n - assigned
            if raw - quota > 0.5 or quota > remaining:
                quota = min(quota, remaining)
        quotas[c] = quota
        assigned += quota
    drift = n - assigned
    if drift:
        quotas[order[0]] += drift

    by_cat: Dict[Any, List[int]] = {}
    for i, q in enumerate(qa):
        by_cat.setdefault(q.get("category"), []).append(i)

    chosen_idx: List[int] = []
    for c in order:
        pool = by_cat[c]
        k = min(quotas[c], len(pool))
        chosen_idx.extend(rng.sample(pool, k))
    chosen_idx.sort()

    return [
        QASample(
            conv_id=conv["sample_id"],
            qa_idx=i,
            question=qa[i]["question"],
            # Category 5 (adversarial) questions have no correct answer in
            # the conversation; LoCoMo stores the expected refusal under
            # "adversarial_answer" instead of "answer".
            answer=str(qa[i].get("answer", qa[i].get("adversarial_answer", ""))),
            category=qa[i].get("category"),
        )
        for i in chosen_idx
    ]


def build_manifest(
    conversations: List[Dict[str, Any]], per_conv_n: int, seed: int
) -> Dict[str, Any]:
    """Select conv_ids (by dataset order, deterministic) + per-conversation
    stratified QA subsets, and return both the samples and a manifest dict
    documenting exactly how they were chosen (reproducibility)."""
    samples: List[QASample] = []
    manifest: Dict[str, Any] = {
        "source": "snap-research/locomo data/locomo10.json",
        "seed": seed,
        "per_conversation_n": per_conv_n,
        "selection": "proportional stratified random by LoCoMo category, "
        "largest-remainder quotas, per conversation",
        "conversations": [],
    }
    for conv in conversations:
        sub = stratified_subset(conv, per_conv_n, seed)
        samples.extend(sub)
        manifest["conversations"].append(
            {
                "conv_id": conv["sample_id"],
                "total_qa": len(conv["qa"]),
                "selected_qa": len(sub),
                "category_counts": dict(
                    Counter(CATEGORY_NAMES.get(s.category, s.category) for s in sub)
                ),
                "qa_idx": [s.qa_idx for s in sub],
            }
        )
    manifest["total_samples"] = len(samples)
    return {"samples": samples, "manifest": manifest}
