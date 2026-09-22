"""
Minimal LLM-judge for QA correctness against LoCoMo's gold answer.

This is a FRESH implementation, not the original mem_bench benchmark's judge
(that package's source evaporated - see run_benchmark.py's module docstring).
Absolute accuracy numbers here are therefore not directly comparable to the
earlier BM25/hydradna run in benchmarks/results/locomo_subset_fix/ - only the
three strategies run against THIS judge, in THIS run, are comparable to each
other, which is the actual thing this benchmark is testing.

Judge token cost is tracked but reported separately - it's a benchmarking
device, not an operational cost of any retrieval strategy.
"""

from typing import Tuple

from llm_client import JUDGE_MODEL, Usage, chat

JUDGE_SYSTEM = (
    "You grade a candidate answer against a gold reference answer for a "
    "question-answering benchmark. Respond with exactly one line: "
    "'CORRECT' or 'INCORRECT'. Minor phrasing/format differences (e.g. "
    "'7 May 2023' vs 'May 7, 2023') are CORRECT if they convey the same "
    "fact. A vague, missing, or contradictory answer is INCORRECT."
)


def judge(question: str, gold: str, candidate: str) -> Tuple[bool, Usage]:
    user = f"Question: {question}\nGold answer: {gold}\nCandidate answer: {candidate}\n\nVerdict:"
    r = chat(JUDGE_SYSTEM, user, model=JUDGE_MODEL)
    correct = r.text.strip().upper().startswith("CORRECT")
    return correct, r.usage
