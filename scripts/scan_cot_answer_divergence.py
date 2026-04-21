"""One-shot recon: does Qwen3-14B's CoT-implied answer diverge from its final
answer on our Stage 1 GPQA-Diamond results?

Throwaway. Not a library module. Not tested. Not imported anywhere.
Inspired by the Arcuschin et al. (2503.08679) "Implicit Post-Hoc Rationalization"
finding — if the CoT concludes C but the model outputs D, that's the pattern.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/scan_cot_answer_divergence.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path("benchmarks/results/qwen3_14b_gpqa_full/results.jsonl")

# Last-mention regex over the trailing 500 chars. Matches:
#   "the answer is (A)" / "answer is: B" / "(C)" / "Answer: D"
# Case-insensitive. Takes the LAST match in the window as the CoT conclusion.
_CAND = re.compile(
    r"(?:\banswer\s*(?:is|:)?\s*|\\boxed\{|\b)\(?([A-Da-d])\)?",
    re.IGNORECASE,
)


def cot_implied_answer(cot: str) -> str | None:
    if not cot:
        return None
    tail = cot[-500:]
    matches = list(_CAND.finditer(tail))
    if not matches:
        return None
    return matches[-1].group(1).upper()


def classify(cot_ans: str | None, final_ans: str) -> str:
    if cot_ans is None or not cot_ans:
        return "cot_unclear"
    if not final_ans:
        return "cot_unclear"
    if cot_ans == final_ans:
        return "aligned"
    return "diverged"


def main() -> None:
    rows = [json.loads(line) for line in RESULTS.open()]
    cells: dict[tuple[bool, str], list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("inference_timeout"):
            continue
        cot_ans = cot_implied_answer(r["raw_cot"])
        cls = classify(cot_ans, r["final_answer"])
        cells[(r["is_correct"], cls)].append({**r, "_cot_implied": cot_ans})

    print("=" * 70)
    print(
        f"Scanned {sum(len(v) for v in cells.values())} usable trajectories "
        f"(excluding {sum(1 for r in rows if r.get('inference_timeout'))} timeout)"
    )
    print("=" * 70)

    order = [
        (True, "aligned"),
        (True, "diverged"),
        (True, "cot_unclear"),
        (False, "aligned"),
        (False, "diverged"),
        (False, "cot_unclear"),
    ]
    for key in order:
        correct, cls = key
        bucket = cells[key]
        label = f"{'Correct' if correct else 'Incorrect'} + {cls}"
        print(f"\n{label}: {len(bucket)}")
        examples = bucket[:3]
        for ex in examples:
            print(
                f"  - {ex['question_id']}  "
                f"cot_implied={ex['_cot_implied']!r}  "
                f"final={ex['final_answer']!r}  "
                f"correct_answer={ex['correct_answer']!r}",
            )

    # Specifically: dump the last 300 chars of CoT + final answer for
    # every correct+diverged and incorrect+diverged example.
    for correct in (True, False):
        bucket = cells[(correct, "diverged")]
        if not bucket:
            continue
        heading = f"{'CORRECT' if correct else 'INCORRECT'} + DIVERGED — qualitative evidence"
        print(f"\n{'=' * 70}\n{heading}\n{'=' * 70}")
        for ex in bucket[:3]:
            print(f"\n--- {ex['question_id']} ---")
            print(f"  cot_implied_answer = {ex['_cot_implied']!r}")
            print(f"  final_answer       = {ex['final_answer']!r}")
            print(f"  correct_answer     = {ex['correct_answer']!r}")
            print(f"  is_correct         = {ex['is_correct']}")
            tail = ex["raw_cot"][-300:]
            print("  last 300 chars of CoT:")
            print("  " + tail.replace("\n", "\n  "))

    # Also check gpqa_diamond_001 specifically (the motivating example).
    print(f"\n{'=' * 70}\nDirect check on gpqa_diamond_001\n{'=' * 70}")
    q001 = next((r for r in rows if r["question_id"] == "gpqa_diamond_001"), None)
    if q001 is None:
        print("  NOT FOUND")
    else:
        cot_ans = cot_implied_answer(q001["raw_cot"])
        cls = classify(cot_ans, q001["final_answer"])
        print(
            f"  cot_implied={cot_ans!r}  final={q001['final_answer']!r}  "
            f"correct={q001['correct_answer']!r}  "
            f"is_correct={q001['is_correct']}  classification={cls}"
        )
        print("  last 500 chars of CoT:")
        print("  " + q001["raw_cot"][-500:].replace("\n", "\n  "))


if __name__ == "__main__":
    main()
