"""One-off runner: post-hoc rationalization detector on Stage 1 trajectories.

Loads benchmarks/results/qwen3_14b_gpqa_full/results.jsonl, converts each
correct-answer row into a Trajectory, runs the Arcuschin-inspired LLM-as-
judge detector (Haiku 4.5), and reports counts.

Usage:
    PYTHONPATH=. python scripts/run_post_hoc_rationalization.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from cotsuite.core.trajectory import Reasoning, Trajectory, Turn
from cotsuite.tests.post_hoc_rationalization import post_hoc_rationalization

RESULTS_PATH = Path("benchmarks/results/qwen3_14b_gpqa_full/results.jsonl")
OUTPUT_PATH = Path("benchmarks/results/qwen3_14b_gpqa_full/post_hoc_rationalization.jsonl")
CONCURRENCY = 8


def _row_to_trajectory(row: dict) -> Trajectory:
    """Convert a Stage-1 RunRow dict into a cotsuite.Trajectory."""
    return Trajectory(
        turns=[
            Turn(role="user", text=row["question_text"]),
            Turn(
                role="assistant",
                text=row["raw_model_content"],
                reasoning=[
                    Reasoning(
                        text=row["raw_cot"],
                        provider="qwen",
                        is_summary=False,
                    ),
                ],
            ),
        ],
        final_answer=row["final_answer"],
    )


async def _process_one(row: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        traj = _row_to_trajectory(row)
        try:
            result = await post_hoc_rationalization(traj)
            return {
                "question_id": row["question_id"],
                "is_correct": row["is_correct"],
                "final_answer": row["final_answer"],
                "cot_conclusion": result.cot_conclusion,
                "diverged": result.diverged,
                "acknowledged": result.acknowledged,
                "confidence": result.confidence,
                "judge_reasoning": result.judge_reasoning,
                "raw_response": result.autorater_raw_response,
                "error": None,
            }
        except Exception as exc:
            return {
                "question_id": row["question_id"],
                "is_correct": row["is_correct"],
                "final_answer": row.get("final_answer"),
                "cot_conclusion": None,
                "diverged": None,
                "acknowledged": None,
                "confidence": None,
                "judge_reasoning": None,
                "raw_response": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def _main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2
    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    # Skip the inference-timeout row (no reasoning to evaluate).
    rows = [r for r in rows if not r.get("inference_timeout", False)]

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_process_one(r, sem) for r in rows]
    out: list[dict] = []
    for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
        result = await fut
        out.append(result)
        tag = (
            "ERR"
            if result["error"]
            else (
                "DIV!"
                if result["diverged"] and not result["acknowledged"]
                else ("div-ack" if result["diverged"] else "ok")
            )
        )
        print(
            f"  [{idx}/{len(rows)}] {result['question_id']} {tag}",
            file=sys.stderr,
        )

    # Preserve input order.
    by_id = {r["question_id"]: r for r in out}
    ordered = [by_id[r["question_id"]] for r in rows]
    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in ordered) + "\n")

    _report(ordered)
    return 0


def _report(rows: list[dict]) -> None:
    total = len(rows)
    errors = [r for r in rows if r["error"]]
    usable = [r for r in rows if not r["error"]]
    correct = [r for r in usable if r["is_correct"]]
    incorrect = [r for r in usable if not r["is_correct"]]

    def _counts(rs: list[dict]) -> dict:
        diverged = [r for r in rs if r["diverged"]]
        div_unack = [r for r in diverged if not r["acknowledged"]]
        div_ack = [r for r in diverged if r["acknowledged"]]
        aligned = [r for r in rs if not r["diverged"]]
        return {
            "n": len(rs),
            "aligned": len(aligned),
            "diverged_acknowledged": len(div_ack),
            "diverged_unacknowledged_the_pattern": len(div_unack),
            "diverged_total": len(diverged),
            "div_unack_ids": [r["question_id"] for r in div_unack],
            "div_ack_ids": [r["question_id"] for r in div_ack],
        }

    print("=" * 70)
    print(f"Processed {total} trajectories ({len(errors)} judge errors, {len(usable)} usable).")
    print("=" * 70)
    print()
    print("CORRECT trajectories (Arcuschin pattern on correct answers):")
    cc = _counts(correct)
    for k in (
        "n",
        "aligned",
        "diverged_total",
        "diverged_acknowledged",
        "diverged_unacknowledged_the_pattern",
    ):
        print(f"  {k}: {cc[k]}")
    print(f"  3 examples of the pattern: {cc['div_unack_ids'][:3]}")
    print()
    print("INCORRECT trajectories:")
    ic = _counts(incorrect)
    for k in (
        "n",
        "aligned",
        "diverged_total",
        "diverged_acknowledged",
        "diverged_unacknowledged_the_pattern",
    ):
        print(f"  {k}: {ic[k]}")
    print(f"  3 examples of the pattern: {ic['div_unack_ids'][:3]}")
    print()
    print("Confidence distribution on div-unack cases:")
    conf_correct = [r["confidence"] for r in correct if r["diverged"] and not r["acknowledged"]]
    conf_incorrect = [r["confidence"] for r in incorrect if r["diverged"] and not r["acknowledged"]]
    if conf_correct:
        print(f"  correct mean conf: {sum(conf_correct) / len(conf_correct):.3f}")
    if conf_incorrect:
        print(f"  incorrect mean conf: {sum(conf_incorrect) / len(conf_incorrect):.3f}")
    print()
    if errors:
        print(f"Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e['question_id']}: {e['error']}")
    print()
    print("=" * 70)
    print("Correct+diverged+unacknowledged — 3 examples with judge reasoning:")
    print("=" * 70)
    for r in correct:
        if r["diverged"] and not r["acknowledged"]:
            print(f"\n--- {r['question_id']} (conf {r['confidence']:.2f}) ---")
            print(f"  cot_conclusion={r['cot_conclusion']!r}  final_answer={r['final_answer']!r}")
            print(f"  judge_reasoning: {r['judge_reasoning']}")
            if correct.index(r) >= 2 or (correct.index(r) + 1) >= 3:
                break

    # Cot_conclusion distribution for correct+diverged
    cc_dist = Counter(
        r["cot_conclusion"] for r in correct if r["diverged"] and not r["acknowledged"]
    )
    if cc_dist:
        print(f"\ncot_conclusion distribution across correct+div+unack: {dict(cc_dist)}")


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
