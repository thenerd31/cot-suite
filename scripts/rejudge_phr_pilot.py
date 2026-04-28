"""Pilot re-judge of Qwen3-Thinking-14B PHR with the corrected parser.

Re-judges ONLY the trajectories where the corrected parser changed the
final-answer letter AND the new letter is scorable (not "").
Trajectories where the parser is unchanged keep their existing
judgments — same final_answer in same prompt = same judgment.
Trajectories where the parser now returns "" are flagged unscorable
without judging (no PHR signal possible without a final commitment).

Output: ``post_hoc_rationalization_v2.jsonl`` with the merged judgments
plus a ``judge_method`` field tagging each row as ``"existing"`` or
``"rejudged_v2"`` for transparency.

Usage:
    PYTHONPATH=. python scripts/rejudge_phr_pilot.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from cotsuite.core.trajectory import Reasoning, Trajectory, Turn
from cotsuite.parsing import extract_answer_letter
from cotsuite.tests.post_hoc_rationalization import post_hoc_rationalization

RESULTS_PATH = Path("benchmarks/results/qwen3_14b_gpqa_full/results.jsonl")
EXISTING_JUDGE_PATH = Path(
    "benchmarks/results/qwen3_14b_gpqa_full/post_hoc_rationalization.jsonl"
)
OUTPUT_PATH = Path(
    "benchmarks/results/qwen3_14b_gpqa_full/post_hoc_rationalization_v2.jsonl"
)
CONCURRENCY = 4


def _row_to_trajectory(row: dict, final_answer: str) -> Trajectory:
    """Build a Trajectory using the corrected final_answer."""
    return Trajectory(
        turns=[
            Turn(role="user", text=row["question_text"]),
            Turn(
                role="assistant",
                text=row["raw_model_content"],
                reasoning=[
                    Reasoning(text=row["raw_cot"], provider="qwen", is_summary=False),
                ],
            ),
        ],
        final_answer=final_answer,
    )


async def _judge_one(row: dict, new_fa: str, sem: asyncio.Semaphore) -> dict:
    """Re-judge one trajectory with the corrected final_answer."""
    async with sem:
        traj = _row_to_trajectory(row, new_fa)
        try:
            result = await post_hoc_rationalization(traj)
            return {
                "question_id": row["question_id"],
                "is_correct": new_fa == row["correct_answer"],
                "final_answer": new_fa,
                "cot_conclusion": result.cot_conclusion,
                "diverged": result.diverged,
                "acknowledged": result.acknowledged,
                "confidence": result.confidence,
                "judge_reasoning": result.judge_reasoning,
                "raw_response": result.autorater_raw_response,
                "error": None,
                "judge_method": "rejudged_v2",
            }
        except Exception as exc:
            return {
                "question_id": row["question_id"],
                "is_correct": new_fa == row["correct_answer"],
                "final_answer": new_fa,
                "cot_conclusion": None,
                "diverged": None,
                "acknowledged": None,
                "confidence": None,
                "judge_reasoning": None,
                "raw_response": None,
                "error": f"{type(exc).__name__}: {exc}",
                "judge_method": "rejudged_v2",
            }


async def _main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    existing = {
        json.loads(line)["question_id"]: json.loads(line)
        for line in EXISTING_JUDGE_PATH.read_text().splitlines()
        if line.strip()
    }

    # Categorize each trajectory.
    rejudge_targets: list[tuple[dict, str]] = []
    new_unscorable: list[dict] = []
    unchanged: list[dict] = []

    for row in rows:
        mc = row.get("raw_model_content", "")
        if not mc:
            continue
        new_fa = extract_answer_letter(mc)
        old_fa = row["final_answer"]

        if new_fa == old_fa:
            unchanged.append(row)
        elif new_fa == "":
            new_unscorable.append(row)
        else:
            rejudge_targets.append((row, new_fa))

    print(
        f"Pilot inputs: {len(rows)} total trajectories",
        f"  unchanged (keep existing judgment):     {len(unchanged)}",
        f"  newly unscorable (skip judgment):       {len(new_unscorable)}",
        f"  to re-judge with corrected final_answer: {len(rejudge_targets)}",
        sep="\n",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # Re-judge the changed-and-scorable subset.
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_judge_one(row, new_fa, sem) for row, new_fa in rejudge_targets]
    rejudged: list[dict] = []
    for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
        result = await fut
        rejudged.append(result)
        tag = "ERR" if result["error"] else (
            "DIV!" if result["diverged"] and not result["acknowledged"]
            else ("div-ack" if result["diverged"] else "ok")
        )
        print(
            f"  [{idx}/{len(rejudge_targets)}] {result['question_id']} {tag}",
            file=sys.stderr,
        )

    # Merge: existing judgments for unchanged + new for re-judged + unscorable stubs.
    rejudged_by_id = {r["question_id"]: r for r in rejudged}
    merged: list[dict] = []
    for row in rows:
        qid = row["question_id"]
        mc = row.get("raw_model_content", "")
        if not mc:
            continue
        new_fa = extract_answer_letter(mc)
        if qid in rejudged_by_id:
            merged.append(rejudged_by_id[qid])
        elif new_fa == "":
            merged.append({
                "question_id": qid,
                "is_correct": False,
                "final_answer": "",
                "cot_conclusion": None,
                "diverged": None,
                "acknowledged": None,
                "confidence": None,
                "judge_reasoning": "Unscorable: corrected parser returned no final answer.",
                "raw_response": None,
                "error": None,
                "judge_method": "skip_unscorable_v2",
            })
        else:
            ex = existing.get(qid, {})
            ex_out = dict(ex)
            ex_out["judge_method"] = "existing"
            ex_out["is_correct"] = new_fa == row["correct_answer"]
            merged.append(ex_out)

    OUTPUT_PATH.write_text("\n".join(json.dumps(r) for r in merged) + "\n")

    # Compute and report headline numbers.
    correct = [r for r in merged if r.get("is_correct")]
    correct_strict_phr = [
        r for r in correct
        if r.get("diverged") and not r.get("acknowledged")
    ]

    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print("Pilot results — Qwen3-Thinking-14B (corrected parser)", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Total trajectories:               {len(merged)}", file=sys.stderr)
    print(f"Correct under v2 parser:          {len(correct)}", file=sys.stderr)
    print(f"Strict PHR on correct:            {len(correct_strict_phr)}", file=sys.stderr)
    if correct:
        rate = 100 * len(correct_strict_phr) / len(correct)
        print(f"Strict PHR rate:                  {rate:.2f}%", file=sys.stderr)
    print(f"Strict PHR question_ids:          {[r['question_id'] for r in correct_strict_phr]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
