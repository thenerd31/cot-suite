"""Full re-judge of all 8 models + B4 GPT-4o-mini validation with the corrected parser.

Generalization of ``scripts/rejudge_phr_pilot.py``. For each model:

1. Re-extract ``final_answer`` from ``raw_model_content`` using the
   corrected parser (``cotsuite.parsing.extract_answer_letter``).
2. Re-judge ONLY trajectories where the final_answer changed AND the
   new value is scorable (not "").
3. Trajectories where the parser is unchanged keep their existing
   judgments — same final_answer in the same prompt = same judgment.
4. Trajectories where the parser now returns "" are flagged unscorable
   without judging.

For the B4 validation: same algorithm but the source files live under
``validation/b4_arcuschin_raw.jsonl`` and ``raw_body`` is the GPT-4o-mini
output field (rather than ``raw_model_content``).

Output: ``post_hoc_rationalization_v2.jsonl`` per model, plus a
combined ``rejudge_v2_summary.json`` with the corrected PHR rates.

Usage:
    PYTHONPATH=. python scripts/rejudge_phr_full.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

from cotsuite.core.trajectory import Reasoning, Trajectory, Turn
from cotsuite.parsing import extract_answer_letter
from cotsuite.tests.post_hoc_rationalization import post_hoc_rationalization

CONCURRENCY = 4

# (results_path, existing_judge_path, output_path, raw_field, name)
TARGETS: list[tuple[Path, Path, Path, str, str]] = [
    (
        Path("benchmarks/results/qwen3_8b_gpqa_full/results.jsonl"),
        Path("benchmarks/results/qwen3_8b_gpqa_full/post_hoc_rationalization.jsonl"),
        Path("benchmarks/results/qwen3_8b_gpqa_full/post_hoc_rationalization_v2.jsonl"),
        "raw_model_content",
        "Qwen3-Thinking-8B",
    ),
    (
        Path("benchmarks/results/qwen3_32b_gpqa_full/results.jsonl"),
        Path("benchmarks/results/qwen3_32b_gpqa_full/post_hoc_rationalization.jsonl"),
        Path("benchmarks/results/qwen3_32b_gpqa_full/post_hoc_rationalization_v2.jsonl"),
        "raw_model_content",
        "Qwen3-Thinking-32B",
    ),
    (
        Path("benchmarks/results/ds_r1_distill_qwen_14b_full/results.jsonl"),
        Path(
            "benchmarks/results/ds_r1_distill_qwen_14b_full/"
            "post_hoc_rationalization.jsonl"
        ),
        Path(
            "benchmarks/results/ds_r1_distill_qwen_14b_full/"
            "post_hoc_rationalization_v2.jsonl"
        ),
        "raw_model_content",
        "DS-R1-Distill-Qwen-14B",
    ),
    (
        Path("benchmarks/results/ds_r1_distill_llama_70b_full/results.jsonl"),
        Path(
            "benchmarks/results/ds_r1_distill_llama_70b_full/"
            "post_hoc_rationalization.jsonl"
        ),
        Path(
            "benchmarks/results/ds_r1_distill_llama_70b_full/"
            "post_hoc_rationalization_v2.jsonl"
        ),
        "raw_model_content",
        "DS-R1-Distill-Llama-70B",
    ),
    (
        Path("benchmarks/results/qwen25_7b_instruct_full/results.jsonl"),
        Path(
            "benchmarks/results/qwen25_7b_instruct_full/post_hoc_rationalization.jsonl"
        ),
        Path(
            "benchmarks/results/qwen25_7b_instruct_full/"
            "post_hoc_rationalization_v2.jsonl"
        ),
        "raw_model_content",
        "Qwen2.5-7B-Instruct",
    ),
    (
        Path("benchmarks/results/qwen25_72b_instruct_full/results.jsonl"),
        Path(
            "benchmarks/results/qwen25_72b_instruct_full/"
            "post_hoc_rationalization.jsonl"
        ),
        Path(
            "benchmarks/results/qwen25_72b_instruct_full/"
            "post_hoc_rationalization_v2.jsonl"
        ),
        "raw_model_content",
        "Qwen2.5-72B-Instruct",
    ),
    (
        Path("benchmarks/results/llama31_8b_instruct_full/results.jsonl"),
        Path(
            "benchmarks/results/llama31_8b_instruct_full/post_hoc_rationalization.jsonl"
        ),
        Path(
            "benchmarks/results/llama31_8b_instruct_full/"
            "post_hoc_rationalization_v2.jsonl"
        ),
        "raw_model_content",
        "Llama-3.1-8B-Instruct",
    ),
]


def _row_to_trajectory(
    row: dict, final_answer: str, raw_field: str, *, b4_mode: bool = False
) -> Trajectory:
    """Convert a results row into a cotsuite.Trajectory using the corrected final_answer."""
    if b4_mode:
        # B4 schema: question + raw_body. Reasoning derived by splitting on "Final Answer:".
        question = row.get("mcq_prompt", "")
        body = row.get("raw_body", "")
        reasoning_text = row.get("reasoning", "") or body
        return Trajectory(
            turns=[
                Turn(role="user", text=question),
                Turn(
                    role="assistant",
                    text=body,
                    reasoning=[Reasoning(text=reasoning_text, provider="openai")],
                ),
            ],
            final_answer=final_answer,
        )
    return Trajectory(
        turns=[
            Turn(role="user", text=row["question_text"]),
            Turn(
                role="assistant",
                text=row[raw_field],
                reasoning=[
                    Reasoning(text=row["raw_cot"], provider="qwen", is_summary=False),
                ],
            ),
        ],
        final_answer=final_answer,
    )


async def _judge_one(
    row: dict,
    new_fa: str,
    correct_answer: str,
    raw_field: str,
    sem: asyncio.Semaphore,
    *,
    b4_mode: bool = False,
) -> dict:
    """Re-judge one trajectory."""
    async with sem:
        traj = _row_to_trajectory(row, new_fa, raw_field, b4_mode=b4_mode)
        try:
            result = await post_hoc_rationalization(traj)
            return {
                "question_id": row["question_id"],
                "is_correct": new_fa == correct_answer and bool(new_fa),
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
                "is_correct": new_fa == correct_answer and bool(new_fa),
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


async def _process_one_model(
    results_path: Path,
    existing_judge_path: Path,
    output_path: Path,
    raw_field: str,
    name: str,
    *,
    b4_mode: bool = False,
) -> dict:
    """Re-judge one model run + write corrected JSONL + return summary dict."""
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    existing = {}
    if existing_judge_path.exists():
        existing = {
            json.loads(line)["question_id"]: json.loads(line)
            for line in existing_judge_path.read_text().splitlines()
            if line.strip()
        }

    rejudge_targets: list[tuple[dict, str]] = []
    for row in rows:
        if b4_mode:
            mc = row.get("raw_body", "")
        else:
            mc = row.get(raw_field, "")
        if not mc:
            continue
        new_fa = extract_answer_letter(mc)
        old_fa = row.get("final_answer", "")
        if new_fa != old_fa and new_fa:
            rejudge_targets.append((row, new_fa))

    print(
        f"[{name}] rows={len(rows)} to_rejudge={len(rejudge_targets)}",
        file=sys.stderr,
    )

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [
        _judge_one(row, new_fa, row["correct_answer"], raw_field, sem, b4_mode=b4_mode)
        for row, new_fa in rejudge_targets
    ]
    rejudged: list[dict] = []
    for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
        result = await fut
        rejudged.append(result)
        tag = "ERR" if result["error"] else (
            "DIV!" if result["diverged"] and not result["acknowledged"]
            else ("div-ack" if result["diverged"] else "ok")
        )
        print(f"  [{name} {idx}/{len(rejudge_targets)}] {result['question_id']} {tag}", file=sys.stderr)

    rejudged_by_id = {r["question_id"]: r for r in rejudged}
    merged: list[dict] = []
    for row in rows:
        qid = row["question_id"]
        if b4_mode:
            mc = row.get("raw_body", "")
        else:
            mc = row.get(raw_field, "")
        if not mc:
            continue
        new_fa = extract_answer_letter(mc)
        correct = row["correct_answer"]
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
            ex_out["is_correct"] = new_fa == correct and bool(new_fa)
            merged.append(ex_out)

    output_path.write_text("\n".join(json.dumps(r) for r in merged) + "\n")

    # Summary
    correct = [r for r in merged if r.get("is_correct")]
    correct_strict_phr = [
        r for r in correct
        if r.get("diverged") and not r.get("acknowledged")
    ]
    correct_phr_inclack = [r for r in correct if r.get("diverged")]
    return OrderedDict([
        ("model", name),
        ("n_total", len(merged)),
        ("n_correct", len(correct)),
        ("phr_strict_n", len(correct_strict_phr)),
        ("phr_strict_pct", 100 * len(correct_strict_phr) / len(correct) if correct else 0),
        ("phr_inclack_n", len(correct_phr_inclack)),
        ("phr_inclack_pct", 100 * len(correct_phr_inclack) / len(correct) if correct else 0),
        ("phr_strict_qids", [r["question_id"] for r in correct_strict_phr]),
        ("rejudged_count", len(rejudge_targets)),
    ])


async def _main() -> int:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    summaries: list[dict] = []
    for results_path, existing, output, raw_field, name in TARGETS:
        s = await _process_one_model(results_path, existing, output, raw_field, name)
        summaries.append(s)

    # Add Qwen3-Thinking-14B from the pilot run.
    pilot_v2 = Path(
        "benchmarks/results/qwen3_14b_gpqa_full/post_hoc_rationalization_v2.jsonl"
    )
    if pilot_v2.exists():
        rows = [json.loads(line) for line in pilot_v2.read_text().splitlines() if line.strip()]
        correct = [r for r in rows if r.get("is_correct")]
        strict = [r for r in correct if r.get("diverged") and not r.get("acknowledged")]
        inclack = [r for r in correct if r.get("diverged")]
        summaries.insert(1, OrderedDict([
            ("model", "Qwen3-Thinking-14B"),
            ("n_total", len(rows)),
            ("n_correct", len(correct)),
            ("phr_strict_n", len(strict)),
            ("phr_strict_pct", 100 * len(strict) / len(correct) if correct else 0),
            ("phr_inclack_n", len(inclack)),
            ("phr_inclack_pct", 100 * len(inclack) / len(correct) if correct else 0),
            ("phr_strict_qids", [r["question_id"] for r in strict]),
            ("rejudged_count", "13 (pilot)"),
        ]))

    # B4 validation re-run.
    b4_results = Path("validation/b4_arcuschin_raw.jsonl")
    b4_v2 = Path("validation/b4_arcuschin_raw_v2.jsonl")
    if b4_results.exists():
        s = await _process_one_model(
            b4_results, Path("/dev/null"), b4_v2, "raw_body", "B4-GPT-4o-mini",
            b4_mode=True,
        )
        summaries.append(s)

    summary_path = Path("benchmarks/results/rejudge_v2_summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")

    print(file=sys.stderr)
    print("=" * 78, file=sys.stderr)
    print(f'{"model":<28} {"n_corr":>6} {"phr_strict":>10} {"strict_%":>8} {"phr_inclack":>11} {"inclack_%":>9}', file=sys.stderr)
    print("-" * 78, file=sys.stderr)
    for s in summaries:
        print(
            f'{s["model"]:<28} {s["n_correct"]:>6} {s["phr_strict_n"]:>10} '
            f'{s["phr_strict_pct"]:>7.2f}% {s["phr_inclack_n"]:>11} {s["phr_inclack_pct"]:>8.2f}%',
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
