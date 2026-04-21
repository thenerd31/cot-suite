"""Cross-rater validation: re-score the 120 Haiku-rated Stage 1 trajectories
with Gemini 2.5 Pro (the paper's original autorater), then compare.

Purpose: the launch's "we validated across two autoraters" claim. If Gemini
and Haiku agree within a tight margin, the single-rater Stage 1 result is
methodologically defensible. If they diverge, we've found a cross-rater
bias finding that's worth discussing separately.

Tripwire: if per-axis mean |Gemini - Haiku| exceeds 0.2, the script prints a
STOP banner and exits nonzero — a tighter threshold than the coarse "more
than 1 point of variance" we tolerated within a single rater.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
from pathlib import Path

from cotmon.autoraters.legibility_coverage import LegibilityCoveragePrompt

RESULTS_PATH = Path("benchmarks/results/qwen3_14b_gpqa_full/results.jsonl")
OUT_PATH = Path("benchmarks/results/qwen3_14b_gpqa_full/gemini_cross_rater.jsonl")
MODEL = "gemini-2.5-pro"
CONCURRENCY = 4  # Gemini's default tier is strict on concurrent requests
AXIS_DIVERGE_TRIPWIRE = 0.2


async def _rate_one(row: dict, prompt: LegibilityCoveragePrompt, sem: asyncio.Semaphore) -> dict:
    from google import genai

    async with sem:
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        # Reconstruct the MCQ prompt as the driver would.
        # For the cross-rater we don't need to re-extract; we just need the
        # same autorater inputs: question, reasoning, answer.
        question = row["question_text"]
        reasoning = row["raw_cot"]
        answer = row["raw_model_content"]
        rendered = prompt.render(
            question=question,
            explanation=reasoning,
            answer=answer,
        )
        try:
            response = await client.aio.models.generate_content(
                model=MODEL,
                contents=rendered,
            )
            raw = response.text or ""
            leg, cov, justification = prompt.parse(raw)
            return {
                "question_id": row["question_id"],
                "haiku_leg": row["autorater_legibility"],
                "haiku_cov": row["autorater_coverage"],
                "gemini_leg": leg,
                "gemini_cov": cov,
                "gemini_justification": justification,
                "gemini_raw_response": raw,
                "error": None,
            }
        except Exception as exc:
            return {
                "question_id": row["question_id"],
                "haiku_leg": row.get("autorater_legibility"),
                "haiku_cov": row.get("autorater_coverage"),
                "gemini_leg": None,
                "gemini_cov": None,
                "gemini_justification": None,
                "gemini_raw_response": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


async def _main() -> int:
    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.", file=sys.stderr)
        return 2

    prompt = LegibilityCoveragePrompt.load()
    rows = [json.loads(line) for line in RESULTS_PATH.read_text().splitlines() if line.strip()]
    rated = [r for r in rows if r.get("autorater_legibility") is not None]
    print(f"Re-rating {len(rated)} trajectories via {MODEL}...", file=sys.stderr)

    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_rate_one(r, prompt, sem) for r in rated]
    out: list[dict] = []
    for idx, fut in enumerate(asyncio.as_completed(tasks), start=1):
        res = await fut
        out.append(res)
        tag = "ERR" if res["error"] else f"H{res['haiku_leg']}{res['haiku_cov']} G{res['gemini_leg']}{res['gemini_cov']}"
        print(f"  [{idx}/{len(rated)}] {res['question_id']} {tag}", file=sys.stderr)

    # Preserve input order
    by_id = {r["question_id"]: r for r in out}
    ordered = [by_id[r["question_id"]] for r in rated if r["question_id"] in by_id]
    OUT_PATH.write_text("\n".join(json.dumps(r) for r in ordered) + "\n")

    _report(ordered)
    return 0


def _report(rows: list[dict]) -> None:
    ok = [r for r in rows if r["error"] is None]
    errors = [r for r in rows if r["error"]]

    haiku_legs = [r["haiku_leg"] for r in ok]
    haiku_covs = [r["haiku_cov"] for r in ok]
    gemini_legs = [r["gemini_leg"] for r in ok]
    gemini_covs = [r["gemini_cov"] for r in ok]

    leg_deltas = [abs(h - g) for h, g in zip(haiku_legs, gemini_legs, strict=True)]
    cov_deltas = [abs(h - g) for h, g in zip(haiku_covs, gemini_covs, strict=True)]

    leg_mean_delta = statistics.fmean(leg_deltas) if leg_deltas else 0.0
    cov_mean_delta = statistics.fmean(cov_deltas) if cov_deltas else 0.0

    print("=" * 70)
    print(f"Cross-rater comparison: {len(ok)} usable / {len(rows)} attempted ({len(errors)} errors)")
    print("=" * 70)
    print()
    print("Haiku 4.5 (Stage 1 values):")
    print(f"  legibility mean = {statistics.fmean(haiku_legs):.3f} sd = {statistics.stdev(haiku_legs):.3f}")
    print(f"  coverage   mean = {statistics.fmean(haiku_covs):.3f} sd = {statistics.stdev(haiku_covs):.3f}")
    print()
    print(f"Gemini 2.5 Pro ({MODEL}):")
    print(f"  legibility mean = {statistics.fmean(gemini_legs):.3f} sd = {statistics.stdev(gemini_legs):.3f}")
    print(f"  coverage   mean = {statistics.fmean(gemini_covs):.3f} sd = {statistics.stdev(gemini_covs):.3f}")
    print()
    print("Per-trajectory absolute delta (|Haiku - Gemini|):")
    print(f"  legibility: mean = {leg_mean_delta:.3f}  max = {max(leg_deltas, default=0)}")
    print(f"  coverage:   mean = {cov_mean_delta:.3f}  max = {max(cov_deltas, default=0)}")
    print()
    # Distribution of per-trajectory score differences (signed, Haiku - Gemini)
    leg_signed = [h - g for h, g in zip(haiku_legs, gemini_legs, strict=True)]
    cov_signed = [h - g for h, g in zip(haiku_covs, gemini_covs, strict=True)]
    print(f"Signed delta (Haiku - Gemini), legibility: mean = {statistics.fmean(leg_signed):+.3f}")
    print(f"Signed delta (Haiku - Gemini), coverage:   mean = {statistics.fmean(cov_signed):+.3f}")
    print()

    if leg_mean_delta > AXIS_DIVERGE_TRIPWIRE or cov_mean_delta > AXIS_DIVERGE_TRIPWIRE:
        print("!" * 70)
        print(
            f"TRIPWIRE: per-axis mean |Haiku - Gemini| exceeded "
            f"{AXIS_DIVERGE_TRIPWIRE}. Stopping — this is a finding worth "
            "discussing before shipping.",
        )
        print("!" * 70)
        sys.exit(3)
    else:
        print(
            f"Both axes within tripwire ({AXIS_DIVERGE_TRIPWIRE}). Cross-rater "
            "validation clean.",
        )

    if errors:
        print()
        print(f"Errors: {len(errors)}")
        for e in errors[:3]:
            print(f"  {e['question_id']}: {e['error']}")


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
