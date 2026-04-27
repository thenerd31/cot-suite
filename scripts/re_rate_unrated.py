"""Re-rate unrated correct rows in a completed run's results.jsonl.

Targets rows where ``is_correct=True`` but ``autorater_legibility`` is None
— these are typically rows that hit an Anthropic BadRequestError mid-run
(credit exhaustion, 2026-04-23) or rows whose autorater output couldn't
be parsed (e.g. Haiku returned JSON with un-escaped newlines in string
values). Reuses ``autorater_fn`` from ``run_qwen3_gpqa.py`` so the
prompt, model, and parser are identical to the live run.

Usage:
    python scripts/re_rate_unrated.py \\
        --results benchmarks/results/qwen3_32b_gpqa_full/results.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from cotsuite.autoraters.legibility_coverage import LegibilityCoveragePrompt
from cotsuite.verify_keys import require_keys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_qwen3_gpqa import autorater_fn, format_mcq_prompt, load_gpqa_diamond


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    require_keys(["anthropic", "huggingface"])

    with args.results.open() as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    unrated_correct = [
        (i, r) for i, r in enumerate(rows)
        if r["is_correct"] and r.get("autorater_legibility") is None
    ]
    if not unrated_correct:
        print("no unrated correct rows; nothing to do.")
        return 0
    print(f"found {len(unrated_correct)} unrated correct rows to re-rate")

    # load_gpqa_diamond returns Samples with correctly shuffled options,
    # keyed by position. Row's question_id ("gpqa_diamond_NNN") maps to
    # position NNN.
    samples = load_gpqa_diamond(limit=None, stub=False)
    sample_by_id = {s.question_id: s for s in samples}

    prompt = LegibilityCoveragePrompt.load()
    autorater = autorater_fn(stub=False)
    sem = asyncio.Semaphore(args.concurrency)

    async def _rerate(idx: int, row: dict) -> tuple[int, dict]:
        async with sem:
            sample = sample_by_id[row["question_id"]]
            mcq = format_mcq_prompt(sample)
            rendered = prompt.render(
                question=mcq,
                explanation=row["raw_cot"],
                answer=row["raw_model_content"],
            )
            try:
                raw, leg, cov, just, parse_err = await autorater(rendered)
                row["autorater_raw_response"] = raw
                row["autorater_legibility"] = leg
                row["autorater_coverage"] = cov
                row["autorater_justification"] = just
                row["parse_error"] = parse_err
                tag = f"leg={leg} cov={cov}" if leg is not None else f"PARSE_FAIL: {parse_err[:60]}"
            except Exception as exc:
                row["parse_error"] = f"{type(exc).__name__}: {exc}"
                tag = f"ERR: {exc}"
            print(f"  [{idx}] {row['question_id']} {tag}", file=sys.stderr)
            return idx, row

    results = await asyncio.gather(
        *[_rerate(i, r) for i, r in unrated_correct],
    )
    for i, r in results:
        rows[i] = r

    args.results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    now_rated = sum(1 for r in rows if r["is_correct"] and r.get("autorater_legibility") is not None)
    still_unrated = sum(
        1 for r in rows if r["is_correct"] and r.get("autorater_legibility") is None
    )
    print(f"done. correct-rated: {now_rated}, still-unrated: {still_unrated}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
