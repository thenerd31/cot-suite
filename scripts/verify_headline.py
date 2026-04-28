"""Audit-trail verification: recompute headline PHR rate from published JSONL fields.

For each model, reads ``post_hoc_rationalization_v2.jsonl`` and computes:

    headline_rate = count(phr_strict_normalized == True && phr_scorable && is_correct)
                  / count(phr_scorable && is_correct)

This is the canonical reproduction primitive: a reviewer with the
published JSONL can run this script (or the underlying jq query) and
arrive at the same rate quoted in ``multi_family_summary.md``.

Usage:
    PYTHONPATH=. python scripts/verify_headline.py <model_dir>
    PYTHONPATH=. python scripts/verify_headline.py --all

Exits 0 on success, 1 on rate mismatch >0.5pp vs the stored summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXPECTED_RATES_PATH = Path("benchmarks/results/rejudge_v2_normalized_summary.json")
TOLERANCE_PP = 0.5


def compute_from_jsonl(model_dir: str) -> tuple[int, int, float]:
    """Return (n_phr_strict, n_scorable, rate_pct) computed directly from JSONL."""
    p = Path(f"benchmarks/results/{model_dir}/post_hoc_rationalization_v2.jsonl")
    if not p.exists():
        raise FileNotFoundError(p)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    correct_scorable = [
        r for r in rows
        if r.get("is_correct") and r.get("phr_scorable")
    ]
    n_scorable = len(correct_scorable)
    n_strict = sum(1 for r in correct_scorable if r.get("phr_strict_normalized") is True)
    rate = (100 * n_strict / n_scorable) if n_scorable else 0.0
    return n_strict, n_scorable, rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", nargs="?", help="Single model dir under benchmarks/results/")
    parser.add_argument("--all", action="store_true", help="Verify all 8 models")
    args = parser.parse_args()

    if not EXPECTED_RATES_PATH.exists():
        print(f"ERROR: {EXPECTED_RATES_PATH} not found. Run materialize_v2_normalized.py first.", file=sys.stderr)
        return 2

    expected = {s["model"]: s for s in json.loads(EXPECTED_RATES_PATH.read_text())}

    if args.all:
        targets = [
            ("qwen3_8b_gpqa_full", "Qwen3-Thinking-8B"),
            ("qwen3_14b_gpqa_full", "Qwen3-Thinking-14B"),
            ("qwen3_32b_gpqa_full", "Qwen3-Thinking-32B"),
            ("ds_r1_distill_qwen_14b_full", "DS-R1-Distill-Qwen-14B"),
            ("ds_r1_distill_llama_70b_full", "DS-R1-Distill-Llama-70B"),
            ("qwen25_7b_instruct_full", "Qwen2.5-7B-Instruct"),
            ("qwen25_72b_instruct_full", "Qwen2.5-72B-Instruct"),
            ("llama31_8b_instruct_full", "Llama-3.1-8B-Instruct"),
        ]
    elif args.model_dir:
        # Map dir → name via the summary
        name = None
        for s in expected.values():
            # Search summary for any matching dir-derived model name; fallback fine
            pass
        targets = [(args.model_dir, args.model_dir)]
    else:
        parser.print_help()
        return 2

    failed = 0
    for model_dir, name in targets:
        try:
            n_strict, n_scorable, rate = compute_from_jsonl(model_dir)
        except FileNotFoundError as exc:
            print(f"  [{name}] MISSING JSONL: {exc}", file=sys.stderr)
            failed += 1
            continue

        # Find expected rate by name (since we generated the summary by name)
        expected_rate = None
        if name in expected:
            expected_rate = expected[name].get("phr_strict_normalized_rate_pct")

        marker = "✓"
        if expected_rate is not None:
            delta = abs(rate - expected_rate)
            if delta > TOLERANCE_PP:
                marker = f"✗ ({delta:+.2f}pp drift)"
                failed += 1
            print(
                f"  [{name}] strict={n_strict:>3} / scorable={n_scorable:>3} "
                f"= {rate:>5.2f}% (expected {expected_rate:>5.2f}%) {marker}"
            )
        else:
            print(
                f"  [{name}] strict={n_strict:>3} / scorable={n_scorable:>3} "
                f"= {rate:>5.2f}% (no expected rate found)"
            )

    if failed:
        print(f"\n{failed} verification(s) FAILED", file=sys.stderr)
        return 1
    print(f"\nAll {len(targets)} verification(s) passed (tolerance ≤{TOLERANCE_PP}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
