"""$0 degeneracy-aware re-analysis of the committed E-Z cross-judge scores.

THE STATISTIC IS KNOWN. Gwet's AC1/AC2 (Gwet, 2008) is a standard prevalence-robust
agreement coefficient; the "kappa paradox" — high observed agreement with low or
negative κ under skewed marginals — is Feinstein & Cicchetti (1990). This script
contributes nothing to that statistics; it *applies* it as a per-axis degeneracy
diagnostic for CoT-monitorability metrics, and uses it to re-read the project's own
3-judge E-Z legibility/coverage result.

It recomputes — with **zero model calls**, purely from the committed
``benchmarks/results/ez_cross_judge/cross_judge_scores.jsonl`` (785 Haiku-rated GPQA
trajectories re-scored by Sonnet-4.6 + Gemini-2.5-Pro on the byte-identical
Appendix-C prompt) — for each axis (legibility, coverage) and judge pair:
quadratic-weighted Cohen's κ, Gwet's AC2 (same weights), raw observed agreement
``p_o``, and each judge's dominant-category fraction (the saturation flag,
threshold 0.70). Output: a table + ``degeneracy_reanalysis.json`` beside the
existing ``kappa_summary.json``.

The read (both directions, honestly):
- **Legibility** is saturated (every judge's dominant category > 0.70): κ collapses
  to 0.19-0.34 while p_o ≈ 0.97-0.99 and AC2 ≈ 0.96-0.99 → the low κ is a
  degeneracy/ceiling artifact, NOT substantive judge disagreement. The 36% legibility
  ranking-reversals (see kappa_summary.json) are the same artifact.
- **Coverage** has real variance (dominant fraction 0.52-0.80): κ (0.52-0.71) is
  much closer to AC2 than on legibility and the rankings are judge-stable → genuine
  agreement on the axis with signal.

Applied claim: a monitorability metric can look judge-unreliable purely because its
score distribution is saturated; report a prevalence-robust coefficient + a
saturation flag ALONGSIDE κ before concluding judge-sensitivity. Frontier-model
legibility (cf. Emmons-Zimmermann's "high monitorability") is exactly that saturated
regime.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/degeneracy_reanalysis_ez.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cotsuite.judges import judge_agreement
from cotsuite.judges.kappa import DEGENERACY_THRESHOLD
from cotsuite.judges.multi_judge import MultiJudgeResult

SCORES_PATH = Path("benchmarks/results/ez_cross_judge/cross_judge_scores.jsonl")
OUTPUT_PATH = Path("benchmarks/results/ez_cross_judge/degeneracy_reanalysis.json")
JUDGES = ("haiku", "sonnet-4.6", "gemini-2.5-pro")
AXES = ("leg", "cov")
NUM_CATEGORIES = 5  # 0-4 Likert


def _load() -> list[dict[str, Any]]:
    return [json.loads(line) for line in SCORES_PATH.read_text().splitlines() if line.strip()]


def _agreement_for_axis(records: list[dict[str, Any]], axis: str):  # type: ignore[no-untyped-def]
    survivors = [r for r in records if all(r.get(f"{j}_{axis}") is not None for j in JUDGES)]
    results = [
        MultiJudgeResult(per_judge_scores={j: float(r[f"{j}_{axis}"]) for j in JUDGES})
        for r in survivors
    ]
    return len(survivors), judge_agreement(results, num_categories=NUM_CATEGORIES)


def main() -> int:
    if not SCORES_PATH.exists():
        print(f"ERROR: committed scores not found: {SCORES_PATH}")
        return 2
    records = _load()

    print("=" * 78)
    print("E-Z cross-judge degeneracy re-analysis  ($0 — committed data, no model calls)")
    print("Statistic is KNOWN: Gwet AC1/AC2 (2008); κ-paradox = Feinstein-Cicchetti (1990).")
    print("Contribution = applied per-axis degeneracy diagnostic, not the coefficient.")
    print("=" * 78)
    print(f"source: {SCORES_PATH} ({len(records)} records)  | num_categories={NUM_CATEGORIES}")

    out: dict[str, Any] = {
        "source": str(SCORES_PATH),
        "n_records": len(records),
        "num_categories": NUM_CATEGORIES,
        "degeneracy_threshold": DEGENERACY_THRESHOLD,
        "statistic_provenance": "Gwet AC1/AC2 (2008); kappa paradox Feinstein-Cicchetti (1990) — known statistic, applied here",
        "axes": {},
    }

    for axis in AXES:
        n_surv, agg = _agreement_for_axis(records, axis)
        print(f"\n[{axis}]  survivors={n_surv}")
        print(f"  dominant-fraction (saturation flag, >{DEGENERACY_THRESHOLD:.0%} = degenerate):")
        for j in JUDGES:
            frac = agg.per_judge_dominant_fraction[j]
            flag = "  ⚠ DEGENERATE" if frac > DEGENERACY_THRESHOLD else ""
            print(f"    {j:<16} {frac:.3f}{flag}")
        print(f"  {'judge pair':<32}{'κ (weighted)':>14}{'AC2':>10}{'p_o':>10}")
        for pair in agg.pairwise_kappa:
            label = f"{pair[0]} ↔ {pair[1]}"
            print(
                f"    {label:<30}{agg.pairwise_kappa[pair]:>14.3f}"
                f"{agg.pairwise_ac1[pair]:>10.3f}{agg.pairwise_p_o[pair]:>10.3f}"
            )
        out["axes"][axis] = {
            "n_survivors": n_surv,
            "per_judge_dominant_fraction": agg.per_judge_dominant_fraction,
            "pairwise_kappa": {f"{a}__{b}": v for (a, b), v in agg.pairwise_kappa.items()},
            "pairwise_ac2": {f"{a}__{b}": v for (a, b), v in agg.pairwise_ac1.items()},
            "pairwise_p_o": {f"{a}__{b}": v for (a, b), v in agg.pairwise_p_o.items()},
            "degeneracy_warnings": agg.degeneracy_warnings,
        }

    print("\nRead:")
    print("  legibility — every judge degenerate; κ collapses but p_o & AC2 stay high")
    print("               → ceiling artifact, not substantive judge disagreement.")
    print("  coverage   — real variance; κ recovers toward AC2, rankings judge-stable → genuine agreement.")

    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWrote {OUTPUT_PATH}  ($0 — no model calls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
