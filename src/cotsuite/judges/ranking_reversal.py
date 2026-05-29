"""Ranking-reversal detection between judges (classifier-sensitivity).

Young 2603.20172 documents that swapping the LLM judge can *reverse* the
relative ordering of the subjects being evaluated (models, methods, items) —
not merely shift absolute scores. This module detects those reversals.

Two granularities are provided:

* :func:`detect_ranking_reversals` — the detailed, subject-named detector.
  Given ``{judge: {subject: score}}`` it returns every subject-pair on which
  two judges disagree about the order (a strict sign flip of the score
  difference). This is the form used when subjects are named (models in a
  scaling table, biases in a catalog, …).
* :func:`ranking_reversal_summary` — one human-readable string per judge-pair,
  reporting the reversal rate. Used by ``judge_agreement`` to populate
  ``AgreementResult.ranking_reversal_warnings`` without emitting an unbounded
  ``O(subjects²)`` list.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations

from pydantic import BaseModel, ConfigDict


class RankingReversal(BaseModel):
    """One subject-pair on which two judges disagree about the ordering.

    ``judge_a`` ranks ``subject_x`` strictly above (or below) ``subject_y``
    while ``judge_b`` ranks them in the opposite order. The four raw scores are
    retained so callers can report the magnitude of the disagreement.
    """

    model_config = ConfigDict(frozen=True)

    judge_a: str
    judge_b: str
    subject_x: str
    subject_y: str
    a_x: float
    a_y: float
    b_x: float
    b_y: float


def detect_ranking_reversals(
    per_judge_subject_scores: Mapping[str, Mapping[str, float]],
) -> list[RankingReversal]:
    """Return every strict pairwise ranking reversal between judges.

    Args:
        per_judge_subject_scores: ``{judge_name: {subject_name: score}}``. Only
            subjects scored by *both* judges of a pair are compared.

    Returns:
        A list of :class:`RankingReversal`, one per (judge-pair, subject-pair)
        on which the two judges' score differences have opposite (nonzero)
        sign. Ties (zero difference) are not reversals.
    """
    judges = list(per_judge_subject_scores)
    reversals: list[RankingReversal] = []
    for judge_a, judge_b in combinations(judges, 2):
        scores_a = per_judge_subject_scores[judge_a]
        scores_b = per_judge_subject_scores[judge_b]
        common = sorted(set(scores_a) & set(scores_b))
        for subj_x, subj_y in combinations(common, 2):
            diff_a = scores_a[subj_x] - scores_a[subj_y]
            diff_b = scores_b[subj_x] - scores_b[subj_y]
            if diff_a != 0 and diff_b != 0 and (diff_a > 0) != (diff_b > 0):
                reversals.append(
                    RankingReversal(
                        judge_a=judge_a,
                        judge_b=judge_b,
                        subject_x=subj_x,
                        subject_y=subj_y,
                        a_x=scores_a[subj_x],
                        a_y=scores_a[subj_y],
                        b_x=scores_b[subj_x],
                        b_y=scores_b[subj_y],
                    )
                )
    return reversals


def ranking_reversal_summary(
    per_judge_subject_scores: Mapping[str, Mapping[str, float]],
) -> list[str]:
    """Summarize ranking reversals as one warning string per disagreeing pair.

    For each judge-pair that has at least one reversal, returns a string of the
    form ``"judges 'A' vs 'B': 3 of 10 comparable subject-pairs rank-reversed
    (30%)"``. Judge-pairs with no reversals produce no string.
    """
    judges = list(per_judge_subject_scores)
    out: list[str] = []
    for judge_a, judge_b in combinations(judges, 2):
        scores_a = per_judge_subject_scores[judge_a]
        scores_b = per_judge_subject_scores[judge_b]
        common = sorted(set(scores_a) & set(scores_b))
        total = 0
        reversed_count = 0
        for subj_x, subj_y in combinations(common, 2):
            diff_a = scores_a[subj_x] - scores_a[subj_y]
            diff_b = scores_b[subj_x] - scores_b[subj_y]
            if diff_a != 0 and diff_b != 0:
                total += 1
                if (diff_a > 0) != (diff_b > 0):
                    reversed_count += 1
        if reversed_count:
            rate = reversed_count / total
            out.append(
                f"judges {judge_a!r} vs {judge_b!r}: {reversed_count} of {total} "
                f"comparable subject-pairs rank-reversed ({rate:.0%})"
            )
    return out
