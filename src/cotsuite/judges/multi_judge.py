"""Multi-judge wrapper and cross-item agreement aggregation.

Two granularities:

* :func:`run_multi_judge` is **per-item**. It fans one prompt across N judges
  concurrently and returns a :class:`MultiJudgeResult` carrying each judge's
  parsed output and numeric score for *that item*. It does NOT compute κ or
  ranking reversals — those are population statistics, undefined on a single
  item.
* :func:`judge_agreement` is the **cross-item aggregator**. Given the list of
  per-item :class:`MultiJudgeResult`, it computes pairwise quadratic-weighted
  Cohen's κ, per-judge mean scores, >70%-dominant-category degeneracy
  warnings, and ranking-reversal warnings, returning an
  :class:`AgreementResult`.

The scorers emit the **per-item** ``MultiJudgeResult`` payload into
``Score.metadata["multi_judge"]`` and the **cross-item** ``AgreementResult`` is
obtained by calling :func:`agreement_from_sample_scores` on the eval log's
scores (or :func:`judge_agreement` directly). We deliberately do NOT bolt a
bespoke ``@metric`` onto the existing scorers: Inspect's ``metrics=`` is fixed
at decoration time, the single-judge path must stay byte-identical for
backward-compat, and post-hoc aggregation over a log's scores is the idiomatic
way to compute a cross-sample statistic. A caller who wants κ in the Inspect
log can wrap :func:`judge_agreement` in a one-line custom ``@metric``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any, TypeVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from cotsuite.judges.kappa import (
    DEGENERACY_THRESHOLD,
    cohen_kappa_quadratic,
    dominant_category_fraction,
    gwet_ac1,
    observed_agreement,
)
from cotsuite.judges.ranking_reversal import ranking_reversal_summary
from cotsuite.models.clients import GraderClient

T = TypeVar("T")


class MultiJudgeResult(BaseModel):
    """Per-item result of running one prompt through N judges.

    Carries no agreement statistics — κ and ranking reversals are cross-item
    quantities computed by :func:`judge_agreement`.
    """

    model_config = ConfigDict(frozen=True)

    per_judge_scores: dict[str, float]
    per_judge_raw: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict)


class AgreementResult(BaseModel):
    """Cross-item agreement statistics over a population of judged items."""

    model_config = ConfigDict(frozen=True)

    pairwise_kappa: dict[tuple[str, str], float] = Field(default_factory=dict)
    ranking_reversal_warnings: list[str] = Field(default_factory=list)
    per_judge_mean_scores: dict[str, float] = Field(default_factory=dict)
    degeneracy_warnings: list[str] = Field(default_factory=list)
    n_items: int = 0
    num_categories: int | None = None
    # Prevalence-robust additions (additive, backward-compatible). AC1/AC2 and raw
    # observed agreement let a κ-degeneracy/ceiling artifact be told apart from
    # substantive disagreement; per_judge_dominant_fraction is the saturation flag
    # (compare against DEGENERACY_THRESHOLD = 0.70).
    pairwise_ac1: dict[tuple[str, str], float] = Field(default_factory=dict)
    pairwise_p_o: dict[tuple[str, str], float] = Field(default_factory=dict)
    per_judge_dominant_fraction: dict[str, float] = Field(default_factory=dict)


async def run_multi_judge(
    judges: Mapping[str, GraderClient],
    prompt: str,
    item: Any,
    parse_response: Callable[[str], T],
    score_of: Callable[[T], float],
) -> MultiJudgeResult:
    """Run ``prompt`` through every judge concurrently; score each output.

    Args:
        judges: ``{judge_name: GraderClient}``. Names label the κ pairs later.
        prompt: the fully-rendered judge prompt (identical for every judge).
        item: the item under judgement, retained verbatim in ``result.raw`` for
            traceability (not passed to ``parse_response``; capture it in a
            closure if your parser needs it).
        parse_response: ``(completion: str) -> T`` — parse one judge's raw text.
        score_of: ``(T) -> float`` — map a parsed output to its numeric score.

    Returns:
        A :class:`MultiJudgeResult` with ``per_judge_scores`` (name → float),
        ``per_judge_raw`` (name → parsed ``T``), and ``raw`` (the item plus the
        raw completion strings).
    """
    names = list(judges)
    completions = await asyncio.gather(*(judges[name].complete(prompt) for name in names))
    per_judge_scores: dict[str, float] = {}
    per_judge_raw: dict[str, Any] = {}
    raw_completions: dict[str, str] = {}
    for name, completion in zip(names, completions, strict=True):
        parsed = parse_response(completion)
        per_judge_raw[name] = parsed
        per_judge_scores[name] = float(score_of(parsed))
        raw_completions[name] = completion
    return MultiJudgeResult(
        per_judge_scores=per_judge_scores,
        per_judge_raw=per_judge_raw,
        raw={"item": item, "completions": raw_completions},
    )


def _to_labels(values: Iterable[float], num_categories: int) -> list[int]:
    """Round/clip float scores to integer ordinal categories ``0..K-1`` for κ.

    Cohen's κ is categorical; judge scores that are means over runs are
    real-valued, so we round to the nearest category and clip into range. This
    is a deliberate, documented lossy step — callers wanting integer-exact κ
    should pass integer per-item scores (e.g. a single run per judge).
    """
    arr = np.rint(np.asarray(list(values), dtype=float)).astype(int)
    arr = np.clip(arr, 0, num_categories - 1)
    return arr.tolist()


def judge_agreement(
    results: Iterable[MultiJudgeResult],
    num_categories: int,
) -> AgreementResult:
    """Aggregate per-item results into cross-item agreement statistics.

    Computes pairwise quadratic-weighted Cohen's κ between every judge pair,
    per-judge mean scores, degeneracy warnings (a judge whose label
    distribution exceeds the >70% dominant-category threshold), and
    ranking-reversal warnings (judges that order items differently).

    Args:
        results: the per-item :class:`MultiJudgeResult` objects. The judge set
            of the first result defines the expected judges; a result missing
            any of those judges raises ``ValueError``.
        num_categories: ordinal-category count ``K`` for κ (e.g. 2 for a
            binary verbalize/PHR signal, 5 for a 0-4 Likert autorater).

    Returns:
        An :class:`AgreementResult`. Empty input yields an empty result with
        ``n_items == 0``.
    """
    results = list(results)
    if not results:
        return AgreementResult(n_items=0, num_categories=num_categories)

    judges = list(results[0].per_judge_scores)
    columns: dict[str, list[float]] = {judge: [] for judge in judges}
    for result in results:
        for judge in judges:
            if judge not in result.per_judge_scores:
                raise ValueError(f"inconsistent judge set: {judge!r} missing from a result")
            columns[judge].append(result.per_judge_scores[judge])

    per_judge_mean = {judge: sum(col) / len(col) for judge, col in columns.items()}
    labels = {judge: _to_labels(col, num_categories) for judge, col in columns.items()}

    degeneracy_warnings: list[str] = []
    per_judge_dominant_fraction: dict[str, float] = {}
    for judge in judges:
        fraction = dominant_category_fraction(labels[judge])
        per_judge_dominant_fraction[judge] = fraction
        if fraction > DEGENERACY_THRESHOLD:
            degeneracy_warnings.append(
                f"judge {judge!r}: one category covers {fraction:.0%} of "
                f"{len(labels[judge])} items (> {DEGENERACY_THRESHOLD:.0%}); "
                f"pairwise κ involving it is unstable"
            )

    pairwise_kappa: dict[tuple[str, str], float] = {}
    pairwise_ac1: dict[tuple[str, str], float] = {}
    pairwise_p_o: dict[tuple[str, str], float] = {}
    for judge_a, judge_b in combinations(judges, 2):
        la, lb = labels[judge_a], labels[judge_b]
        pairwise_kappa[(judge_a, judge_b)] = cohen_kappa_quadratic(
            la, lb, num_categories=num_categories
        )
        # AC1/AC2 + raw observed agreement on the same weighting as the κ above,
        # so a low κ with high AC1 + high p_o reads as a degeneracy artifact.
        pairwise_ac1[(judge_a, judge_b)] = gwet_ac1(la, lb, num_categories=num_categories)
        pairwise_p_o[(judge_a, judge_b)] = observed_agreement(
            la, lb, num_categories=num_categories
        )

    subject_scores = {
        judge: {str(i): columns[judge][i] for i in range(len(results))} for judge in judges
    }
    ranking_reversal_warnings = ranking_reversal_summary(subject_scores)

    return AgreementResult(
        pairwise_kappa=pairwise_kappa,
        ranking_reversal_warnings=ranking_reversal_warnings,
        per_judge_mean_scores=per_judge_mean,
        degeneracy_warnings=degeneracy_warnings,
        n_items=len(results),
        num_categories=num_categories,
        pairwise_ac1=pairwise_ac1,
        pairwise_p_o=pairwise_p_o,
        per_judge_dominant_fraction=per_judge_dominant_fraction,
    )


@dataclass
class InspectGraderAdapter:
    """Expose an Inspect ``Model`` (``.generate``) as a :class:`GraderClient`.

    The multi-judge wrapper speaks the ``GraderClient`` protocol (``.complete``);
    Inspect grader models speak ``.generate(...).completion``. This thin adapter
    bridges the two so the Inspect scorers can reuse :func:`run_multi_judge`.
    """

    model: Any

    async def complete(self, prompt: str) -> str:
        """Generate against the wrapped Inspect model, returning the text."""
        out = await self.model.generate(prompt)
        return out.completion


def agreement_from_sample_scores(
    scores: Iterable[Any],
    num_categories: int,
    *,
    metadata_key: str = "multi_judge",
) -> AgreementResult:
    """Rebuild an :class:`AgreementResult` from Inspect scores' metadata.

    Each score is expected to carry ``metadata[metadata_key]["per_judge_scores"]``
    (as the list-mode scorers emit). Accepts either ``Score`` objects (with a
    ``.metadata`` attribute) or Inspect ``SampleScore`` objects (with a nested
    ``.score.metadata``). Scores lacking the multi-judge payload are skipped.
    """
    rebuilt: list[MultiJudgeResult] = []
    for entry in scores:
        score_obj = getattr(entry, "score", entry)
        metadata = getattr(score_obj, "metadata", None) or {}
        payload = metadata.get(metadata_key)
        if payload and "per_judge_scores" in payload:
            rebuilt.append(
                MultiJudgeResult(
                    per_judge_scores={k: float(v) for k, v in payload["per_judge_scores"].items()}
                )
            )
    return judge_agreement(rebuilt, num_categories)
