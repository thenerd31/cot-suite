"""Multi-judge agreement primitives.

Reusable, non-Inspect-native substrate for cross-judge faithfulness work
(Emmons-Zimmermann cross-judge validation, the v0.1.1 cross-judge ablations,
and the Turpin/Chen cue-verbalization consumers). Deliberately *not* placed
under ``cotsuite.inspect`` because ``_cue_judge`` and the reproduction scripts
that consume these are not Inspect-native.

Public surface:

* :func:`run_multi_judge` / :class:`MultiJudgeResult` — per-item fan-out.
* :func:`judge_agreement` / :class:`AgreementResult` — cross-item κ + warnings.
* :func:`cohen_kappa_quadratic`, :func:`dominant_category_fraction` — the κ core.
* :func:`detect_ranking_reversals`, :func:`ranking_reversal_summary`,
  :class:`RankingReversal` — classifier-sensitivity detection.
* :class:`InspectGraderAdapter`, :func:`agreement_from_sample_scores` — Inspect glue.
"""

from __future__ import annotations

from cotsuite.judges.kappa import (
    DEGENERACY_THRESHOLD,
    cohen_kappa_quadratic,
    dominant_category_fraction,
    gwet_ac1,
    observed_agreement,
)
from cotsuite.judges.multi_judge import (
    AgreementResult,
    InspectGraderAdapter,
    MultiJudgeResult,
    agreement_from_sample_scores,
    judge_agreement,
    run_multi_judge,
)
from cotsuite.judges.ranking_reversal import (
    RankingReversal,
    detect_ranking_reversals,
    ranking_reversal_summary,
)

__all__ = [
    "DEGENERACY_THRESHOLD",
    "AgreementResult",
    "InspectGraderAdapter",
    "MultiJudgeResult",
    "RankingReversal",
    "agreement_from_sample_scores",
    "cohen_kappa_quadratic",
    "detect_ranking_reversals",
    "dominant_category_fraction",
    "gwet_ac1",
    "observed_agreement",
    "judge_agreement",
    "ranking_reversal_summary",
    "run_multi_judge",
]
