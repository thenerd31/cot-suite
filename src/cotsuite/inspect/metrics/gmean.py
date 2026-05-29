"""Inspect AI @metric adapter for the vendored OpenAI g-mean² metric.

Bridges Inspect's per-sample Scores to the pandas-based
``cotsuite.metrics.gmean.bootstrapped_gmean_metric``. Each Score must carry
metadata with integer 0/1 keys ``x`` (arm: control vs intervention), ``y``
(outcome), ``z`` (monitor prediction), plus the grouping columns (default
``("task",)``) identifying a paired-arm instance.

Two entry points:

* :func:`inspect_gmean_metric` — plain adapter
  ``(scores, *, group_cols, bootstrap) -> dict[str, float]``; directly testable.
* :func:`gmean2_metric` — the same wrapped as an Inspect ``@metric`` factory,
  attachable to a scorer / Task and serializable by Inspect's registry.

``n_eligible`` is the count of instances passing the *base* minimal criterion
(``effect_size > 0`` and ``P(Y=1|X=1) > 0``), computed deterministically via
``gmean_minimal_criterion`` — distinct from the stricter cross-fit Wald
eligibility applied inside the bootstrap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from inspect_ai.scorer import metric

from cotsuite.metrics.gmean import (
    BootstrapConfig,
    bootstrapped_gmean_metric,
    gmean_minimal_criterion,
)

if TYPE_CHECKING:
    from inspect_ai.scorer import Metric, SampleScore, Value

# Constant final-aggregation column: collapse all instances into one corpus-level
# g-mean² number. group_cols identify instances; final_groups aggregate them.
_CORPUS_COL = "__corpus__"


def _metadata_of(score: Any) -> dict[str, Any]:
    """Return a score's metadata, accepting Score or SampleScore-like objects."""
    meta = getattr(score, "metadata", None)
    if meta is None:
        inner = getattr(score, "score", None)
        meta = getattr(inner, "metadata", None)
    return meta or {}


def inspect_gmean_metric(
    scores: Sequence[Any],
    *,
    group_cols: Sequence[str] = ("task",),
    bootstrap: BootstrapConfig | None = None,
) -> dict[str, float]:
    """Assemble Inspect scores into the g-mean² DataFrame and compute the metric.

    Args:
        scores: Inspect ``Score`` / ``SampleScore`` objects. Each must carry
            metadata with integer 0/1 ``x``, ``y``, ``z`` plus every
            ``group_cols`` key.
        group_cols: instance-identifying columns (default ``("task",)``).
        bootstrap: bootstrap/cross-fit settings; defaults to
            ``BootstrapConfig()``.

    Returns:
        ``{"gmean2_mean", "gmean2_std", "n_eligible"}``. Empty input yields
        ``{"gmean2_mean": nan, "gmean2_std": nan, "n_eligible": 0}``.

    Raises:
        ValueError: if any score's metadata is missing ``x`` / ``y`` / ``z`` or
            a grouping column.
    """
    cfg = bootstrap or BootstrapConfig()
    cols = list(group_cols)
    rows: list[dict[str, Any]] = []
    for s in scores:
        md = _metadata_of(s)
        for c in cols:
            if c not in md:
                raise ValueError(f"score metadata missing group column {c!r}; have {sorted(md)}")
        for key in ("x", "y", "z"):
            if key not in md:
                raise ValueError(f"score metadata missing required key {key!r}; have {sorted(md)}")
        row: dict[str, Any] = {c: md[c] for c in cols}
        row["x"] = int(md["x"])
        row["y"] = int(md["y"])
        row["z"] = int(md["z"])
        rows.append(row)

    if not rows:
        return {"gmean2_mean": float("nan"), "gmean2_std": float("nan"), "n_eligible": 0}

    df = pd.DataFrame(rows)
    df[_CORPUS_COL] = 0

    base = gmean_minimal_criterion(df, group_cols=cols)
    n_eligible = int(base["eligible"].sum())

    final, _, _ = bootstrapped_gmean_metric(
        df, group_cols=cols, final_groups=[_CORPUS_COL], bootstrap=cfg
    )
    return {
        "gmean2_mean": float(final["gmean2_mean"].iloc[0]),
        "gmean2_std": float(final["gmean2_std"].iloc[0]),
        "n_eligible": n_eligible,
    }


@metric
def gmean2_metric(
    group_cols: Sequence[str] = ("task",),
    bootstrap: BootstrapConfig | None = None,
) -> Metric:
    """Inspect ``@metric`` factory wrapping :func:`inspect_gmean_metric`.

    Attach to a scorer / Task; Inspect calls the returned metric with the run's
    SampleScores and records ``gmean2_mean`` / ``gmean2_std`` / ``n_eligible``.
    """

    def compute(scores: list[SampleScore]) -> Value:
        """Compute the g-mean² summary over a run's SampleScores."""
        return inspect_gmean_metric(scores, group_cols=group_cols, bootstrap=bootstrap)

    return compute
