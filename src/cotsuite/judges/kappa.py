"""Quadratic-weighted Cohen's κ, implemented from scratch on numpy.

No scikit-learn dependency in production (the project ships only ``numpy`` for
numeric work). The unit tests pin the implementation to hand-computed κ values
on small inputs (see ``tests/test_kappa.py``); sklearn is *not* a runtime or
test dependency.

Quadratic weights: ``w[i, j] = (i - j)**2 / (K - 1)**2`` for ``K`` ordinal
categories ``0 .. K-1``. κ is

    κ = 1 - (Σ w·O) / (Σ w·E)

where ``O`` is the observed co-occurrence matrix and ``E`` the expected matrix
under independence (outer product of the marginals / n). When ``Σ w·E == 0``
(no expected disagreement — e.g. a rater used a single category for every item)
κ is mathematically undefined and we return ``nan`` rather than dividing by
zero. The caller (``judge_agreement``) layers a >70%-dominant-category warning
on top of this so degenerate distributions are surfaced, not silently scored.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np

# A pairwise κ involving a judge whose label distribution is more concentrated
# than this is flagged as degenerate by ``judge_agreement`` — κ is numerically
# defined but statistically unstable when one category dominates.
DEGENERACY_THRESHOLD = 0.70


def cohen_kappa_quadratic(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    num_categories: int,
) -> float:
    """Return the quadratic-weighted Cohen's κ between two raters.

    Args:
        rater_a: integer category labels (``0 .. num_categories-1``) from rater A.
        rater_b: integer category labels from rater B, same length as ``rater_a``.
        num_categories: the number of ordinal categories ``K`` (``>= 1``).

    Returns:
        κ in ``[-1, 1]``, or ``nan`` when κ is undefined (no expected
        disagreement, i.e. ``Σ w·E == 0`` — typically a single-category rater
        or ``K == 1``).

    Raises:
        ValueError: on length mismatch, empty input, or out-of-range labels.
    """
    a = np.asarray(rater_a, dtype=int)
    b = np.asarray(rater_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"rater length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("cannot compute κ on empty input")
    if num_categories < 1:
        raise ValueError(f"num_categories must be >= 1, got {num_categories}")
    if a.min() < 0 or a.max() >= num_categories or b.min() < 0 or b.max() >= num_categories:
        raise ValueError(
            f"labels must be in [0, {num_categories}); got a∈[{a.min()},{a.max()}] b∈[{b.min()},{b.max()}]"
        )

    n = a.size
    k = num_categories
    observed = np.zeros((k, k), dtype=float)
    np.add.at(observed, (a, b), 1.0)
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / n

    idx = np.arange(k)
    if k > 1:
        weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    else:
        weights = np.zeros((k, k), dtype=float)

    denom = float((weights * expected).sum())
    if denom == 0.0:
        return float("nan")
    numer = float((weights * observed).sum())
    return 1.0 - numer / denom


def dominant_category_fraction(labels: Sequence[int]) -> float:
    """Return the fraction of items occupied by the single most common label.

    Used as the degeneracy signal: a value above :data:`DEGENERACY_THRESHOLD`
    means one category dominates and any κ involving this rater is unstable.
    Returns ``0.0`` on empty input.
    """
    items = list(labels)
    if not items:
        return 0.0
    counts = Counter(items)
    return max(counts.values()) / len(items)
