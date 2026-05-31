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


# ---------------------------------------------------------------------------
# Prevalence-robust agreement (Gwet's AC1/AC2) + raw observed agreement.
#
# The *statistic* here is not novel: Gwet's AC1 (Gwet, 2008) is a standard
# chance-corrected agreement coefficient designed to be robust to the base-rate
# / prevalence problem in Cohen's κ — the "kappa paradox" (Feinstein & Cicchetti,
# 1990) where high observed agreement coincides with low or negative κ because κ's
# chance-correction term is inflated under skewed marginals. Crucially, AC1 is
# *defined* even when one category dominates (where ``cohen_kappa_quadratic``
# returns ``nan``), which is exactly the saturated-score regime of frontier-model
# CoT-monitorability metrics. We add it so a degeneracy/ceiling artifact can be
# distinguished from substantive judge disagreement; the applied diagnostic is the
# contribution, not the coefficient.
# ---------------------------------------------------------------------------


def _confusion(
    rater_a: Sequence[int], rater_b: Sequence[int], num_categories: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validated K×K observed matrix + row/col marginals + n. Same checks as κ."""
    a = np.asarray(rater_a, dtype=int)
    b = np.asarray(rater_b, dtype=int)
    if a.shape != b.shape:
        raise ValueError(f"rater length mismatch: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("cannot compute agreement on empty input")
    if num_categories < 1:
        raise ValueError(f"num_categories must be >= 1, got {num_categories}")
    if a.min() < 0 or a.max() >= num_categories or b.min() < 0 or b.max() >= num_categories:
        raise ValueError(
            f"labels must be in [0, {num_categories}); got a∈[{a.min()},{a.max()}] b∈[{b.min()},{b.max()}]"
        )
    observed = np.zeros((num_categories, num_categories), dtype=float)
    np.add.at(observed, (a, b), 1.0)
    return observed, observed.sum(axis=1), observed.sum(axis=0), a.size


def _agreement_weights(k: int) -> np.ndarray:
    """Quadratic *agreement* weights v[i,j] = 1 - (i-j)²/(K-1)² (1 on diagonal).

    The complement of the disagreement weights ``cohen_kappa_quadratic`` uses, so a
    weighted coefficient here pairs apples-to-apples with the weighted κ.
    """
    idx = np.arange(k)
    if k > 1:
        return 1.0 - (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    return np.ones((k, k), dtype=float)


def observed_agreement(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    num_categories: int,
    weighted: bool = True,
) -> float:
    """Raw observed agreement ``p_o`` — the quantity κ never exposes.

    Args:
        rater_a: integer category labels (``0 .. num_categories-1``) from rater A.
        rater_b: integer category labels from rater B, same length as ``rater_a``.
        num_categories: the number of categories ``K``.
        weighted: when ``True`` (default), quadratic-weighted observed agreement
            ``Σ v·O / n`` (``v`` = agreement weights) — comparable to the weighted κ.
            When ``False``, the unweighted diagonal fraction ``trace(O) / n``.
    """
    observed, _, _, n = _confusion(rater_a, rater_b, num_categories)
    if weighted:
        return float((_agreement_weights(num_categories) * observed).sum() / n)
    return float(np.trace(observed) / n)


def gwet_ac1(
    rater_a: Sequence[int],
    rater_b: Sequence[int],
    *,
    num_categories: int,
    weighted: bool = True,
) -> float:
    """Gwet's AC1 (``weighted=False``) / AC2 (``weighted=True``, quadratic weights).

    Prevalence-robust chance-corrected agreement (Gwet, 2008). ``p_e`` uses the
    averaged marginal prevalence ``π_k = (row_k + col_k)/(2n)``:

        p_e = (T_v / (K(K-1))) · Σ_k π_k (1 - π_k),   AC = (p_a - p_e) / (1 - p_e)

    where ``p_a`` is the (optionally weighted) observed agreement and
    ``T_v = Σ v`` (``= K`` unweighted). Unlike κ, this is **defined when a rater is
    single-category** (``Σ π(1-π) → 0`` ⇒ ``p_e → 0`` ⇒ ``AC → p_a``), so it does
    not collapse on saturated/ceiling distributions. Returns ``nan`` only for
    ``K < 2`` or the (unreachable for ``K ≥ 2``) ``p_e == 1`` case.
    """
    if num_categories < 2:
        return float("nan")
    observed, row, col, n = _confusion(rater_a, rater_b, num_categories)
    v = _agreement_weights(num_categories) if weighted else np.eye(num_categories)
    p_a = float((v * observed).sum() / n)
    pi = (row + col) / (2.0 * n)
    t_v = float(v.sum())
    p_e = t_v / (num_categories * (num_categories - 1)) * float((pi * (1.0 - pi)).sum())
    if p_e == 1.0:
        return float("nan")
    return (p_a - p_e) / (1.0 - p_e)
