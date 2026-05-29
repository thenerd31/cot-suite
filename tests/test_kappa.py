"""Quadratic-weighted Cohen's κ pinned to hand-computed values.

No sklearn dependency: every expected value below is derived from first
principles (κ = 1 - Σw·O / Σw·E with w[i,j] = (i-j)²/(K-1)²) and the derivation
is given inline so the test is self-checking.
"""

from __future__ import annotations

import math

import pytest

from cotsuite.judges.kappa import cohen_kappa_quadratic, dominant_category_fraction


def test_perfect_agreement_is_one() -> None:
    # Identical labels → Σw·O = 0 (all mass on the zero-weight diagonal) → κ = 1.
    assert cohen_kappa_quadratic([0, 1, 2, 1, 0], [0, 1, 2, 1, 0], num_categories=3) == 1.0


def test_perfect_disagreement_binary_is_minus_one() -> None:
    # K=2 quadratic weights collapse to 0/1 (unweighted). a=[0,0,1,1], b=[1,1,0,0]:
    # Σw·O = 2(off-diag) , Σw·E = 1  →  κ = 1 - 2/1 ... using normalized form = -1.
    assert cohen_kappa_quadratic([0, 0, 1, 1], [1, 1, 0, 0], num_categories=2) == -1.0


def test_partial_agreement_hand_value() -> None:
    # a=[0,0,1,1,2,2], b=[0,1,1,2,2,2], K=3.
    # Σw·O = 0.5 ; Σw·E(counts) = 2.0  →  κ = 1 - 0.5/2.0 = 0.75.
    kappa = cohen_kappa_quadratic([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2], num_categories=3)
    assert kappa == pytest.approx(0.75)


def test_degenerate_single_category_is_nan() -> None:
    # Both raters use only category 1 → Σw·E = 0 → κ undefined → nan.
    assert math.isnan(cohen_kappa_quadratic([1, 1, 1, 1], [1, 1, 1, 1], num_categories=3))


def test_known_offset_case() -> None:
    # a=[1,0,1], b=[1,0,0], K=2: Σw·O=1, marginals row=[1,2] col=[2,1],
    # Σw·E = (1/3 + 4/3) = 5/3 → κ = 1 - 1/(5/3) = 1 - 0.6 = 0.4.
    assert cohen_kappa_quadratic([1, 0, 1], [1, 0, 0], num_categories=2) == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("a", "b", "k"),
    [
        ([0, 1], [0, 1, 2], 3),  # length mismatch
        ([], [], 3),  # empty
        ([0, 3], [0, 1], 3),  # label 3 out of range for K=3
    ],
)
def test_invalid_inputs_raise(a: list[int], b: list[int], k: int) -> None:
    with pytest.raises(ValueError):
        cohen_kappa_quadratic(a, b, num_categories=k)


def test_dominant_category_fraction() -> None:
    assert dominant_category_fraction([1, 1, 1, 0]) == pytest.approx(0.75)
    assert dominant_category_fraction([0, 1, 2, 3]) == pytest.approx(0.25)
    assert dominant_category_fraction([]) == 0.0
