"""Quadratic-weighted Cohen's κ pinned to hand-computed values.

No sklearn dependency: every expected value below is derived from first
principles (κ = 1 - Σw·O / Σw·E with w[i,j] = (i-j)²/(K-1)²) and the derivation
is given inline so the test is self-checking.
"""

from __future__ import annotations

import math

import pytest

from cotsuite.judges.kappa import (
    cohen_kappa_quadratic,
    dominant_category_fraction,
    gwet_ac1,
    observed_agreement,
)


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


# --- Gwet AC1 / observed agreement (prevalence-robust; the κ-paradox tooling) ---
#
# The κ "prevalence paradox" (Feinstein & Cicchetti, 1990) is a KNOWN statistic.
# These tests pin our AC1/p_o implementation to hand-computed values, including an
# explicit paradox case, so correctness on the known paradox is verified.

# Textbook 2×2 paradox: 100 items, confusion O = [[90, 5], [5, 0]].
#   a = 95 zeros + 5 ones; b = 90 zeros, then 5 ones, then 5 zeros.
_PARADOX_A = [0] * 95 + [1] * 5
_PARADOX_B = [0] * 90 + [1] * 5 + [0] * 5


def test_observed_agreement_unweighted_and_perfect() -> None:
    # p_o = (O_00 + O_11)/n = (90 + 0)/100 = 0.90 (diagonal fraction).
    assert observed_agreement(_PARADOX_A, _PARADOX_B, num_categories=2, weighted=False) == pytest.approx(0.90)
    # Identical labels → p_o = 1.0 (weighted or not).
    assert observed_agreement([0, 1, 2], [0, 1, 2], num_categories=3, weighted=True) == pytest.approx(1.0)
    assert observed_agreement([0, 1, 2], [0, 1, 2], num_categories=3, weighted=False) == pytest.approx(1.0)


def test_gwet_ac1_textbook_paradox() -> None:
    # The Feinstein-Cicchetti paradox: HIGH observed agreement (0.90) but κ goes
    # NEGATIVE because the skewed marginals (95/5) inflate κ's chance term — while
    # AC1, using prevalence π=(0.95,0.05), stays high.
    # κ = 1 - Σw·O/Σw·E = 1 - 10/9.5 = -0.05263.
    # p_e(AC1) = Σπ(1-π)/(K-1) = 0.095/1 = 0.095; AC1 = (0.90-0.095)/(1-0.095) = 0.8895.
    kappa = cohen_kappa_quadratic(_PARADOX_A, _PARADOX_B, num_categories=2)
    ac1 = gwet_ac1(_PARADOX_A, _PARADOX_B, num_categories=2, weighted=False)
    p_o = observed_agreement(_PARADOX_A, _PARADOX_B, num_categories=2, weighted=False)
    assert kappa == pytest.approx(-0.05263, abs=1e-4)
    assert p_o == pytest.approx(0.90)
    assert ac1 == pytest.approx(0.8895, abs=1e-4)
    # The paradox, asserted in one line: high agreement, negative κ, high AC1.
    assert kappa < 0.0 < 0.85 < ac1


def test_gwet_ac1_defined_where_kappa_is_nan() -> None:
    # Single-category raters: κ is undefined (nan, Σw·E = 0), but AC1 is DEFINED —
    # π collapses so p_e → 0 and AC1 → p_o = 1.0. This is the saturated-ceiling
    # regime where AC1 is exactly the point.
    assert math.isnan(cohen_kappa_quadratic([1, 1, 1, 1], [1, 1, 1, 1], num_categories=3))
    assert gwet_ac1([1, 1, 1, 1], [1, 1, 1, 1], num_categories=3) == pytest.approx(1.0)


def test_gwet_ac1_perfect_agreement_and_edge_cases() -> None:
    assert gwet_ac1([0, 1, 2, 1], [0, 1, 2, 1], num_categories=3, weighted=True) == pytest.approx(1.0)
    assert gwet_ac1([0, 1, 2, 1], [0, 1, 2, 1], num_categories=3, weighted=False) == pytest.approx(1.0)
    # K < 2 → AC1 undefined → nan.
    assert math.isnan(gwet_ac1([0, 0, 0], [0, 0, 0], num_categories=1))


@pytest.mark.parametrize(
    ("a", "b", "k"),
    [
        ([0, 1], [0, 1, 2], 3),  # length mismatch
        ([], [], 3),  # empty
        ([0, 3], [0, 1], 3),  # out of range
    ],
)
def test_gwet_ac1_invalid_inputs_raise(a: list[int], b: list[int], k: int) -> None:
    with pytest.raises(ValueError):
        gwet_ac1(a, b, num_categories=k)
