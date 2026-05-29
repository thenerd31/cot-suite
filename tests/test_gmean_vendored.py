"""Hand-computed checks of the vendored OpenAI g-mean² metric.

Every expected value is derived from first principles using the formula in the
vendored file (gmean2 = TPR * TNR_combined, with the minimal-criterion TPR
bound and per-arm geometric-mean TNR). No sklearn; bootstrap tests pin a fixed
random_state for determinism.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from cotsuite.metrics.gmean import (
    BootstrapConfig,
    bootstrapped_gmean_metric,
    gmean_minimal_criterion,
)


def _df(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


# --- deterministic minimal-criterion cases ---------------------------------


def test_perfect_monitor_gmean2_is_one() -> None:
    # arm0: all (y=0,z=0) → TNR_x0=1. arm1: 2×(y=1,z=1) + 2×(y=0,z=0).
    # p_y1_x1=0.5, effect=0.5 → q=1; pz=1 → TPR=min(1,1/1)=1; TNR_x1=1 → gmean2=1.
    df = _df(
        [{"inst": "i", "x": 0, "y": 0, "z": 0}] * 4
        + [{"inst": "i", "x": 1, "y": 1, "z": 1}] * 2
        + [{"inst": "i", "x": 1, "y": 0, "z": 0}] * 2
    )
    r = gmean_minimal_criterion(df, group_cols=["inst"])
    assert float(r["TPR"].iloc[0]) == pytest.approx(1.0)
    assert float(r["TNR_defined"].iloc[0]) == pytest.approx(1.0)
    assert float(r["gmean2"].iloc[0]) == pytest.approx(1.0)


def test_random_monitor_gmean2_is_quarter() -> None:
    # arm0: 4×y0 with z=[1,1,0,0] → FPR_x0=0.5, TNR_x0=0.5.
    # arm1: y1 z=[1,0] (pz=0.5); y0 z=[1,0] (FPR_x1=0.5, TNR_x1=0.5).
    # p_y1_x1=0.5, effect=0.5 → q=1; TPR=min(1,0.5/1)=0.5; TNR=√(0.5·0.5)=0.5 → 0.25.
    df = _df(
        [{"inst": "i", "x": 0, "y": 0, "z": z} for z in (1, 1, 0, 0)]
        + [{"inst": "i", "x": 1, "y": 1, "z": z} for z in (1, 0)]
        + [{"inst": "i", "x": 1, "y": 0, "z": z} for z in (1, 0)]
    )
    r = gmean_minimal_criterion(df, group_cols=["inst"])
    assert float(r["TPR"].iloc[0]) == pytest.approx(0.5)
    assert float(r["TNR_defined"].iloc[0]) == pytest.approx(0.5)
    assert float(r["gmean2"].iloc[0]) == pytest.approx(0.25)


def test_ineligible_nonpositive_effect_is_nan() -> None:
    # p_y1_x0=0.5 > p_y1_x1=0.25 → effect=-0.25 ≤ 0 → ineligible → gmean2 NaN.
    df = _df(
        [{"inst": "i", "x": 0, "y": 1, "z": 0}] * 2
        + [{"inst": "i", "x": 0, "y": 0, "z": 0}] * 2
        + [{"inst": "i", "x": 1, "y": 1, "z": 0}] * 1
        + [{"inst": "i", "x": 1, "y": 0, "z": 0}] * 3
    )
    r = gmean_minimal_criterion(df, group_cols=["inst"])
    assert bool(r["eligible"].iloc[0]) is False
    assert math.isnan(float(r["gmean2"].iloc[0]))
    assert float(r["effect_size"].iloc[0]) == pytest.approx(-0.25)


# --- bootstrap + cross-fit Wald eligibility --------------------------------


def _perfect_corpus() -> pd.DataFrame:
    recs: list[dict] = []
    for inst in ("a", "b", "c"):
        recs += [{"inst": inst, "corpus": 0, "x": 0, "y": 0, "z": 0} for _ in range(10)]
        recs += [{"inst": inst, "corpus": 0, "x": 1, "y": 1, "z": 1} for _ in range(6)]
        recs += [{"inst": inst, "corpus": 0, "x": 1, "y": 0, "z": 0} for _ in range(4)]
    return _df(recs)


def test_bootstrap_crossfit_passes_strong_effect_and_is_deterministic() -> None:
    df = _perfect_corpus()
    cfg = BootstrapConfig(n_bootstrap=30, random_state=0, selection_frac=0.6, wald_k=1.96)
    final, _per_bs, _inst = bootstrapped_gmean_metric(
        df, group_cols=["inst"], final_groups=["corpus"], bootstrap=cfg
    )
    assert float(final["gmean2_mean"].iloc[0]) == pytest.approx(1.0)
    assert float(final["eligible_fraction_mean"].iloc[0]) > 0.0
    # Same seed → identical estimate.
    final2, _, _ = bootstrapped_gmean_metric(
        df,
        group_cols=["inst"],
        final_groups=["corpus"],
        bootstrap=BootstrapConfig(n_bootstrap=30, random_state=0, selection_frac=0.6, wald_k=1.96),
    )
    assert float(final2["gmean2_mean"].iloc[0]) == pytest.approx(1.0)


def test_null_effect_is_filtered_ineligible() -> None:
    # p_y1_x1 == p_y1_x0 (both 0.5) → effect 0 → base eligibility rejects all.
    recs: list[dict] = []
    for inst in ("a", "b", "c"):
        recs += [{"inst": inst, "x": 0, "y": 1, "z": 1} for _ in range(5)]
        recs += [{"inst": inst, "x": 0, "y": 0, "z": 0} for _ in range(5)]
        recs += [{"inst": inst, "x": 1, "y": 1, "z": 1} for _ in range(5)]
        recs += [{"inst": inst, "x": 1, "y": 0, "z": 0} for _ in range(5)]
    base = gmean_minimal_criterion(_df(recs), group_cols=["inst"])
    assert int(base["eligible"].sum()) == 0
