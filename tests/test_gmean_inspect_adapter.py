"""Inspect adapter for the g-mean² metric — mock-based, $0, no network."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import pytest

import cotsuite.inspect.metrics.gmean as adapter_mod
from cotsuite.inspect.metrics.gmean import _metadata_of, gmean2_metric, inspect_gmean_metric
from cotsuite.metrics.gmean import BootstrapConfig


@dataclass
class _FakeScore:
    metadata: dict


def _scores(records: list[tuple]) -> list[_FakeScore]:
    return [_FakeScore({"task": t, "x": x, "y": y, "z": z}) for (t, x, y, z) in records]


def test_metadata_of_reads_direct_and_nested() -> None:
    assert _metadata_of(_FakeScore({"x": 1})) == {"x": 1}

    @dataclass
    class _Inner:
        metadata: dict

    @dataclass
    class _SampleScore:
        score: object
        metadata: object = None  # SampleScore-like: metadata lives on .score

    assert _metadata_of(_SampleScore(score=_Inner({"x": 2}))) == {"x": 2}


def test_adapter_assembles_dataframe_and_calls_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_boot(df, *, group_cols, final_groups, bootstrap, **_):  # type: ignore[no-untyped-def]
        captured["df"] = df.copy()
        captured["group_cols"] = group_cols
        captured["final_groups"] = final_groups
        out = pd.DataFrame([{final_groups[0]: 0, "gmean2_mean": 0.42, "gmean2_std": 0.1}])
        return out, None, None

    monkeypatch.setattr(adapter_mod, "bootstrapped_gmean_metric", fake_boot)
    # instance t1: arm0 (y0), arm1 (y1,z1) → eligible (effect 1>0) → n_eligible=1.
    out = inspect_gmean_metric(_scores([("t1", 0, 0, 0), ("t1", 1, 1, 1)]), group_cols=["task"])

    df = captured["df"]
    assert {"task", "x", "y", "z"}.issubset(df.columns)
    assert captured["group_cols"] == ["task"]
    assert out["gmean2_mean"] == pytest.approx(0.42)
    assert out["gmean2_std"] == pytest.approx(0.1)
    assert out["n_eligible"] == 1


def test_adapter_matches_direct_call_end_to_end() -> None:
    records: list[tuple] = []
    for t in ("t1", "t2"):
        records += [(t, 0, 0, 0)] * 10 + [(t, 1, 1, 1)] * 6 + [(t, 1, 0, 0)] * 4
    cfg = BootstrapConfig(n_bootstrap=20, random_state=0)
    out = inspect_gmean_metric(_scores(records), group_cols=["task"], bootstrap=cfg)
    assert out["gmean2_mean"] == pytest.approx(1.0)
    assert out["n_eligible"] == 2


def test_gmean2_metric_factory_is_callable_and_matches_adapter() -> None:
    # The @metric-registered factory returns a callable that produces the same
    # dict as the plain adapter on identical inputs.
    records: list[tuple] = []
    for t in ("t1", "t2"):
        records += [(t, 0, 0, 0)] * 10 + [(t, 1, 1, 1)] * 6 + [(t, 1, 0, 0)] * 4
    cfg = BootstrapConfig(n_bootstrap=10, random_state=0)
    scores = _scores(records)

    metric_fn = gmean2_metric(group_cols=["task"], bootstrap=cfg)
    assert callable(metric_fn)
    assert metric_fn(scores) == inspect_gmean_metric(scores, group_cols=["task"], bootstrap=cfg)


def test_adapter_missing_key_raises() -> None:
    bad = [_FakeScore({"task": "t", "x": 0, "y": 0})]  # missing z
    with pytest.raises(ValueError, match="z"):
        inspect_gmean_metric(bad, group_cols=["task"])


def test_adapter_missing_group_col_raises() -> None:
    bad = [_FakeScore({"x": 0, "y": 0, "z": 0})]  # missing "task"
    with pytest.raises(ValueError, match="task"):
        inspect_gmean_metric(bad, group_cols=["task"])


def test_adapter_empty_returns_nan() -> None:
    out = inspect_gmean_metric([], group_cols=["task"])
    assert math.isnan(out["gmean2_mean"])
    assert out["n_eligible"] == 0
