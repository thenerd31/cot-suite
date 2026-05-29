"""Ranking-reversal detection (Young 2603.20172 classifier-sensitivity)."""

from __future__ import annotations

from cotsuite.judges.ranking_reversal import (
    detect_ranking_reversals,
    ranking_reversal_summary,
)


def test_detects_a_strict_reversal() -> None:
    # judge_a: m1 > m2 ; judge_b: m1 < m2  → one reversal on (m1, m2).
    scores = {
        "judge_a": {"m1": 4.0, "m2": 2.0},
        "judge_b": {"m1": 1.0, "m2": 3.0},
    }
    reversals = detect_ranking_reversals(scores)
    assert len(reversals) == 1
    r = reversals[0]
    assert {r.subject_x, r.subject_y} == {"m1", "m2"}
    assert r.judge_a == "judge_a" and r.judge_b == "judge_b"


def test_no_reversal_when_judges_agree_on_order() -> None:
    scores = {
        "judge_a": {"m1": 4.0, "m2": 2.0, "m3": 1.0},
        "judge_b": {"m1": 9.0, "m2": 5.0, "m3": 3.0},  # same order, different scale
    }
    assert detect_ranking_reversals(scores) == []


def test_ties_are_not_reversals() -> None:
    scores = {
        "judge_a": {"m1": 2.0, "m2": 2.0},  # tie
        "judge_b": {"m1": 1.0, "m2": 5.0},
    }
    assert detect_ranking_reversals(scores) == []


def test_only_common_subjects_compared() -> None:
    scores = {
        "judge_a": {"m1": 4.0, "m2": 2.0, "only_a": 9.0},
        "judge_b": {"m1": 1.0, "m2": 3.0, "only_b": 0.0},
    }
    reversals = detect_ranking_reversals(scores)
    # only (m1, m2) is comparable; only_a / only_b are dropped.
    assert len(reversals) == 1
    assert {reversals[0].subject_x, reversals[0].subject_y} == {"m1", "m2"}


def test_summary_reports_rate() -> None:
    scores = {
        "judge_a": {"m1": 4.0, "m2": 2.0},
        "judge_b": {"m1": 1.0, "m2": 3.0},
    }
    summary = ranking_reversal_summary(scores)
    assert len(summary) == 1
    assert "1 of 1" in summary[0]
    assert "100%" in summary[0]


def test_summary_empty_when_consistent() -> None:
    scores = {
        "judge_a": {"m1": 4.0, "m2": 2.0},
        "judge_b": {"m1": 8.0, "m2": 2.0},
    }
    assert ranking_reversal_summary(scores) == []
