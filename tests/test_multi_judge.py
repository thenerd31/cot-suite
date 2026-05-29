"""run_multi_judge (per-item fan-out) + judge_agreement (cross-item κ)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError

from cotsuite.judges import (
    AgreementResult,
    MultiJudgeResult,
    agreement_from_sample_scores,
    judge_agreement,
    run_multi_judge,
)


@dataclass
class _StubJudge:
    """GraderClient stub: returns a fixed reply, records prompts seen."""

    reply: str
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.reply


# --- run_multi_judge --------------------------------------------------------


async def test_run_multi_judge_fans_and_scores() -> None:
    judges = {"a": _StubJudge("YES"), "b": _StubJudge("NO")}
    result = await run_multi_judge(
        judges,
        "PROMPT",
        item={"id": 1},
        parse_response=lambda c: c.strip() == "YES",
        score_of=lambda v: 1.0 if v else 0.0,
    )
    assert result.per_judge_scores == {"a": 1.0, "b": 0.0}
    assert result.per_judge_raw == {"a": True, "b": False}
    assert result.raw["item"] == {"id": 1}
    assert result.raw["completions"] == {"a": "YES", "b": "NO"}
    # every judge saw the same rendered prompt exactly once.
    assert judges["a"].calls == ["PROMPT"]
    assert judges["b"].calls == ["PROMPT"]


def test_results_are_frozen() -> None:
    mjr = MultiJudgeResult(per_judge_scores={"a": 1.0})
    with pytest.raises(ValidationError):
        mjr.per_judge_scores = {}  # type: ignore[misc]
    ag = AgreementResult()
    with pytest.raises(ValidationError):
        ag.n_items = 5  # type: ignore[misc]


# --- judge_agreement --------------------------------------------------------


def test_judge_agreement_kappa_and_means() -> None:
    # j1 labels [1,0,1], j2 labels [1,0,0], K=2 → κ = 0.4 (hand-computed).
    results = [
        MultiJudgeResult(per_judge_scores={"j1": 1.0, "j2": 1.0}),
        MultiJudgeResult(per_judge_scores={"j1": 0.0, "j2": 0.0}),
        MultiJudgeResult(per_judge_scores={"j1": 1.0, "j2": 0.0}),
    ]
    ag = judge_agreement(results, num_categories=2)
    assert ag.n_items == 3
    assert ag.num_categories == 2
    assert ag.pairwise_kappa[("j1", "j2")] == pytest.approx(0.4)
    assert ag.per_judge_mean_scores["j1"] == pytest.approx(2 / 3)
    assert ag.per_judge_mean_scores["j2"] == pytest.approx(1 / 3)
    assert ag.degeneracy_warnings == []  # both judges at 66.7%, below 70%


def test_degeneracy_warning_fires_above_threshold() -> None:
    # j2 emits category 1 for 9 of 10 items (90% > 70%); j1 is spread across 3.
    results = [
        MultiJudgeResult(per_judge_scores={"j1": float(i % 3), "j2": 1.0 if i < 9 else 0.0})
        for i in range(10)
    ]
    ag = judge_agreement(results, num_categories=3)
    assert any("j2" in w for w in ag.degeneracy_warnings)
    assert not any("'j1'" in w for w in ag.degeneracy_warnings)


def test_judge_agreement_empty() -> None:
    ag = judge_agreement([], num_categories=2)
    assert ag.n_items == 0
    assert ag.pairwise_kappa == {}
    assert ag.per_judge_mean_scores == {}


def test_inconsistent_judge_set_raises() -> None:
    results = [
        MultiJudgeResult(per_judge_scores={"j1": 1.0, "j2": 0.0}),
        MultiJudgeResult(per_judge_scores={"j1": 1.0}),  # missing j2
    ]
    with pytest.raises(ValueError, match="inconsistent judge set"):
        judge_agreement(results, num_categories=2)


def test_three_judges_all_pairs() -> None:
    results = [
        MultiJudgeResult(per_judge_scores={"a": 1.0, "b": 1.0, "c": 0.0}),
        MultiJudgeResult(per_judge_scores={"a": 0.0, "b": 0.0, "c": 1.0}),
    ]
    ag = judge_agreement(results, num_categories=2)
    assert set(ag.pairwise_kappa) == {("a", "b"), ("a", "c"), ("b", "c")}


# --- agreement_from_sample_scores -------------------------------------------


@dataclass
class _FakeScore:
    metadata: dict


@dataclass
class _FakeSampleScore:
    score: _FakeScore


def test_agreement_from_flat_scores_skips_missing() -> None:
    scores = [
        _FakeScore({"multi_judge": {"per_judge_scores": {"a": 1.0, "b": 1.0}}}),
        _FakeScore({"multi_judge": {"per_judge_scores": {"a": 0.0, "b": 0.0}}}),
        _FakeScore({"other": 1}),  # no multi_judge payload → skipped
        _FakeScore({"multi_judge": {"per_judge_scores": {"a": 1.0, "b": 0.0}}}),
    ]
    ag = agreement_from_sample_scores(scores, num_categories=2)
    assert ag.n_items == 3
    assert ("a", "b") in ag.pairwise_kappa


def test_agreement_from_nested_sample_scores() -> None:
    scores = [
        _FakeSampleScore(_FakeScore({"multi_judge": {"per_judge_scores": {"a": 1.0, "b": 1.0}}})),
        _FakeSampleScore(_FakeScore({"multi_judge": {"per_judge_scores": {"a": 0.0, "b": 1.0}}})),
    ]
    ag = agreement_from_sample_scores(scores, num_categories=2)
    assert ag.n_items == 2
