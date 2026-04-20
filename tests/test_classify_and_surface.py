"""Tests for the classification dispatcher and reasoning-surface guards."""

from __future__ import annotations

import pytest

from cotdiv.core.classify import classify
from cotdiv.core.schemas import TestResult
from cotdiv.core.trajectory import Reasoning, Trajectory, Turn
from cotdiv.metrics.verbosity import reasoning_surface_health, verbosity


def test_classify_computational_when_both_aocs_above_threshold() -> None:
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.35),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.42),
    }
    assert classify(tests) == "computational"


def test_classify_mixed_when_only_one_above_threshold() -> None:
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.30),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.05),
    }
    assert classify(tests) == "mixed"


def test_classify_rationalization_when_both_near_zero_and_cc_shap_low() -> None:
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.01),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.02),
        "metrics.cc_shap": TestResult(name="metrics.cc_shap", aoc=0.03),
    }
    assert classify(tests) == "rationalization"


def test_classify_unknown_when_summarized_reasoning() -> None:
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.4),
    }
    assert classify(tests, has_summarized_reasoning=True) == "unknown"


def test_classify_unknown_when_no_tests_run() -> None:
    assert classify({}) == "unknown"


def test_classify_mixed_when_cc_shap_missing_and_aocs_low() -> None:
    # Both low but no CC-SHAP available → defaults to rationalization per
    # memo rule (cc_near_zero is True when cc is None).
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.05),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.04),
    }
    assert classify(tests) == "rationalization"


@pytest.mark.asyncio
async def test_verbosity_counts_reasoning_chars() -> None:
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                reasoning=[
                    Reasoning(text="hello"),
                    Reasoning(text="world"),
                ],
            ),
        ],
    )
    result = await verbosity(traj)
    assert result.value == 10.0


def test_surface_health_no_reasoning_for_empty_cot() -> None:
    traj = Trajectory(turns=[Turn(role="assistant", text="42")])
    assert reasoning_surface_health(traj) == "no_reasoning"


def test_surface_health_no_reasoning_when_too_short() -> None:
    traj = Trajectory(
        turns=[Turn(role="assistant", reasoning=[Reasoning(text="ok")])],
    )
    assert reasoning_surface_health(traj) == "no_reasoning"


def test_surface_health_summarized_only_on_claude_4x_summaries() -> None:
    long_summary = "Summarized reasoning. " * 10
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                reasoning=[Reasoning(text=long_summary, is_summary=True)],
            ),
        ],
    )
    assert reasoning_surface_health(traj) == "summarized_only"


def test_surface_health_illegible_flags_non_ascii_heavy_cot() -> None:
    # 80% CJK / 20% ASCII — should trip the illegibility threshold.
    non_ascii = "日本語で推論している文章。" * 10
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                reasoning=[Reasoning(text=non_ascii, is_summary=False)],
            ),
        ],
    )
    assert reasoning_surface_health(traj) == "illegible"


def test_surface_health_healthy_on_normal_cot() -> None:
    cot = (
        "Let me think step by step. First I consider the options. "
        "Then I evaluate each one. The answer is B."
    )
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                reasoning=[Reasoning(text=cot, is_summary=False)],
            ),
        ],
    )
    assert reasoning_surface_health(traj) == "healthy"
