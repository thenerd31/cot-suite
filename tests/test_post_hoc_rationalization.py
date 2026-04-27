"""Unit tests for Arcuschin 2503.08679 post-hoc rationalization detector.

All three tests use a ScriptedClient mock — zero outbound calls. The
detector's behavior is validated structurally; quality of Haiku's
actual judgments is measured by the live-run report, not by these tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from cotsuite.core.classify import classify
from cotsuite.core.schemas import TestResult
from cotsuite.core.trajectory import Reasoning, Trajectory, Turn
from cotsuite.tests.post_hoc_rationalization import (
    PostHocRationalizationPrompt,
    post_hoc_rationalization,
)


@dataclass
class ScriptedClient:
    script: list[str]
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.script.pop(0) if self.script else ""


def _traj(*, question: str, reasoning: str, final_text: str, final_answer: str) -> Trajectory:
    return Trajectory(
        turns=[
            Turn(role="user", text=question),
            Turn(
                role="assistant",
                text=final_text,
                reasoning=[Reasoning(text=reasoning, provider="qwen", is_summary=False)],
            ),
        ],
        final_answer=final_answer,
    )


@pytest.mark.asyncio
async def test_detector_fires_on_gpqa_001_style_mid_cot_flip() -> None:
    """The gpqa_diamond_001 motivating case: CoT considered C then settled
    on D with low confidence; final answer D. If the judge sees structural
    flipping in the CoT with no clear acknowledgment, it should fire.

    We simulate the judge's response directly via ScriptedClient; this tests
    the detector's plumbing, not Haiku's actual judgment quality.
    """
    traj = _traj(
        question="Chemistry: how many carbons after reactions 1-3?",
        reasoning=(
            "First reaction adds 2 carbons, so 10. Then reaction 2 adds 2 more, "
            "making 12. So I think the answer is C, 12. "
            "Actually, wait, maybe reaction 2 doesn't add carbons. "
            "I'll go with D. However, I'm not fully confident."
        ),
        final_text="Answer: D",
        final_answer="D",
    )
    fake_judge = ScriptedClient(
        script=[
            json.dumps(
                {
                    "cot_conclusion": "C",
                    "diverged": True,
                    "acknowledged": False,
                    "confidence": 0.72,
                    "judge_reasoning": "CoT initially argues for C (12 carbons) and flips "
                    "to D without acknowledging the reasoning error that motivated the flip.",
                },
            ),
        ],
    )
    result = await post_hoc_rationalization(traj, judge=fake_judge)
    assert result.diverged is True
    assert result.acknowledged is False
    assert result.cot_conclusion == "C"
    assert result.final_answer == "D"
    assert 0.0 <= result.confidence <= 1.0
    assert result.judge_reasoning  # non-empty
    assert result.autorater_raw_response  # raw body preserved


@pytest.mark.asyncio
async def test_detector_does_not_fire_on_aligned_trajectory() -> None:
    traj = _traj(
        question="What is 2+2?",
        reasoning="Adding two and two gives four. So the answer is 4.",
        final_text="Answer: 4",
        final_answer="4",
    )
    fake_judge = ScriptedClient(
        script=[
            json.dumps(
                {
                    "cot_conclusion": "4",
                    "diverged": False,
                    "acknowledged": False,
                    "confidence": 0.95,
                    "judge_reasoning": "Reasoning concludes 4; final answer 4. Aligned.",
                },
            ),
        ],
    )
    result = await post_hoc_rationalization(traj, judge=fake_judge)
    assert result.diverged is False
    assert result.cot_conclusion == "4"


@pytest.mark.asyncio
async def test_detector_returns_valid_result_object_shape() -> None:
    """Smoke: detector must return a fully-populated result with every
    documented field, regardless of the judge's specific scores."""
    traj = _traj(
        question="test",
        reasoning="some reasoning",
        final_text="the final answer",
        final_answer="X",
    )
    fake_judge = ScriptedClient(
        script=[
            json.dumps(
                {
                    "cot_conclusion": "X",
                    "diverged": False,
                    "acknowledged": False,
                    "confidence": 0.5,
                    "judge_reasoning": "ok",
                },
            ),
        ],
    )
    result = await post_hoc_rationalization(traj, judge=fake_judge)
    assert isinstance(result.cot_conclusion, str)
    assert isinstance(result.final_answer, str)
    assert isinstance(result.diverged, bool)
    assert isinstance(result.acknowledged, bool)
    assert isinstance(result.confidence, float)
    assert isinstance(result.judge_reasoning, str)
    assert isinstance(result.autorater_raw_response, str)


@pytest.mark.asyncio
async def test_detector_raises_on_missing_reasoning() -> None:
    traj = Trajectory(
        turns=[Turn(role="user", text="?"), Turn(role="assistant", text="X")],
        final_answer="X",
    )
    with pytest.raises(ValueError, match="non-empty reasoning"):
        await post_hoc_rationalization(traj, judge=ScriptedClient(script=[]))


def test_prompt_has_placeholders_and_integrity() -> None:
    prompt = PostHocRationalizationPrompt.load()
    for placeholder in ("{question}", "{reasoning}", "{final_output}", "{final_answer}"):
        assert placeholder in prompt.template
    assert len(prompt.sha256) == 64


def test_classify_post_hoc_rationalization_short_circuits_lanham() -> None:
    """When the Arcuschin detector fires, that label overrides whatever
    Lanham would have produced — even computational-looking AOCs."""
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.4),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.45),
        "arcuschin.post_hoc_rationalization": TestResult(
            name="arcuschin.post_hoc_rationalization",
            aoc=1.0,  # fired
        ),
    }
    assert classify(tests) == "post_hoc_rationalization"


def test_classify_ignores_arcuschin_when_not_fired() -> None:
    tests = {
        "lanham.early_answering": TestResult(name="lanham.early_answering", aoc=0.35),
        "lanham.mistake_injection": TestResult(name="lanham.mistake_injection", aoc=0.42),
        "arcuschin.post_hoc_rationalization": TestResult(
            name="arcuschin.post_hoc_rationalization",
            aoc=0.0,  # did not fire
        ),
    }
    assert classify(tests) == "computational"
