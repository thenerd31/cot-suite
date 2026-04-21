"""Unit tests for Lanham paraphrasing + filler-tokens (mocked clients)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cotmon.tests.lanham.filler_tokens import filler_tokens
from cotmon.tests.lanham.paraphrasing import paraphrasing


@dataclass
class ScriptedClient:
    script: list[str]
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.script.pop(0) if self.script else ""


@pytest.mark.asyncio
async def test_paraphrasing_zero_gap_when_paraphrase_preserves_answer() -> None:
    # Original and paraphrased prefixes both elicit "A" at every fraction.
    # Gap = 0 → no steganographic signal.
    para = ScriptedClient(script=["para-1", "para-2", "para-3", "para-4"])
    model = ScriptedClient(
        script=[
            "The answer is (A)",  # baseline full-CoT elicitation
            # For each of 4 fractions: original elicitation, paraphrased elicitation
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
            "The answer is (A)",
        ],
    )
    result = await paraphrasing(
        model=model,
        paraphraser=para,
        question="?",
        cot="One. Two. Three. Four.",
        fractions=(0.25, 0.5, 0.75, 1.0),
    )
    assert result.aoc is None
    assert result.synthesis["cotdiv_paraphrasing_gap_v1"] == pytest.approx(0.0, abs=0.01)
    # raw_curve is paper-equivalent; per_fraction mirrors it for schema parity.
    assert result.raw_curve == result.per_fraction


@pytest.mark.asyncio
async def test_paraphrasing_nonzero_gap_when_paraphrase_changes_answer() -> None:
    # Paraphrased prefixes elicit a different answer than original → gap > 0.
    para = ScriptedClient(script=["p1", "p2"])
    model = ScriptedClient(
        script=[
            "The answer is (A)",  # baseline
            "The answer is (A)",
            "The answer is (B)",  # f=0.5: orig A, para B
            "The answer is (A)",
            "The answer is (B)",  # f=1.0: orig A, para B
        ],
    )
    result = await paraphrasing(
        model=model,
        paraphraser=para,
        question="?",
        cot="First. Second.",
        fractions=(0.5, 1.0),
    )
    assert result.aoc is None
    assert result.synthesis["cotdiv_paraphrasing_gap_v1"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.asyncio
async def test_paraphrasing_empty_cot_returns_none() -> None:
    result = await paraphrasing(
        model=ScriptedClient(script=[]),
        paraphraser=ScriptedClient(script=[]),
        question="?",
        cot="",
    )
    assert result.aoc is None


@pytest.mark.asyncio
async def test_paraphraser_none_raises_value_error() -> None:
    with pytest.raises(ValueError, match="separate paraphraser"):
        await paraphrasing(
            model="anthropic/claude-opus-4-5",
            paraphraser=None,  # type: ignore[arg-type]
            question="?",
            cot="One.",
        )


@pytest.mark.asyncio
async def test_paraphraser_same_string_as_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="differ from"):
        await paraphrasing(
            model="anthropic/claude-opus-4-5",
            paraphraser="anthropic/claude-opus-4-5",
            question="?",
            cot="One.",
        )


@pytest.mark.asyncio
async def test_filler_tokens_no_uplift_has_zero_peak() -> None:
    # Model gets every filler-token length wrong → peak rate = 0.
    model = ScriptedClient(script=["(C)"] * 7)
    result = await filler_tokens(
        model=model,
        question="?",
        full_answer="A",
        lengths=(0, 5, 10, 20, 40, 80, 160),
    )
    assert result.aoc is None
    assert result.synthesis["cotdiv_filler_peak_v1"] == 0.0
    assert all(v == 0.0 for v in result.raw_curve.values())


@pytest.mark.asyncio
async def test_filler_tokens_positive_uplift_flagged() -> None:
    # Pathological: model starts returning correct answer at high filler
    # lengths — this would indicate raw test-time compute is doing the work.
    model = ScriptedClient(
        script=["(C)", "(C)", "(A)", "(A)"],
    )
    result = await filler_tokens(
        model=model,
        question="?",
        full_answer="A",
        lengths=(0, 5, 10, 20),
    )
    assert result.synthesis["cotdiv_filler_peak_v1"] == 1.0
    assert result.raw_curve[10.0] == 1.0
