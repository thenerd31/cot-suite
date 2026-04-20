"""Unit tests for Lanham mistake-injection (mocked clients)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cotdiv.tests.lanham.mistake_injection import _select_indices, mistake_injection


def test_select_indices_no_cap_returns_all() -> None:
    assert _select_indices(5, None) == [0, 1, 2, 3, 4]


def test_select_indices_cap_smaller_than_n() -> None:
    idx = _select_indices(20, 4)
    assert len(idx) == 4
    assert idx[0] == 0
    assert idx[-1] < 20


def test_select_indices_cap_larger_than_n() -> None:
    assert _select_indices(3, 10) == [0, 1, 2]


@dataclass
class ScriptedClient:
    """Returns canned answers in order each time .complete is called."""

    script: list[str]
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.script:
            return ""
        return self.script.pop(0)


@pytest.mark.asyncio
async def test_post_hoc_cot_has_zero_aoc() -> None:
    # Model always says "A" no matter what CoT we give it — CoT has no effect.
    # Baseline (full CoT): A. For each of 2 sentences, mistake-injected CoT
    # re-elicitation also returns A. AOC = 0.
    mistake_gen = ScriptedClient(script=["WRONG_A.", "WRONG_B."])
    # Per-sentence flow when resample_tail=True and i<n-1:
    #   continuation call (A), elicitation call (The answer is A)
    # Last sentence: elicitation only.
    under_test = ScriptedClient(
        script=[
            "The answer is (A)",  # baseline full-CoT elicitation
            "So the answer is A",  # continuation for i=0
            "The answer is (A)",  # elicitation for i=0
            "The answer is (A)",  # elicitation for i=1 (last sentence, no continuation)
        ],
    )
    result = await mistake_injection(
        model=under_test,
        mistake_generator=mistake_gen,
        question="Pick A.",
        cot="Sentence A. Sentence B.",
        length_weighted=False,
    )
    assert result.aoc == pytest.approx(0.0, abs=0.01)
    assert all(v == 1.0 for v in result.per_fraction.values())


@pytest.mark.asyncio
async def test_faithful_cot_has_positive_aoc() -> None:
    # Model responds to the injected mistake by switching from A to B.
    mistake_gen = ScriptedClient(script=["BAD_0.", "BAD_1."])
    under_test = ScriptedClient(
        script=[
            "The answer is (A)",  # baseline full-CoT elicitation
            "continuation 0",  # continuation for i=0
            "The answer is (B)",  # elicitation for i=0 — flipped!
            "The answer is (B)",  # elicitation for i=1 — flipped!
        ],
    )
    result = await mistake_injection(
        model=under_test,
        mistake_generator=mistake_gen,
        question="?",
        cot="Alpha reasoning. Beta reasoning.",
        length_weighted=False,
    )
    assert result.aoc == pytest.approx(1.0, abs=0.01)
    assert all(v == 0.0 for v in result.per_fraction.values())


@pytest.mark.asyncio
async def test_empty_cot_returns_none_aoc() -> None:
    result = await mistake_injection(
        model=ScriptedClient(script=[]),
        mistake_generator=ScriptedClient(script=[]),
        question="?",
        cot="",
    )
    assert result.aoc is None


@pytest.mark.asyncio
async def test_mistake_generator_none_raises_value_error() -> None:
    with pytest.raises(ValueError, match="separate model"):
        await mistake_injection(
            model="anthropic/claude-opus-4-5",
            mistake_generator=None,  # type: ignore[arg-type]
            question="?",
            cot="One.",
        )


@pytest.mark.asyncio
async def test_mistake_generator_same_string_as_model_raises_value_error() -> None:
    with pytest.raises(ValueError, match="differ from"):
        await mistake_injection(
            model="anthropic/claude-opus-4-5",
            mistake_generator="anthropic/claude-opus-4-5",
            question="?",
            cot="One.",
        )


@pytest.mark.asyncio
async def test_resample_tail_false_skips_continuation_calls() -> None:
    mistake_gen = ScriptedClient(script=["BAD."])
    under_test = ScriptedClient(
        script=[
            "The answer is (A)",  # baseline elicitation
            "The answer is (A)",  # elicitation for i=0 (no continuation)
        ],
    )
    result = await mistake_injection(
        model=under_test,
        mistake_generator=mistake_gen,
        question="?",
        cot="Only one sentence.",
        resample_tail=False,
        length_weighted=False,
    )
    assert result.aoc is not None
    # Under-test client should have been called exactly twice: baseline +
    # one elicitation for the single sentence. No continuation.
    assert len(under_test.calls) == 2
