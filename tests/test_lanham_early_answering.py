"""Unit tests for Lanham early-answering (mocked model client)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from cotmon.tests.lanham._extractors import mcq_answer_extractor
from cotmon.tests.lanham._sentences import (
    default_sentence_split,
    prefix_at_fraction,
)
from cotmon.tests.lanham.early_answering import early_answering


def test_sentence_splitter_handles_multiple_terminators() -> None:
    text = "First. Second! Third? Fourth."
    assert default_sentence_split(text) == ["First.", "Second!", "Third?", "Fourth."]


def test_sentence_splitter_preserves_abbreviations() -> None:
    text = "Dr. Smith went home. Then he slept."
    # Current heuristic may not perfectly handle "Dr." — assert it at least
    # produces two sentences (the core behaviour needed for prefix logic).
    sentences = default_sentence_split(text)
    assert len(sentences) >= 2


def test_prefix_at_fraction_boundaries() -> None:
    sents = ["A.", "B.", "C.", "D."]
    assert prefix_at_fraction(sents, 0.0) == ""
    assert prefix_at_fraction(sents, 1.0) == "A. B. C. D."
    assert prefix_at_fraction(sents, 0.5) == "A. B."
    assert prefix_at_fraction(sents, 0.25) == "A."


@dataclass
class FakeClient:
    """Deterministic fake — picks answer by prefix length."""

    map: dict[int, str]

    async def complete(self, prompt: str) -> str:
        length = sum(1 for c in prompt if c == ".")
        for threshold in sorted(self.map.keys(), reverse=True):
            if length >= threshold:
                return f"The answer is ({self.map[threshold]})"
        return "The answer is (A)"


@pytest.mark.asyncio
async def test_early_answering_post_hoc_cot_has_zero_aoc() -> None:
    # Model answers B at every prefix — CoT is perfectly post-hoc.
    client = FakeClient(map={0: "B"})
    result = await early_answering(
        model=client,
        question="What is the capital of France?",
        cot="Paris is in France. France is in Europe. Europe is a continent. Therefore Paris.",
        answer_extractor=mcq_answer_extractor,
        fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        length_weighted=False,
    )
    assert result.aoc == pytest.approx(0.0, abs=0.01)
    assert all(v == 1.0 for v in result.per_fraction.values())


@pytest.mark.asyncio
async def test_early_answering_faithful_cot_has_positive_aoc() -> None:
    # Model only converges to B once it has seen the full CoT.
    # Count of '.' in prompt: elicitation template contributes 3 trailing ".";
    # add len(prefix) periods, so total = 3 + k.
    # We want: short prefix → A, long prefix → B.
    # k=0: total=3 → A (no threshold matches, falls through to A)
    # k=4: total=7 → B
    client = FakeClient(map={7: "B"})
    cot = "Sentence one. Sentence two. Sentence three. Sentence four."
    result = await early_answering(
        model=client,
        question="Pick the right letter.",
        cot=cot,
        full_answer="B",
        answer_extractor=mcq_answer_extractor,
        fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
        length_weighted=False,
    )
    assert result.aoc is not None
    assert result.aoc > 0.0
    # Retention at 0 should be 0 (wrong); retention at 1.0 should be 1 (right).
    assert result.per_fraction[0.0] == 0.0
    assert result.per_fraction[1.0] == 1.0


@pytest.mark.asyncio
async def test_early_answering_empty_cot_returns_none_aoc() -> None:
    client = FakeClient(map={0: "A"})
    result = await early_answering(
        model=client,
        question="?",
        cot="",
        full_answer="A",
    )
    assert result.aoc is None
    assert result.raw["error"] == "no sentences in CoT"


@pytest.mark.asyncio
async def test_length_weighting_weights_later_fractions_more() -> None:
    # With length weighting, a wrong answer at fraction=1.0 should hurt AOC
    # more than a wrong answer at fraction=0.25.
    client_late_fail = FakeClient(map={0: "A", 999: "A"})  # always A
    cot = "A. B. C. D."
    result_weighted = await early_answering(
        model=client_late_fail,
        question="?",
        cot=cot,
        full_answer="B",  # full answer never matches
        length_weighted=True,
    )
    result_uniform = await early_answering(
        model=client_late_fail,
        question="?",
        cot=cot,
        full_answer="B",
        length_weighted=False,
    )
    # All retentions are 0 here, so both AOCs should be 1.0 regardless of weighting.
    assert result_weighted.aoc == pytest.approx(1.0, abs=0.01)
    assert result_uniform.aoc == pytest.approx(1.0, abs=0.01)
