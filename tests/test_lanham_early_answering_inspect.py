"""Unit tests for the Lanham early-answering Inspect task (Phase 3 POC).

$0 — no live calls: the solver's ``get_model()`` is monkeypatched to a fake model
whose completions are scripted (the native Lanham tests use a ``ScriptedClient``
the same way). The re-elicitation loop (1 full-answer + 5 prefix-fraction calls)
runs against the fake, so AOC and the model call-count are deterministic.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

import cotsuite.inspect.solvers.lanham_early_answering as lanham_solver_mod
from cotsuite.inspect.scorers import cot_lanham_early_answering_aoc
from cotsuite.inspect.solvers import cot_lanham_early_answering

_COT = "Step one considers the options. Step two narrows them down. Therefore the answer is reached."
_QUESTION = "Question: pick one. (A) x (B) y"


class ContentReasoning:
    """Stand-in matched by ``_extract_reasoning`` via ``type(block).__name__``."""

    def __init__(self, reasoning: str) -> None:
        self.reasoning = reasoning


class _FakeModel:
    """Inspect-model stand-in: scripted ``generate`` (clamps to last) + call count."""

    def __init__(self, completions: list[str]) -> None:
        self._completions = completions
        self.calls = 0

    async def generate(self, prompt, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        idx = min(self.calls, len(self._completions) - 1)
        self.calls += 1
        return SimpleNamespace(completion=self._completions[idx])


def _patch_model(monkeypatch: pytest.MonkeyPatch, fake: _FakeModel) -> None:
    monkeypatch.setattr(lanham_solver_mod, "get_model", lambda *a, **k: fake)


def _make_state(cot: str) -> SimpleNamespace:
    msg = SimpleNamespace(content=[ContentReasoning(cot)])
    return SimpleNamespace(messages=[msg], input_text=_QUESTION, metadata={})


async def _noop_generate(state):  # noqa: ANN001
    return state


@pytest.mark.asyncio
async def test_constant_answer_aoc_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # Model always answers A → every prefix retains the full answer → AOC = 0.
    fake = _FakeModel(["Answer: A"])
    _patch_model(monkeypatch, fake)
    state = await cot_lanham_early_answering()(_make_state(_COT), _noop_generate)
    payload = state.metadata["lanham_early_answering"]
    assert payload["aoc"] == pytest.approx(0.0)
    # 1 full-answer elicitation + 5 prefix fractions = 6 re-elicitations.
    assert fake.calls == 6


@pytest.mark.asyncio
async def test_total_flip_aoc_one(monkeypatch: pytest.MonkeyPatch) -> None:
    # Full answer A, but every prefix answers B → no retention anywhere → AOC = 1.
    fake = _FakeModel(["Answer: A"] + ["Answer: B"] * 5)
    _patch_model(monkeypatch, fake)
    state = await cot_lanham_early_answering()(_make_state(_COT), _noop_generate)
    assert state.metadata["lanham_early_answering"]["aoc"] == pytest.approx(1.0)
    assert fake.calls == 6


@pytest.mark.asyncio
async def test_empty_cot_skips_without_calling_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel(["Answer: A"])
    _patch_model(monkeypatch, fake)
    state = await cot_lanham_early_answering()(_make_state("   "), _noop_generate)
    payload = state.metadata["lanham_early_answering"]
    assert payload["aoc"] is None
    assert payload["skip_reason"] == "empty_reasoning"
    assert fake.calls == 0  # short-circuits before re-eliciting


@pytest.mark.asyncio
async def test_scorer_surfaces_aoc(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel(["Answer: A"] + ["Answer: B"] * 5)
    _patch_model(monkeypatch, fake)
    state = await cot_lanham_early_answering()(_make_state(_COT), _noop_generate)
    score = await cot_lanham_early_answering_aoc()(state, None)
    assert score.value == pytest.approx(1.0)
    assert score.metadata["per_fraction"] is not None


@pytest.mark.asyncio
async def test_scorer_nan_when_solver_not_run() -> None:
    # No solver output in metadata → scalar NaN sentinel (excluded from the metric).
    state = SimpleNamespace(metadata={})
    score = await cot_lanham_early_answering_aoc()(state, None)
    assert isinstance(score.value, float) and math.isnan(score.value)
    assert score.metadata["skip_reason"] == "no_aoc_in_metadata"


@pytest.mark.asyncio
async def test_scorer_nan_on_empty_cot(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeModel(["Answer: A"])
    _patch_model(monkeypatch, fake)
    state = await cot_lanham_early_answering()(_make_state(""), _noop_generate)
    score = await cot_lanham_early_answering_aoc()(state, None)
    assert isinstance(score.value, float) and math.isnan(score.value)
    assert score.metadata["skip_reason"] == "empty_reasoning"


def test_solver_registers_with_inspect() -> None:
    from inspect_ai._util.registry import registry_info

    assert registry_info(cot_lanham_early_answering()).name == "cot_lanham_early_answering"


def test_scorer_registers_with_inspect() -> None:
    from inspect_ai._util.registry import registry_info

    assert registry_info(cot_lanham_early_answering_aoc()).name == "cot_lanham_early_answering_aoc"
