"""Regression guard: validate_b4_arcuschin._process_one must persist raw
GPT output even when the downstream judge call raises.

Motivation: on 2026-04-23 the Haiku judge hit an Anthropic credit-exhaustion
error on all 38 correct-answer trajectories. The detector-failure branch
of ``_process_one`` assembled its return dict from scratch instead of
extending the persisted base row, so 38 GPT-4o-mini bodies were dropped
on the floor and had to be re-spent to recover the run. Invariant under
test: once GPT returns a body, ``raw_body`` and ``reasoning`` appear in
every non-``gpt_call`` phase output — judge verdict is optional.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_validator():
    """Import scripts/validate_b4_arcuschin.py as a module.

    It lives outside any package, so we load it by path. Cached per
    session so repeat calls don't re-parse.
    """
    cached = sys.modules.get("_validate_b4_arcuschin")
    if cached is not None:
        return cached
    path = Path(__file__).resolve().parent.parent / "scripts" / "validate_b4_arcuschin.py"
    spec = importlib.util.spec_from_file_location("_validate_b4_arcuschin", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_validate_b4_arcuschin"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- Mocks ------------------------------------------------------------


@dataclass
class _FakeChatMessage:
    content: str


@dataclass
class _FakeChoice:
    message: _FakeChatMessage


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeChatCompletions:
    def __init__(self, body: str):
        self._body = body

    async def create(self, **_: object) -> _FakeResponse:
        return _FakeResponse(choices=[_FakeChoice(message=_FakeChatMessage(content=self._body))])


class _FakeClient:
    def __init__(self, body: str):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions(body))


def _sample() -> dict:
    return {
        "question_id": "gpqa_diamond_042",
        "question": "What is the spin-orbit correction to hydrogen's 2p level?",
        "options": {"A": "alpha1", "B": "alpha2", "C": "alpha3", "D": "alpha4"},
        "correct_answer": "B",
    }


# ---- Tests ------------------------------------------------------------


def test_persistence_invariant_when_judge_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression. Judge raises; raw_body + reasoning must survive."""
    mod = _load_validator()

    gpt_body = "Step 1: recall fine-structure.\nStep 2: compute.\nFinal Answer: B"

    async def _boom(_traj):
        raise RuntimeError("credit balance too low")

    monkeypatch.setattr(mod, "post_hoc_rationalization", _boom)

    sem = asyncio.Semaphore(1)
    client = _FakeClient(gpt_body)
    row = asyncio.run(mod._process_one(_sample(), sem, client))

    assert row["phase"] == "detector"
    assert "error" in row
    assert "credit balance too low" in row["error"]
    # The actual regression guard — these two fields were dropped in the
    # 2026-04-23 incident:
    assert row["raw_body"] == gpt_body
    assert "Step 1" in row["reasoning"]
    assert row["reasoning_len"] > 0
    # Answer-extraction still works on the failure path:
    assert row["final_answer"] == "B"
    assert row["is_correct"] is True


def test_persistence_on_filter_wrong_answer() -> None:
    """Wrong-answer rows also must carry raw_body, so a later re-run can
    inspect WHY the model went wrong without hitting OpenAI again."""
    mod = _load_validator()
    body = "Reasoning...\nFinal Answer: A"
    sem = asyncio.Semaphore(1)
    client = _FakeClient(body)
    row = asyncio.run(mod._process_one(_sample(), sem, client))
    assert row["phase"] == "filter"
    assert row["is_correct"] is False
    assert row["raw_body"] == body
    assert row["reasoning_len"] > 0


def test_persistence_on_complete(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judge-success path also carries raw_body (not just judge fields)."""
    mod = _load_validator()

    async def _ok(_traj):
        return SimpleNamespace(
            cot_conclusion="B",
            diverged=False,
            acknowledged=False,
            confidence=0.9,
            judge_reasoning="aligned",
        )

    monkeypatch.setattr(mod, "post_hoc_rationalization", _ok)

    body = "Good reasoning.\nFinal Answer: B"
    sem = asyncio.Semaphore(1)
    client = _FakeClient(body)
    row = asyncio.run(mod._process_one(_sample(), sem, client))
    assert row["phase"] == "complete"
    assert row["raw_body"] == body
    assert row["diverged"] is False
    assert row["cot_conclusion"] == "B"


def test_gpt_call_failure_omits_body() -> None:
    """If GPT itself raises, there IS no body to persist — this path
    does NOT need raw_body. The invariant is 'once body exists, persist
    it,' not 'every row has a body.'"""
    mod = _load_validator()

    class _ExplodingCompletions:
        async def create(self, **_: object) -> None:
            raise RuntimeError("OpenAI 500")

    bad_client = SimpleNamespace(chat=SimpleNamespace(completions=_ExplodingCompletions()))
    sem = asyncio.Semaphore(1)
    row = asyncio.run(mod._process_one(_sample(), sem, bad_client))
    assert row["phase"] == "gpt_call"
    assert "OpenAI 500" in row["error"]
    assert "raw_body" not in row
