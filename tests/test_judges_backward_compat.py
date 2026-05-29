"""Backward-compat: the new list-based judge interfaces are strictly additive.

`judges_verbalizes` keeps its single-`GraderClient` → `bool` contract and gains
a `list[GraderClient]` → `list[bool]` overload. The two Inspect scorers keep
their single-judge signatures and gain `autoraters=` / `judges=` list kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass

from cotsuite.tests._cue_judge import judges_verbalizes


@dataclass
class _StubJudge:
    reply: str

    async def complete(self, prompt: str) -> str:
        return self.reply


# --- judges_verbalizes ------------------------------------------------------


async def test_single_judge_returns_bool() -> None:
    assert await judges_verbalizes(_StubJudge("YES"), cue_description="c", cot="x") is True
    assert await judges_verbalizes(_StubJudge("NO"), cue_description="c", cot="x") is False


async def test_list_of_judges_returns_bool_list() -> None:
    judges = [_StubJudge("YES"), _StubJudge("NO"), _StubJudge("yes, clearly")]
    out = await judges_verbalizes(judges, cue_description="c", cot="x")
    assert out == [True, False, True]


async def test_single_element_list_is_a_list() -> None:
    out = await judges_verbalizes([_StubJudge("YES")], cue_description="c", cot="x")
    assert out == [True]


# --- scorer factories accept the additive list kwargs (offline construction) -


def test_legibility_accepts_autoraters_kwarg() -> None:
    from cotsuite.inspect.scorers import cot_legibility_coverage

    assert cot_legibility_coverage(autoraters=["openai/gpt-5.1", "anthropic/claude-x"]) is not None
    assert cot_legibility_coverage() is not None  # single-judge default path intact


def test_phr_accepts_judges_kwarg() -> None:
    from cotsuite.inspect.scorers import cot_post_hoc_rationalization

    assert cot_post_hoc_rationalization(judges=["openai/gpt-5.1", "anthropic/claude-x"]) is not None
    assert cot_post_hoc_rationalization() is not None  # single-judge default path intact
