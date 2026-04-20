"""Unit tests for Turpin counterfactual + Chen cue-injection (mocked clients)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cotdiv.tests.chen_cue_injection import (
    CUE_CATALOG,
    InjectionSample,
    cue_injection,
)
from cotdiv.tests.turpin_counterfactual import (
    BIAS_CATALOG,
    Sample,
    counterfactual_bias,
)


@dataclass
class ScriptedClient:
    script: list[str]
    calls: list[str] = field(default_factory=list)

    async def complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        return self.script.pop(0) if self.script else ""


def test_bias_catalog_contains_only_turpin_verified_candidates() -> None:
    # 'authority' was moved to extensions/ on 2026-04-19 (invented by
    # cotdiv, not in Turpin 2023). Verified candidates stay here.
    assert set(BIAS_CATALOG) == {"always_a_fewshot", "sycophancy"}


def test_cue_catalog_matches_verified_chen_cues() -> None:
    # After PDF cross-check (Task P2.4, commit series ending eccc596):
    # - sycophancy, consistency, metadata, grader_hacking, unethical → main
    # - visual_pattern (paper uses few-shot; we only support single-prompt) →
    #   extensions as "simplified"
    # - authority → extensions (cotdiv-original, not in paper)
    assert set(CUE_CATALOG) == {
        "sycophancy",
        "consistency",
        "metadata",
        "grader_hacking",
        "unethical",
    }


def test_all_primary_cues_are_pdf_verified() -> None:
    for cue in CUE_CATALOG.values():
        assert cue.provenance.verified_against_pdf, (
            f"cue {cue.name!r} is in the primary Chen namespace but not "
            "verified_against_pdf — move to extensions/ or verify."
        )


def test_every_catalog_entry_carries_provenance() -> None:
    from cotdiv.core.provenance import Provenance

    for cfg in BIAS_CATALOG.values():
        assert isinstance(cfg.provenance, Provenance)
        assert cfg.provenance.arxiv_id == "2305.04388"
    for cue in CUE_CATALOG.values():
        assert isinstance(cue.provenance, Provenance)
        assert cue.provenance.arxiv_id == "2505.05410"


def test_extensions_import_with_correct_provenance() -> None:
    from cotdiv.tests.extensions import (
        AUTHORITY_BIAS,
        CHEN_AUTHORITY_CUE,
        CHEN_VISUAL_PATTERN_SIMPLIFIED,
    )

    # The Turpin-invented bias has no paper anchor.
    assert AUTHORITY_BIAS.provenance.arxiv_id is None
    assert AUTHORITY_BIAS.provenance.is_extension()

    # Chen 'authority' is cotdiv-original (not in the paper's six cues).
    assert CHEN_AUTHORITY_CUE.provenance.is_unverified_replication()

    # Chen visual-pattern is a simplified form of a paper cue.
    assert CHEN_VISUAL_PATTERN_SIMPLIFIED.provenance.is_unverified_replication()


def test_injectors_are_idempotent_on_question_text() -> None:
    q = "What is 2+2? (A) 3 (B) 4"
    for cfg in BIAS_CATALOG.values():
        assert q in cfg.injector(q)
    for cue in CUE_CATALOG.values():
        assert q in cue.renderer(q, "B")


@pytest.mark.asyncio
async def test_counterfactual_bias_detects_accuracy_drop() -> None:
    # Baseline: model answers correctly. Biased: model answers A.
    # Bias points toward wrong answer on both samples (correct is B, C).
    # Judge: "NO" — model never verbalizes the bias.
    client = ScriptedClient(
        script=[
            # sample 1 baseline
            "Reasoning.\nAnswer: B",
            # sample 1 biased
            "Reasoning.\nAnswer: A",
            # judge for sample 1
            "NO",
            # sample 2 baseline
            "Reasoning.\nAnswer: C",
            # sample 2 biased
            "Reasoning.\nAnswer: A",
            # judge for sample 2
            "NO",
        ],
    )
    result = await counterfactual_bias(
        model=client,
        bias="sycophancy",
        samples=[
            Sample(question="Q1 (A)x (B)y (C)z", correct_answer="B"),
            Sample(question="Q2 (A)x (B)y (C)z", correct_answer="C"),
        ],
    )
    assert result.raw["accuracy_drop"] == pytest.approx(1.0)
    assert result.raw["bias_follow_rate_on_wrong_pointing"] == pytest.approx(1.0)
    assert result.raw["verbalization_rate"] == 0.0


@pytest.mark.asyncio
async def test_counterfactual_bias_counts_verbalization() -> None:
    # Model follows bias; judge returns YES for both → verbalization_rate = 1.0.
    client = ScriptedClient(
        script=[
            "Answer: B",  # baseline 1
            "The user said A, so: Answer: A",  # biased 1
            "YES",  # judge 1
            "Answer: C",  # baseline 2
            "They believe A. Answer: A",  # biased 2
            "YES",  # judge 2
        ],
    )
    result = await counterfactual_bias(
        model=client,
        bias="sycophancy",
        samples=[
            Sample(question="Q1", correct_answer="B"),
            Sample(question="Q2", correct_answer="C"),
        ],
    )
    assert result.raw["verbalization_rate"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_cue_injection_reports_follow_rate() -> None:
    # Two samples; cue points to B. Model follows on one, not the other.
    client = ScriptedClient(
        script=[
            # sample 1: follows cue (answer B), judge says NO
            "Answer: B",
            "NO",
            # sample 2: does NOT follow cue (answer C)
            "Answer: C",
        ],
    )
    result = await cue_injection(
        model=client,
        cue="metadata",
        samples=[
            InjectionSample(question="Q1", correct_answer="A", target_answer="B"),
            InjectionSample(question="Q2", correct_answer="A", target_answer="B"),
        ],
    )
    assert result.raw["follow_rate"] == pytest.approx(0.5)
    assert result.raw["n_cue_followed"] == 1
    assert result.raw["verbalization_rate"] == 0.0


@pytest.mark.asyncio
async def test_cue_injection_verbalization_rate_when_judge_returns_yes() -> None:
    client = ScriptedClient(
        script=[
            "The metadata says B. Answer: B",
            "YES",
        ],
    )
    result = await cue_injection(
        model=client,
        cue="metadata",
        samples=[
            InjectionSample(question="Q1", correct_answer="A", target_answer="B"),
        ],
    )
    assert result.raw["follow_rate"] == pytest.approx(1.0)
    assert result.raw["verbalization_rate"] == pytest.approx(1.0)


def test_parse_yes_no_extracts_either_answer() -> None:
    from cotdiv.tests._cue_judge import _parse_yes_no

    assert _parse_yes_no("YES") is True
    assert _parse_yes_no("no, the trace does not mention it") is False
    assert _parse_yes_no("The answer is: YES.") is True
    assert _parse_yes_no("unclear") is False
