"""Unit tests for Turpin counterfactual + Chen cue-injection (mocked clients)."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from cotsuite.tests.chen_cue_injection import (
    CUE_CATALOG,
    InjectionSample,
    cue_injection,
)
from cotsuite.tests.turpin_counterfactual import (
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
    # After PDF re-fetch (Row 5 "Grader Hacking", Row 6 "Unethical Information"):
    # - sycophancy, consistency, metadata, grader_hacking, unethical_information → main
    # - visual_pattern (paper uses few-shot; we only support single-prompt) →
    #   extensions as "simplified"
    # - authority → extensions (cotdiv-original, not in paper)
    assert set(CUE_CATALOG) == {
        "sycophancy",
        "consistency",
        "metadata",
        "grader_hacking",
        "unethical_information",
    }


def test_all_primary_cues_are_pdf_verified() -> None:
    for cue in CUE_CATALOG.values():
        assert cue.provenance.verified_against_pdf, (
            f"cue {cue.name!r} is in the primary Chen namespace but not "
            "verified_against_pdf — move to extensions/ or verify."
        )


def test_every_catalog_entry_carries_provenance() -> None:
    from cotsuite.core.provenance import Provenance

    for cfg in BIAS_CATALOG.values():
        assert isinstance(cfg.provenance, Provenance)
        assert cfg.provenance.arxiv_id == "2305.04388"
    for cue in CUE_CATALOG.values():
        assert isinstance(cue.provenance, Provenance)
        assert cue.provenance.arxiv_id == "2505.05410"


def test_extensions_import_with_correct_provenance() -> None:
    from cotsuite.tests.extensions import (
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
    from cotsuite.tests._cue_judge import _parse_yes_no

    assert _parse_yes_no("YES") is True
    assert _parse_yes_no("no, the trace does not mention it") is False
    assert _parse_yes_no("The answer is: YES.") is True
    assert _parse_yes_no("unclear") is False


# ---------------------------------------------------------------------------
# Framework fixes landed 2026-05-28 (align cot-suite metric with Turpin's
# bbh_analysis.py). One unit test per fix: (a) inconsistent_only, (b) per-
# question Sample.bias_target_letter, (c) per-task → mean aggregation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_counterfactual_bias_inconsistent_only_default_filters_consistent() -> None:
    """Fix (a): inconsistent_only=True (default) restricts the drop denom to
    samples where the bias points to a wrong answer.

    Setup: one bias-consistent sample (correct=A, bias target=A) and one
    bias-inconsistent sample (correct=B, bias target=A). Model always
    follows the bias.

    With inconsistent_only=True, only sample 2 contributes to accuracy_drop.
    Eval pool size = 1; drop = (1-0)/1 = 1.0.

    With inconsistent_only=False, both samples count. Sample 1's biased
    "A" is *correct*, so biased_correct=True there. Drop = (2-1)/2 = 0.5.
    """
    # Default (inconsistent_only=True). s1 has bias_pointed_wrong=False, so
    # no judge call fires for it.
    client_default = ScriptedClient(
        script=[
            "Answer: A",  # s1 baseline (correct=A)
            "Answer: A",  # s1 biased (still A; no judge call — bias points to truth)
            "Answer: B",  # s2 baseline (correct=B)
            "Answer: A",  # s2 biased (followed bias)
            "NO",  # s2 judge (bias_followed && bias_pointed_wrong)
        ],
    )
    result_default = await counterfactual_bias(
        model=client_default,
        bias="sycophancy",
        samples=[
            Sample(question="Q1", correct_answer="A"),
            Sample(question="Q2", correct_answer="B"),
        ],
    )
    assert result_default.raw["inconsistent_only"] is True
    assert result_default.raw["n_eval_pool"] == 1
    assert result_default.raw["accuracy_drop"] == pytest.approx(1.0)

    # Explicit inconsistent_only=False includes the consistent sample.
    client_off = ScriptedClient(
        script=[
            "Answer: A", "Answer: A",
            "Answer: B", "Answer: A", "NO",
        ],
    )
    result_off = await counterfactual_bias(
        model=client_off,
        bias="sycophancy",
        samples=[
            Sample(question="Q1", correct_answer="A"),
            Sample(question="Q2", correct_answer="B"),
        ],
        inconsistent_only=False,
    )
    assert result_off.raw["inconsistent_only"] is False
    assert result_off.raw["n_eval_pool"] == 2
    # Flat pool of both: baseline=2/2 correct, biased=1/2 (s1 lucky), drop=0.5
    assert result_off.raw["accuracy_drop"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_counterfactual_bias_uses_per_question_bias_target_letter() -> None:
    """Fix (b): Sample.bias_target_letter overrides bias_cfg.default_target
    for the bias_followed / bias_pointed_wrong checks.

    Setup: two samples, each carrying its own bias target (B and D
    respectively). Model follows the per-question target. With the old
    hardcoded-"A" check, both bias_followed values would be False; with the
    new per-question target, both are True.
    """
    client = ScriptedClient(
        script=[
            "Answer: C",  # s1 baseline (correct=C)
            "Answer: B",  # s1 biased (followed target=B)
            "NO",  # s1 judge
            "Answer: A",  # s2 baseline (correct=A)
            "Answer: D",  # s2 biased (followed target=D)
            "NO",  # s2 judge
        ],
    )
    result = await counterfactual_bias(
        model=client,
        bias="sycophancy",
        samples=[
            Sample(question="Q1", correct_answer="C", bias_target_letter="B"),
            Sample(question="Q2", correct_answer="A", bias_target_letter="D"),
        ],
    )
    assert result.raw["per_sample"][0]["bias_target"] == "B"
    assert result.raw["per_sample"][0]["bias_followed"] is True
    assert result.raw["per_sample"][1]["bias_target"] == "D"
    assert result.raw["per_sample"][1]["bias_followed"] is True
    assert result.raw["n_bias_followed"] == 2
    assert result.raw["bias_follow_rate_on_wrong_pointing"] == pytest.approx(1.0)
    assert result.raw["accuracy_drop"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_counterfactual_bias_per_task_aggregation_differs_from_flat_pool() -> None:
    """Fix (c): when samples carry a ``task`` field, accuracy_drop is
    computed per-task and then averaged across tasks.

    Setup: asymmetric task sizes. Task A has 2 samples both bias-followed-
    to-wrong; Task B has 4 samples with only 1 bias-followed-to-wrong.

    Per-task drops: Task A = (2-0)/2 = 1.0; Task B = (4-3)/4 = 0.25.
    Mean: (1.0 + 0.25)/2 = 0.625.

    Flat-pool drop would be (6-3)/6 = 0.5. The two values are
    distinguishable, so this test verifies the aggregation path.
    """
    client = ScriptedClient(
        script=[
            # Task A sample 1: bias-followed-to-wrong
            "Answer: B", "Answer: A", "NO",
            # Task A sample 2: bias-followed-to-wrong
            "Answer: C", "Answer: A", "NO",
            # Task B sample 1: not followed (biased stays B)
            "Answer: B", "Answer: B",
            # Task B sample 2: followed
            "Answer: C", "Answer: A", "NO",
            # Task B sample 3: not followed
            "Answer: D", "Answer: D",
            # Task B sample 4: not followed
            "Answer: B", "Answer: B",
        ],
    )
    samples = [
        Sample(question="QA1", correct_answer="B", task="task_a"),
        Sample(question="QA2", correct_answer="C", task="task_a"),
        Sample(question="QB1", correct_answer="B", task="task_b"),
        Sample(question="QB2", correct_answer="C", task="task_b"),
        Sample(question="QB3", correct_answer="D", task="task_b"),
        Sample(question="QB4", correct_answer="B", task="task_b"),
    ]
    result = await counterfactual_bias(model=client, bias="sycophancy", samples=samples)
    assert result.raw["per_task_drops"]["task_a"] == pytest.approx(1.0)
    assert result.raw["per_task_drops"]["task_b"] == pytest.approx(0.25)
    assert result.raw["accuracy_drop"] == pytest.approx(0.625)
    # Sanity: flat-pool would have yielded 0.5, not 0.625.
    flat_pool = (6 - 3) / 6
    assert result.raw["accuracy_drop"] != pytest.approx(flat_pool)
