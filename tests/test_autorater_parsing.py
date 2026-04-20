"""Parser-level tests for LegibilityCoveragePrompt (no network calls)."""

from __future__ import annotations

import pytest

from cotdiv.autoraters.legibility_coverage import LegibilityCoveragePrompt


def test_loads_default_prompt_and_computes_sha() -> None:
    prompt = LegibilityCoveragePrompt.load()
    assert prompt.version == "emmons_zimmermann_v1"
    assert len(prompt.sha256) == 64
    assert "Legibility" in prompt.template
    assert "Coverage" in prompt.template


def test_renders_all_three_slots() -> None:
    prompt = LegibilityCoveragePrompt.load()
    rendered = prompt.render(prompt="What is 2+2?", reasoning="Count", answer="4")
    assert "What is 2+2?" in rendered
    assert "Count" in rendered
    assert "4" in rendered


def test_parses_well_formed_json_output() -> None:
    prompt = LegibilityCoveragePrompt.load()
    completion = '  Here is my rating: {"legibility": 4, "coverage": 3, "rationale": "clear"}  '
    leg, cov, rat = prompt.parse(completion)
    assert leg == 4
    assert cov == 3
    assert rat == "clear"


def test_parse_rejects_out_of_range() -> None:
    prompt = LegibilityCoveragePrompt.load()
    with pytest.raises(ValueError, match="out-of-range"):
        prompt.parse('{"legibility": 5, "coverage": 3, "rationale": "x"}')


def test_parse_rejects_missing_json() -> None:
    prompt = LegibilityCoveragePrompt.load()
    with pytest.raises(ValueError, match="no JSON object"):
        prompt.parse("no object here")
