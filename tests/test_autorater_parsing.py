"""Parser-level tests for LegibilityCoveragePrompt (no network calls)."""

from __future__ import annotations

import pytest

from cotmon.autoraters.legibility_coverage import LegibilityCoveragePrompt


def test_loads_default_prompt_and_computes_sha() -> None:
    prompt = LegibilityCoveragePrompt.load()
    assert prompt.version == "emmons_zimmermann_v1"
    assert len(prompt.sha256) == 64
    assert "Legibility" in prompt.template
    assert "Coverage" in prompt.template


def test_renders_all_three_slots() -> None:
    prompt = LegibilityCoveragePrompt.load()
    rendered = prompt.render(
        question="What is 2+2?",
        explanation="Count on fingers, then add.",
        answer="4",
    )
    assert "What is 2+2?" in rendered
    assert "Count on fingers" in rendered
    assert "4" in rendered


def test_renders_preserves_json_schema_braces() -> None:
    # Appendix C has a JSON schema at the bottom containing `{...}`. The
    # renderer must not interpret those braces as template fields — it should
    # leave them literally intact in the output.
    prompt = LegibilityCoveragePrompt.load()
    rendered = prompt.render(question="q", explanation="e", answer="a")
    assert '"legibility_score": 0 to 4' in rendered
    assert '"coverage_score": 0 to 4' in rendered
    assert '"justification"' in rendered


def test_parses_well_formed_json_output() -> None:
    prompt = LegibilityCoveragePrompt.load()
    completion = (
        'Here is my rating: {"justification": "clear", "legibility_score": 4, "coverage_score": 3}'
    )
    leg, cov, justification = prompt.parse(completion)
    assert leg == 4
    assert cov == 3
    assert justification == "clear"


def test_parses_fenced_code_block_output() -> None:
    # Real autorater outputs will often be wrapped in ```json ... ``` fences
    # per the Appendix C output-format instruction.
    prompt = LegibilityCoveragePrompt.load()
    completion = """```json
{
  "justification": "reasoning was clear and covered all steps",
  "legibility_score": 4,
  "coverage_score": 4
}
```"""
    leg, cov, justification = prompt.parse(completion)
    assert leg == 4
    assert cov == 4
    assert "reasoning was clear" in justification


def test_parse_rejects_out_of_range() -> None:
    prompt = LegibilityCoveragePrompt.load()
    with pytest.raises(ValueError, match="out-of-range"):
        prompt.parse('{"justification": "x", "legibility_score": 5, "coverage_score": 3}')


def test_parse_rejects_missing_json() -> None:
    prompt = LegibilityCoveragePrompt.load()
    with pytest.raises(ValueError, match="no JSON object"):
        prompt.parse("no object here")


def test_parse_rejects_malformed_json() -> None:
    prompt = LegibilityCoveragePrompt.load()
    with pytest.raises(ValueError, match="not valid JSON"):
        prompt.parse("{bogus,")
