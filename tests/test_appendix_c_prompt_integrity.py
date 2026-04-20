"""Structural integrity checks for the verbatim 2510.23966 Appendix C prompt.

These do NOT exercise an autorater — no network calls. They assert the
shipped prompt file has the expected structural features (all 7 example
rows present, all three template placeholders present, SHA-256 matches
the committed canonical hash). Any mutation of the shipped file that
these checks catch was almost certainly an accident — bump the prompt
version instead.
"""

from __future__ import annotations

import hashlib
import re
from importlib import resources

from cotdiv.autoraters.legibility_coverage import LegibilityCoveragePrompt


def _load_template() -> str:
    pkg = resources.files("cotdiv.autoraters.prompts")
    return (pkg / "emmons_zimmermann_v1.txt").read_text(encoding="utf-8")


def test_prompt_sha256_matches_canonical_hash() -> None:
    """The shipped prompt must hash to the value committed alongside it.

    This is the guardrail for the "never mutate a shipped prompt" invariant —
    if this test fails, either the prompt was edited (bump the version, do
    not overwrite) or the canonical hash file drifted.
    """
    template = _load_template()
    actual = hashlib.sha256(template.encode("utf-8")).hexdigest()
    canonical = LegibilityCoveragePrompt.canonical_sha256("emmons_zimmermann_v1")
    assert actual == canonical, f"prompt SHA drift: actual={actual[:12]} canonical={canonical[:12]}"


def test_prompt_contains_all_three_placeholders_exactly_once() -> None:
    template = _load_template()
    for placeholder in ("{question}", "{explanation}", "{answer}"):
        count = template.count(placeholder)
        assert count == 1, f"placeholder {placeholder} appears {count} times (expected 1)"


def test_prompt_contains_seven_example_rating_rows() -> None:
    """Appendix C ships an 8-row examples table: 7 rated + 1 N/A. The rated
    rows each carry a `**Legibility: N/4, Coverage: N/4**` label."""
    template = _load_template()
    ratings = re.findall(r"\*\*Legibility: [0-4]/4, Coverage: [0-4]/4\*\*", template)
    assert len(ratings) == 7, f"expected 7 example ratings, found {len(ratings)}"
    # Plus the single **N/A** row for the incorrect-final-answer example.
    assert template.count("**N/A**") == 1


def test_prompt_uses_standard_triple_backticks_not_curly_quotes() -> None:
    """Regression guard against the three curly-quote fences (U+2018 x3) that
    wrap the JSON schema in the raw paper text. Our shipped file normalizes
    those to standard backticks."""
    template = _load_template()
    # No triple-U+2018 sequences anywhere.
    assert "\u2018\u2018\u2018" not in template
    # The JSON schema fences exist as backticks.
    assert "```json" in template
    assert template.count("```") >= 2


def test_prompt_has_german_umlauts_not_latex_escapes() -> None:
    """Regression guard against the LaTeX \\\"o escapes from the raw paper."""
    template = _load_template()
    for umlaut_word in ("Länge", "kürzeren", "längsten", "längste", "fünf"):
        assert umlaut_word in template, f"missing German umlaut word: {umlaut_word}"
    # No residual LaTeX escapes.
    assert '\\"' not in template


def test_prompt_declares_required_output_schema() -> None:
    template = _load_template()
    for key in ('"justification"', '"legibility_score"', '"coverage_score"'):
        assert key in template, f"output schema missing key: {key}"
