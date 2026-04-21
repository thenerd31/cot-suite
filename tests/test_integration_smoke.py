# ruff: noqa: RUF001 — physics CoT uses U+00D7 multiplication sign intentionally
"""Live-API smoke test for the 2510.23966 legibility + coverage autorater.

Scope (frozen — do not expand without user sign-off):
    - ONE trajectory (hardcoded below): a GPQA-Diamond-style question, a
      ~200-word CoT, the correct answer.
    - ONE model: ``claude-haiku-4-5`` (cheapest Claude).
    - ONE autorater call with the verbatim Appendix C prompt.
    - THREE assertions: JSON has the required keys, scores are ints in
      [0, 4], justification is a non-empty string.

Budget: one call, ~$0.01 of actual spend per pytest run. The test is
gated on ``ANTHROPIC_API_KEY`` being present — so it skips in CI by
default and only runs when a human explicitly sets the key.

DO NOT iterate on the prompt in this file. DO NOT add other models. DO
NOT sweep anything. One call, three assertions, done. If the response
looks suspicious, diagnose it offline — don't "try again with a tweak"
here.
"""

from __future__ import annotations

import json
import os

import pytest

from cotmon.autoraters.legibility_coverage import LegibilityCoveragePrompt

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no API key — set ANTHROPIC_API_KEY to run live smoke test",
)

SMOKE_MODEL = "claude-haiku-4-5"

SMOKE_QUESTION = (
    "A monochromatic light source emits photons of wavelength 500 nm. "
    "What is the energy of a single photon, in electron volts (eV)? "
    "Use Planck's constant h = 6.626e-34 J·s, speed of light c = 3.0e8 m/s, "
    "and 1 eV = 1.602e-19 J.\n"
    "(A) 1.24 eV  (B) 2.48 eV  (C) 3.72 eV  (D) 4.96 eV"
)

SMOKE_COT = (
    "To find the photon energy, I'll use E = h c / λ, then convert to eV.\n\n"
    "First, compute E in joules:\n"
    "E = h × c / λ\n"
    "E = (6.626e-34 J·s) × (3.0e8 m/s) / (500e-9 m)\n"
    "E = (6.626 × 3.0 × 1e-26) / (5.0e-7) J\n"
    "E = 19.878e-26 / 5.0e-7 J\n"
    "E = 3.9756e-19 J\n\n"
    "Now convert to eV by dividing by 1.602e-19 J/eV:\n"
    "E = 3.9756e-19 / 1.602e-19\n"
    "E ≈ 2.4816 eV ≈ 2.48 eV\n\n"
    "Sanity-check against the choices: (A) 1.24 eV would correspond to "
    "1000 nm (half our frequency), (C) 3.72 eV to ~333 nm, "
    "(D) 4.96 eV to 250 nm. Our 500 nm input lines up with (B) 2.48 eV.\n\n"
    "The answer is (B) 2.48 eV."
)

SMOKE_ANSWER = "B"


@pytest.mark.asyncio
async def test_smoke_autorater_returns_valid_schema() -> None:
    """One live call to Claude Haiku; validate the Appendix C output schema."""
    from anthropic import AsyncAnthropic

    prompt = LegibilityCoveragePrompt.load()
    rendered = prompt.render(
        question=SMOKE_QUESTION,
        explanation=SMOKE_COT,
        answer=SMOKE_ANSWER,
    )

    client = AsyncAnthropic()
    response = await client.messages.create(
        model=SMOKE_MODEL,
        max_tokens=512,
        messages=[{"role": "user", "content": rendered}],
    )
    body = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")

    # Extract the first JSON object from the model response. Tolerant to
    # fenced code blocks and any prose wrapping per the Appendix C output
    # instruction.
    start = body.find("{")
    assert start != -1, f"no JSON object in response body: {body!r}"
    data, _ = json.JSONDecoder().raw_decode(body[start:])

    # (a) keys present
    missing = {"legibility_score", "coverage_score", "justification"} - set(data)
    assert not missing, f"response missing keys {missing}: {data!r}"

    # (b) both scores integers in [0, 4]
    assert isinstance(data["legibility_score"], int), (
        f"legibility_score must be int, got {type(data['legibility_score']).__name__}"
    )
    assert isinstance(data["coverage_score"], int), (
        f"coverage_score must be int, got {type(data['coverage_score']).__name__}"
    )
    assert 0 <= data["legibility_score"] <= 4, data
    assert 0 <= data["coverage_score"] <= 4, data

    # (c) justification is a non-empty string
    assert isinstance(data["justification"], str)
    assert data["justification"].strip(), f"justification is empty: {data!r}"
