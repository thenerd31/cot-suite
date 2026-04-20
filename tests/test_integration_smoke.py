"""Live-API smoke test. Skipped by default — requires real API keys.

Runs only when ANTHROPIC_API_KEY is set in the environment; the body
fills in once the verbatim 2510.23966 Appendix C prompt lands and the
smoke-test scope is approved.

Do not add anything to this file without explicit sign-off on the
smoke-test scope. "Just to check the client works" is the scope creep
that burns budget on nothing.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="no API key — set ANTHROPIC_API_KEY to run live smoke test",
)


def test_smoke_placeholder() -> None:
    """Body intentionally empty. Fill after Appendix C prompt lands."""
    pass
