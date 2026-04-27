"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock

import pytest

from cotsuite.core.trajectory import Reasoning, Trajectory, Turn


@pytest.fixture
def sample_trajectory() -> Trajectory:
    """A minimal coherent trajectory for metric tests."""
    return Trajectory(
        turns=[
            Turn(role="user", text="What is 17 * 23?"),
            Turn(
                role="assistant",
                text="391",
                reasoning=[
                    Reasoning(
                        text=(
                            "Let me compute 17 * 23. I can break this down as "
                            "17 * 20 + 17 * 3 = 340 + 51 = 391."
                        ),
                        provider="anthropic",
                        is_summary=False,
                    ),
                ],
            ),
        ],
        final_answer="391",
        metadata={"model": "claude-opus-4-5"},
    )


@pytest.fixture
def fake_grader() -> Callable[[dict[str, Any]], AsyncMock]:
    """Factory producing a fake grader that returns a canned score."""

    def _make(payload: dict[str, Any]) -> AsyncMock:
        import json

        client = AsyncMock()
        client.complete = AsyncMock(return_value=json.dumps(payload))
        return client

    return _make
