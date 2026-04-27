"""Tests for the core Trajectory data model."""

from __future__ import annotations

from cotsuite.core.trajectory import Action, Reasoning, Trajectory, Turn


def test_reasoning_text_concatenates_in_order() -> None:
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                reasoning=[
                    Reasoning(text="first", provider="anthropic"),
                    Reasoning(text="second", provider="anthropic"),
                ],
            ),
            Turn(
                role="assistant",
                reasoning=[Reasoning(text="third", provider="anthropic")],
            ),
        ],
    )
    assert traj.reasoning_text == "first\n\nsecond\n\nthird"


def test_has_summarized_reasoning_flag() -> None:
    raw = Trajectory(
        turns=[Turn(role="assistant", reasoning=[Reasoning(text="x", is_summary=False)])],
    )
    summary = Trajectory(
        turns=[Turn(role="assistant", reasoning=[Reasoning(text="x", is_summary=True)])],
    )
    assert raw.has_summarized_reasoning is False
    assert summary.has_summarized_reasoning is True


def test_actions_collects_across_turns() -> None:
    traj = Trajectory(
        turns=[
            Turn(
                role="assistant",
                actions=[Action(tool_name="search", arguments={"q": "x"})],
            ),
            Turn(
                role="assistant",
                actions=[
                    Action(tool_name="read", arguments={"path": "/a"}),
                    Action(tool_name="write", arguments={"path": "/b"}),
                ],
            ),
        ],
    )
    tools = [a.tool_name for a in traj.actions]
    assert tools == ["search", "read", "write"]


def test_trajectory_is_frozen() -> None:
    traj = Trajectory(turns=[Turn(role="user", text="hi")])
    try:
        traj.final_answer = "mutated"  # type: ignore[misc]
    except (TypeError, ValueError, AttributeError):
        return
    raise AssertionError("Trajectory should be frozen")
