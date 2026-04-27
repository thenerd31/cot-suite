"""Convert Anthropic Messages API output into a Trajectory.

Handles extended thinking blocks. On Claude 4.x, thinking is summarized by
default (billed on full token counts) — we mark those blocks with
`is_summary=True` so Lanham-style CoT interventions can refuse to run.
"""

from __future__ import annotations

from typing import Any

from cotsuite.core.trajectory import Action, Reasoning, Trajectory, Turn

_CLAUDE_SUMMARIZED_FAMILIES = ("claude-opus-4", "claude-sonnet-4", "claude-4")


def _is_summarized(model: str | None) -> bool:
    if not model:
        return False
    return any(model.startswith(f) for f in _CLAUDE_SUMMARIZED_FAMILIES)


def _turn_from_anthropic_message(msg: dict[str, Any], model: str | None) -> Turn:
    role = msg.get("role", "assistant")
    content = msg.get("content", [])
    if isinstance(content, str):
        return Turn(role=role, text=content)

    text_parts: list[str] = []
    reasoning: list[Reasoning] = []
    actions: list[Action] = []

    for block in content:
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            reasoning.append(
                Reasoning(
                    text=block.get("thinking", ""),
                    provider="anthropic",
                    is_summary=_is_summarized(model),
                    signature=block.get("signature"),
                ),
            )
        elif btype == "redacted_thinking":
            reasoning.append(
                Reasoning(
                    text="[REDACTED]",
                    provider="anthropic",
                    is_summary=True,
                ),
            )
        elif btype == "tool_use":
            actions.append(
                Action(
                    tool_name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ),
            )
        elif btype == "tool_result" and actions:
            last = actions[-1]
            actions[-1] = Action(
                tool_name=last.tool_name,
                arguments=last.arguments,
                result=_stringify(block.get("content")),
                error="error" if block.get("is_error") else None,
            )

    return Turn(
        role=role,
        text="".join(text_parts) or None,
        reasoning=reasoning,
        actions=actions,
    )


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in value
        )
    return str(value)


def from_anthropic(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    final_answer: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Trajectory:
    """Convert an Anthropic-style messages list into a `Trajectory`.

    Args:
        messages: The `messages` list passed to / returned from the Anthropic
            Messages API. Supports dict form (both request and response).
        model: The model name (e.g. `"claude-opus-4-5"`) — used to infer
            whether thinking blocks are raw or summarized.
        final_answer: Optional pre-extracted final answer.
        metadata: Extra metadata to attach (sample_id, seed, etc.).
    """
    turns = [_turn_from_anthropic_message(m, model) for m in messages]
    meta = {"provider": "anthropic", "model": model, **(metadata or {})}
    if final_answer is None and turns:
        last = turns[-1]
        if last.role == "assistant":
            final_answer = last.text
    return Trajectory(turns=turns, final_answer=final_answer, metadata=meta)
