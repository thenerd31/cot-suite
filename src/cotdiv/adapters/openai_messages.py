"""Convert OpenAI Chat Completions / Responses API output into a Trajectory.

o-series and GPT-5 reasoning models expose reasoning only as summaries
(`concise` / `detailed` / `auto`). `detailed` is gated on org verification.
All OpenAI reasoning blocks are marked `is_summary=True` unless an explicit
override is passed — Lanham tests refuse to run on them.
"""

from __future__ import annotations

from typing import Any

from cotdiv.core.trajectory import Action, Reasoning, Trajectory, Turn


def _turn_from_openai_message(
    msg: dict[str, Any],
    *,
    summarized: bool,
) -> Turn:
    role = msg.get("role", "assistant")
    content = msg.get("content")

    text: str | None
    reasoning: list[Reasoning] = []
    if isinstance(content, str):
        text = content or None
    elif isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            btype = block.get("type")
            if btype in {"text", "output_text"}:
                text_parts.append(block.get("text", ""))
            elif btype == "reasoning":
                for summary in block.get("summary", []) or []:
                    reasoning.append(
                        Reasoning(
                            text=summary.get("text", ""),
                            provider="openai",
                            is_summary=True,
                        ),
                    )
                if "content" in block:
                    reasoning.append(
                        Reasoning(
                            text=_stringify(block["content"]),
                            provider="openai",
                            is_summary=summarized,
                        ),
                    )
        text = "".join(text_parts) or None
    else:
        text = None

    actions: list[Action] = []
    for tc in msg.get("tool_calls", []) or []:
        fn = tc.get("function", {})
        actions.append(
            Action(
                tool_name=fn.get("name", ""),
                arguments=_parse_args(fn.get("arguments")),
            ),
        )

    return Turn(role=role, text=text, reasoning=reasoning, actions=actions)


def _parse_args(raw: Any) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json

        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {"_raw": str(raw)}


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


def from_openai(
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    final_answer: str | None = None,
    metadata: dict[str, Any] | None = None,
    raw_reasoning: bool = False,
) -> Trajectory:
    """Convert an OpenAI-style messages list into a `Trajectory`.

    Args:
        messages: OpenAI Chat Completions or Responses API messages.
        model: Model name, e.g. `"o4-mini"`, `"gpt-5.1"`.
        raw_reasoning: Set to True ONLY if the caller is on a plan that returns
            raw (not summarized) reasoning — otherwise all OpenAI reasoning
            blocks are conservatively marked as summaries.
    """
    turns = [_turn_from_openai_message(m, summarized=not raw_reasoning) for m in messages]
    meta = {"provider": "openai", "model": model, **(metadata or {})}
    if final_answer is None and turns:
        last = turns[-1]
        if last.role == "assistant":
            final_answer = last.text
    return Trajectory(turns=turns, final_answer=final_answer, metadata=meta)
