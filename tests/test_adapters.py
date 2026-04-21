"""Tests for Anthropic and OpenAI message adapters."""

from __future__ import annotations

from cotmon.adapters import from_anthropic, from_openai


def test_anthropic_thinking_block_marked_as_summary_on_claude_4() -> None:
    messages = [
        {"role": "user", "content": "Solve 2+2"},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Adding 2 and 2 yields 4.", "signature": "sig"},
                {"type": "text", "text": "4"},
            ],
        },
    ]
    traj = from_anthropic(messages, model="claude-opus-4-5")
    assert len(traj.turns) == 2
    assistant = traj.turns[1]
    assert assistant.text == "4"
    assert len(assistant.reasoning) == 1
    assert assistant.reasoning[0].is_summary is True
    assert assistant.reasoning[0].signature == "sig"
    assert traj.final_answer == "4"


def test_anthropic_thinking_block_raw_on_claude_3_7() -> None:
    messages = [
        {"role": "user", "content": "?"},
        {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": "raw"}, {"type": "text", "text": "ok"}],
        },
    ]
    traj = from_anthropic(messages, model="claude-3-7-sonnet")
    assert traj.turns[1].reasoning[0].is_summary is False


def test_anthropic_tool_use_becomes_action() -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "name": "search", "input": {"q": "cats"}},
            ],
        },
    ]
    traj = from_anthropic(messages, model="claude-opus-4-5")
    assert traj.turns[0].actions[0].tool_name == "search"
    assert traj.turns[0].actions[0].arguments == {"q": "cats"}


def test_openai_reasoning_summaries_marked_as_summary() -> None:
    messages = [
        {"role": "user", "content": "Solve"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "reasoning",
                    "summary": [{"text": "Thinking about the problem..."}],
                },
                {"type": "output_text", "text": "42"},
            ],
        },
    ]
    traj = from_openai(messages, model="o4-mini")
    assistant = traj.turns[1]
    assert assistant.reasoning[0].is_summary is True
    assert "Thinking about the problem" in assistant.reasoning[0].text
    assert assistant.text == "42"


def test_openai_tool_call_string_args_parsed() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": '{"id": 7}'}},
            ],
        },
    ]
    traj = from_openai(messages, model="gpt-5.1")
    assert traj.turns[0].actions[0].arguments == {"id": 7}


def test_openai_malformed_tool_args_retained_as_raw() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "bad", "arguments": "not-json"}},
            ],
        },
    ]
    traj = from_openai(messages, model="gpt-5.1")
    assert traj.turns[0].actions[0].arguments == {"_raw": "not-json"}
