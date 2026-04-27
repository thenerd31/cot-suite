"""Canonical trajectory data model consumed by every metric and test.

Every adapter in `cotsuite.adapters` produces these types; every metric consumes
them. Keeping this surface small and immutable is load-bearing for cross-provider
reproducibility.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Provider = Literal[
    "anthropic",
    "openai",
    "gemini",
    "deepseek",
    "grok",
    "qwen",
    "llama",
    "gpt-oss",
    "open",
    "unknown",
]

Role = Literal["system", "user", "assistant", "tool"]


class Reasoning(BaseModel):
    """A single reasoning block (thought, thinking, or reasoning trace).

    `is_summary=True` means the provider returned only a summarized trace
    (Claude 4.x by default, OpenAI o-series `concise`/`auto`). Lanham tests
    that intervene on CoT content MUST refuse to run on summarized reasoning.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    provider: Provider = "unknown"
    is_summary: bool = False
    raw_tokens: int | None = None
    signature: str | None = None


class Action(BaseModel):
    """A single tool/function invocation and its result."""

    model_config = ConfigDict(frozen=True)

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    error: str | None = None


class Turn(BaseModel):
    """One conversational turn with optional interleaved reasoning and actions."""

    model_config = ConfigDict(frozen=True)

    role: Role
    text: str | None = None
    reasoning: list[Reasoning] = Field(default_factory=list)
    actions: list[Action] = Field(default_factory=list)


class Trajectory(BaseModel):
    """A full multi-turn trajectory.

    `final_answer` is the extracted final answer (typically the assistant's
    last text output, minus reasoning). `metadata` carries provider, model
    name, sample ID, seeds, prompt hashes, etc.
    """

    model_config = ConfigDict(frozen=True)

    turns: list[Turn] = Field(default_factory=list)
    final_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def reasoning_text(self) -> str:
        """Concatenated reasoning across all turns (chronological)."""
        parts: list[str] = []
        for turn in self.turns:
            parts.extend(r.text for r in turn.reasoning)
        return "\n\n".join(parts)

    @property
    def has_summarized_reasoning(self) -> bool:
        """True if any reasoning block is a provider summary (not raw CoT)."""
        return any(r.is_summary for t in self.turns for r in t.reasoning)

    @property
    def actions(self) -> list[Action]:
        """All actions across all turns, in chronological order."""
        return [a for t in self.turns for a in t.actions]
