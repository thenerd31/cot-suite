"""Thin async client wrappers over Anthropic / OpenAI / Google / open-weights.

The grader client has one method, `.complete(prompt) -> str`. Real deployments
should plug Inspect AI's `get_model(..., role="grader")` instead; this wrapper
exists so metric code can be used outside of Inspect.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class GraderClient(Protocol):
    async def complete(self, prompt: str) -> str: ...


@dataclass
class GeminiClient:
    model: str

    async def complete(self, prompt: str) -> str:
        from google import genai  # type: ignore[import-not-found]

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        # google-genai exposes sync + async; favour async when available.
        response = await client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        return response.text or ""


@dataclass
class AnthropicClient:
    model: str

    async def complete(self, prompt: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return "".join(parts)


@dataclass
class OpenAIClient:
    model: str

    async def complete(self, prompt: str) -> str:
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


def get_grader_client(spec: str) -> GraderClient:
    """Parse `provider/model` and return an async client.

    Examples:
        - `"google/gemini-2.5-pro"` — Google Gemini via google-genai
        - `"anthropic/claude-opus-4-5"` — Anthropic
        - `"openai/gpt-5.1"` — OpenAI
    """
    if "/" not in spec:
        raise ValueError(f"model spec must be `provider/name`, got {spec!r}")
    provider, name = spec.split("/", 1)
    match provider:
        case "google" | "gemini":
            return GeminiClient(model=name)
        case "anthropic":
            return AnthropicClient(model=name)
        case "openai":
            return OpenAIClient(model=name)
        case _:
            raise ValueError(f"unknown provider {provider!r} (got spec {spec!r})")
