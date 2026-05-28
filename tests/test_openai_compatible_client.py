"""Unit tests for the generic OpenAICompatibleClient + the 4 new provider checks.

All mocked — no network, no API spend. Covers P1a (the OpenAI-compatible
providers: together / fireworks / deepseek / openrouter). Modal (P1b) is out of
scope here; its check is unchanged.
"""

from __future__ import annotations

import sys
import types

import pytest

from cotsuite.models.clients import (
    OPENAI_COMPATIBLE_PROVIDERS,
    AnthropicClient,
    GeminiClient,
    OpenAIClient,
    OpenAICompatibleClient,
    get_grader_client,
)

_PROVIDERS = ["together", "fireworks", "deepseek", "openrouter"]
_BASE_URL = {
    "together": "https://api.together.ai/v1",
    "fireworks": "https://api.fireworks.ai/inference/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_ENV_KEY = {
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
# Mock key: >20 chars, no placeholder markers (notably no "xxx").
_MOCK_KEY = "sk-test-0123456789abcdef0123456789"


# --- dispatch routing -------------------------------------------------------


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_dispatch_routes_to_openai_compatible(provider: str) -> None:
    client = get_grader_client(f"{provider}/some-model-v1")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "some-model-v1"
    assert client.base_url == _BASE_URL[provider]
    assert client.env_key == _ENV_KEY[provider]


def test_dispatch_preserves_namespaced_model_id() -> None:
    # spec splits on the FIRST "/", so provider-namespaced model IDs survive.
    client = get_grader_client("together/Qwen/QwQ-32B")
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "Qwen/QwQ-32B"
    assert client.base_url == _BASE_URL["together"]


def test_dispatch_preserves_existing_providers() -> None:
    assert isinstance(get_grader_client("openai/gpt-5.1"), OpenAIClient)
    assert isinstance(get_grader_client("anthropic/claude-x"), AnthropicClient)
    assert isinstance(get_grader_client("google/gemini-2.5-pro"), GeminiClient)
    assert isinstance(get_grader_client("gemini/gemini-2.5-pro"), GeminiClient)


def test_dispatch_unknown_provider_still_raises() -> None:
    with pytest.raises(ValueError):
        get_grader_client("nonsense/model")


def test_registry_has_exactly_four_providers() -> None:
    assert set(OPENAI_COMPATIBLE_PROVIDERS) == set(_PROVIDERS)
    # Modal must NOT be here (P1b — it is not OpenAI-compatible as deployed).
    assert "modal" not in OPENAI_COMPATIBLE_PROVIDERS


# --- complete() wiring (mocked OpenAI SDK) ----------------------------------


@pytest.mark.asyncio
async def test_complete_calls_sdk_with_base_url_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", _MOCK_KEY)
    captured: dict = {}

    class _Msg:
        content = "hello from a mocked provider"

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = _Chat()

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    client = get_grader_client("together/Qwen/QwQ-32B")
    out = await client.complete("hi there")

    assert out == "hello from a mocked provider"
    assert captured["init"]["base_url"] == "https://api.together.ai/v1"
    assert captured["init"]["api_key"] == _MOCK_KEY
    assert captured["model"] == "Qwen/QwQ-32B"
    assert captured["messages"] == [{"role": "user", "content": "hi there"}]


@pytest.mark.asyncio
async def test_complete_handles_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", _MOCK_KEY)

    class _Msg:
        content = None

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    class _Completions:
        async def create(self, **kwargs):
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _FakeAsyncOpenAI:
        def __init__(self, **kwargs):
            self.chat = _Chat()

    fake_openai = types.ModuleType("openai")
    fake_openai.AsyncOpenAI = _FakeAsyncOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", fake_openai)

    out = await get_grader_client("deepseek/deepseek-v4-pro").complete("x")
    assert out == ""  # None content coalesces to empty string


# --- verify_keys checks (mocked) --------------------------------------------


def _import_checks():
    from cotsuite.verify_keys import (
        PROVIDER_CHECKS,
        check_deepseek,
        check_fireworks,
        check_openrouter,
        check_together,
    )

    return PROVIDER_CHECKS, {
        "together": check_together,
        "fireworks": check_fireworks,
        "deepseek": check_deepseek,
        "openrouter": check_openrouter,
    }


def test_provider_checks_registry_extended() -> None:
    registry, _ = _import_checks()
    for p in _PROVIDERS:
        assert p in registry
    # existing providers still present
    for p in ["anthropic", "openai", "huggingface", "modal"]:
        assert p in registry


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_check_fails_when_key_missing(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_ENV_KEY[provider], raising=False)
    _, checks = _import_checks()
    result = checks[provider]()
    assert result.ok is False
    assert _ENV_KEY[provider] in result.detail


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_check_ok_with_mock_key(provider: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_ENV_KEY[provider], _MOCK_KEY)
    import openai  # real package present in the dev env

    captured: dict = {}

    class _FakeModels:
        def list(self):
            return iter([object()])

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = _FakeModels()

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    _, checks = _import_checks()
    result = checks[provider]()
    assert result.ok is True
    assert captured["base_url"] == _BASE_URL[provider]
    assert captured["api_key"] == _MOCK_KEY
