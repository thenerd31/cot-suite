"""Adapters from provider-native message formats into cotsuite.Trajectory."""

from cotsuite.adapters.anthropic_messages import from_anthropic
from cotsuite.adapters.openai_messages import from_openai

__all__ = ["from_anthropic", "from_openai"]
