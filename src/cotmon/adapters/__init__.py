"""Adapters from provider-native message formats into cotmon.Trajectory."""

from cotmon.adapters.anthropic_messages import from_anthropic
from cotmon.adapters.openai_messages import from_openai

__all__ = ["from_anthropic", "from_openai"]
