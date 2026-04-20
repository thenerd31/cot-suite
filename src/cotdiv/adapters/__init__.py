"""Adapters from provider-native message formats into cotdiv.Trajectory."""

from cotdiv.adapters.anthropic_messages import from_anthropic
from cotdiv.adapters.openai_messages import from_openai

__all__ = ["from_anthropic", "from_openai"]
