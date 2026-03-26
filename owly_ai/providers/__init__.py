"""Provider adapter package."""

__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "GeminiProvider",
    "VertexProvider",
    "ClaudeProvider",
]

from .base import BaseProvider


def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider

        return GeminiProvider
    if name == "VertexProvider":
        from .vertex import VertexProvider

        return VertexProvider
    if name == "ClaudeProvider":
        from .claude import ClaudeProvider

        return ClaudeProvider
    raise AttributeError(name)
