"""Provider adapter package."""

__all__ = ["BaseProvider", "OpenAIProvider", "GeminiProvider"]

from .base import BaseProvider


def __getattr__(name: str):
    if name == "OpenAIProvider":
        from .openai import OpenAIProvider

        return OpenAIProvider
    if name == "GeminiProvider":
        from .gemini import GeminiProvider

        return GeminiProvider
    raise AttributeError(name)
