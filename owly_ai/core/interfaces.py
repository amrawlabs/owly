"""Core runtime interfaces for providers and pipeline components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

from .types import ProviderChunk, ProviderRequest


class BaseProvider(ABC):
    """Contract for provider adapters used by Owly runtime."""

    @abstractmethod
    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        """Yield provider chunks with minimal latency and no buffering."""
