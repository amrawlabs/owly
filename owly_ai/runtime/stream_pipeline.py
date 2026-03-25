"""Runtime stream pipeline orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..core.interfaces import BaseProvider
from ..core.types import Chunk, ProviderRequest
from ..infra.config import OwlyConfig
from .cancellation import cancellable_stream
from .normalizer import normalize_stream


async def run_stream_pipeline(
    provider: BaseProvider,
    request: ProviderRequest,
    config: OwlyConfig,
) -> AsyncIterator[Chunk]:
    """Execute provider -> cancellable -> normalized pipeline."""

    raw_stream = provider.stream(request)
    safe_stream = cancellable_stream(raw_stream)
    async for chunk in normalize_stream(safe_stream, config.stream):
        yield chunk
