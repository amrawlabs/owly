from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from owly.core.types import ProviderChunk
from owly.infra.config import StreamConfig
from owly.runtime.normalizer import normalize_stream


async def _provider_stream(chunks: list[ProviderChunk]) -> AsyncIterator[ProviderChunk]:
    for chunk in chunks:
        yield chunk


@pytest.mark.asyncio
async def test_normalize_stream_filters_and_splits() -> None:
    chunks = [
        ProviderChunk(text=""),
        ProviderChunk(text="hi"),
        ProviderChunk(text=" there"),
        ProviderChunk(text=" " + "x" * 90),
    ]
    config = StreamConfig(min_chunk_chars=3, target_chunk_chars=10, max_chunk_chars=20)

    out = []
    async for chunk in normalize_stream(_provider_stream(chunks), config):
        out.append(chunk)

    assert out
    assert out[-1].is_final is True
    assert all(c.text or c.is_final for c in out)
    assert any(len(c.text) <= 20 for c in out if c.text)
