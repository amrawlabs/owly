from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from owly_ai.core.exceptions import CancellationError
from owly_ai.runtime.cancellation import cancellable_stream


async def _slow_stream() -> AsyncIterator[str]:
    while True:
        await asyncio.sleep(0.05)
        yield "x"


@pytest.mark.asyncio
async def test_cancellable_stream_raises_cancellation_error() -> None:
    async def consume() -> None:
        async for _ in cancellable_stream(_slow_stream()):
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.12)
    task.cancel()

    with pytest.raises(CancellationError):
        await task
