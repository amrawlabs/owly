"""Cancellation primitives for runtime stream execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import TypeVar

from ..core.exceptions import CancellationError
from ..utils.async_utils import aclose_safely

T = TypeVar("T")


async def cancellable_stream(stream: AsyncIterator[T]) -> AsyncIterator[T]:
    """Wrap a stream and convert task cancellation into ``CancellationError``.

    On cancellation this closes upstream immediately to prevent token leakage.
    """

    try:
        async for item in stream:
            yield item
    except asyncio.CancelledError as exc:
        await aclose_safely(stream)
        raise CancellationError("Stream cancelled") from exc
    finally:
        await aclose_safely(stream)
