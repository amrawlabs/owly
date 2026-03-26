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

    On cancellation this closes upstream **once** and guarantees no further yields.
    The ``aclose`` is issued only in the ``CancelledError`` handler to avoid
    double-closing the generator (which can raise ``RuntimeError`` on some
    async generators).
    """

    try:
        async for item in stream:
            yield item
    except asyncio.CancelledError as exc:
        await aclose_safely(stream)
        raise CancellationError("Stream cancelled") from exc
