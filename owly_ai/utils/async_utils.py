"""Async utility helpers used across runtime components."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any


async def aclose_safely(iterator: AsyncIterator[Any]) -> None:
    """Best-effort close of async generators/iterators."""

    aclose = getattr(iterator, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
        except Exception:
            pass


def split_text_realtime(text: str, target_chars: int, max_chars: int) -> Iterator[str]:
    """Split text into predictable small pieces for smooth token rendering."""

    if len(text) <= max_chars:
        yield text
        return

    remaining = text
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, target_chars + 1)
        if split_at <= 0:
            split_at = target_chars
        piece = remaining[:split_at]
        if piece:
            yield piece
        remaining = remaining[split_at:]

    if remaining:
        yield remaining
