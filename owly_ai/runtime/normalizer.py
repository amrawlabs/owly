"""Provider stream normalization for deterministic real-time output."""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..core.types import Chunk, ProviderChunk, ToolCallChunk
from ..infra.config import StreamConfig
from ..utils.async_utils import split_text_realtime


async def normalize_stream(
    stream: AsyncIterator[ProviderChunk],
    config: StreamConfig,
) -> AsyncIterator[Chunk | ToolCallChunk]:
    """Normalize provider chunks with low overhead and predictable cadence.

    This function acts as a buffer and router that:
    1. Consolidates or splits text chunks based on target sizes in StreamConfig.
    2. Detects and yields ToolCallChunk objects as they appear.
    3. Handles the conversion of ProviderChunk into finalized user-facing Chunk objects.

    Fixes applied:
    - P3: Carry buffer uses a list + join instead of O(n²) string concatenation.
    - M1: If the stream produces no text (e.g. tool-only), a final Chunk is still emitted.
    - M2: ToolCallChunk items are tracked so completion can be signalled.

    Args:
        stream: The raw async iterator from a provider adapter.
        config: Configuration defining min/max chunk sizes for normalization.

    Yields:
        Normalized Chunk or ToolCallChunk objects.
    """

    carry_buf: list[str] = []
    last_chunk: Chunk | None = None
    had_any_output = False

    async for provider_chunk in stream:
        if provider_chunk.tool_call_id is not None or provider_chunk.tool_arguments is not None:
            # Flush any carried text before yielding a tool call
            if last_chunk is not None:
                yield last_chunk
                last_chunk = None
            if carry_buf:
                yield Chunk(text="".join(carry_buf), is_final=False)
                carry_buf = []
            had_any_output = True
            yield ToolCallChunk(
                tool_call_id=provider_chunk.tool_call_id or "",
                name=provider_chunk.tool_name,
                arguments=provider_chunk.tool_arguments or "",
                is_final=False,
            )
            continue

        text = provider_chunk.text
        if not text:
            continue

        carry_buf.append(text)
        current_len = sum(len(x) for x in carry_buf)

        if current_len >= config.min_chunk_chars:
            merged = "".join(carry_buf)
            carry_buf = []

            if len(merged) > config.max_chunk_chars:
                for piece in split_text_realtime(merged, config.target_chunk_chars, config.max_chunk_chars):
                    if last_chunk is not None:
                        yield last_chunk
                    last_chunk = Chunk(text=piece, is_final=False)
            else:
                if last_chunk is not None:
                    yield last_chunk
                last_chunk = Chunk(text=merged, is_final=False)
                
            had_any_output = True

    # Flush carry buffer
    if carry_buf:
        merged = "".join(carry_buf)
        if last_chunk is not None:
            yield last_chunk
        last_chunk = Chunk(text=merged, is_final=False)
        had_any_output = True

    # Mark the very last text chunk as final — no empty sentinel emitted.
    # M1 fix: if we only emitted tool calls (no text), emit an empty final Chunk
    # so consumers always receive exactly one is_final=True signal.
    if last_chunk is not None:
        yield Chunk(text=last_chunk.text, is_final=True)
    elif had_any_output:
        # Tool-call-only stream: emit empty final Chunk so consumer knows stream ended.
        yield Chunk(text="", is_final=True)
