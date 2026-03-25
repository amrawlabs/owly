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
    
    Args:
        stream: The raw async iterator from a provider adapter.
        config: Configuration defining min/max chunk sizes for normalization.
        
    Yields:
        Normalized Chunk or ToolCallChunk objects.
    """

    carry = ""
    last_chunk: Chunk | None = None

    async for provider_chunk in stream:
        if provider_chunk.tool_call_id is not None or provider_chunk.tool_arguments is not None:
            # Flush any carried text before yielding a tool call
            if last_chunk is not None:
                yield last_chunk
                last_chunk = None
            if carry:
                yield Chunk(text=carry, is_final=False)
                carry = ""
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

        if carry:
            merged = carry + text
            if len(merged) <= config.target_chunk_chars:
                # Emit the previously held chunk before buffering merged
                if last_chunk is not None:
                    yield last_chunk
                last_chunk = Chunk(text=merged, is_final=False)
                carry = ""
                continue
            if last_chunk is not None:
                yield last_chunk
            last_chunk = Chunk(text=carry, is_final=False)
            carry = ""

        if len(text) > config.max_chunk_chars:
            for piece in split_text_realtime(text, config.target_chunk_chars, config.max_chunk_chars):
                if last_chunk is not None:
                    yield last_chunk
                last_chunk = Chunk(text=piece, is_final=False)
            continue

        if len(text) < config.min_chunk_chars:
            carry = text
            continue

        if last_chunk is not None:
            yield last_chunk
        last_chunk = Chunk(text=text, is_final=False)

    # Flush carry buffer
    if carry:
        if last_chunk is not None:
            yield last_chunk
        last_chunk = Chunk(text=carry, is_final=False)

    # Mark the very last text chunk as final — no empty sentinel emitted
    if last_chunk is not None:
        yield Chunk(text=last_chunk.text, is_final=True)
