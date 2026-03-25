"""Runtime configuration for owly-ai."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class StreamConfig:
    """Controls runtime normalization behavior."""

    min_chunk_chars: int = 8
    target_chunk_chars: int = 24
    max_chunk_chars: int = 48


@dataclass(slots=True, frozen=True)
class OwlyConfig:
    """Top-level runtime configuration."""

    stream: StreamConfig = StreamConfig()
    logger_name: str = "owly-ai"
