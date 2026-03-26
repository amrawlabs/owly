"""Runtime stream pipeline orchestration."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from time import perf_counter

from ..core.interfaces import BaseProvider
from ..core.types import Chunk, ProviderRequest
from ..infra.config import OwlyConfig
from ..infra.logging import get_logger
from .cancellation import cancellable_stream
from .normalizer import normalize_stream


async def run_stream_pipeline(
    provider: BaseProvider,
    request: ProviderRequest,
    config: OwlyConfig,
    semaphore: asyncio.Semaphore | None = None,
) -> AsyncIterator[Chunk]:
    """Execute provider -> cancellable -> normalized pipeline.

    The semaphore (if provided) gates how many streams are active concurrently.
    It is held for the full lifetime of the stream and released when the stream
    terminates.
    """

    logger = get_logger(config.logger_name)
    provider_name = provider.__class__.__name__
    req_id = request.request_id or "unknown"
    started = perf_counter()
    ttft_ms: float | None = None

    logger.info(
        "stream_pipeline_started provider=%s model=%s request_id=%s",
        provider_name,
        request.model,
        req_id,
    )

    async def _execute() -> AsyncIterator[Chunk]:
        nonlocal ttft_ms
        raw_stream = provider.stream(request)
        safe_stream = cancellable_stream(raw_stream)
        async for chunk in normalize_stream(safe_stream, config.stream):
            if ttft_ms is None:
                ttft_ms = (perf_counter() - started) * 1000.0
            yield chunk

    try:
        if semaphore is not None:
            await semaphore.acquire()
        try:
            async for chunk in _execute():
                yield chunk
        finally:
            if semaphore is not None:
                semaphore.release()
    except Exception:
        logger.exception(
            "stream_pipeline_error provider=%s model=%s request_id=%s latency_ms=%.2f ttft_ms=%s",
            provider_name,
            request.model,
            req_id,
            (perf_counter() - started) * 1000.0,
            f"{ttft_ms:.2f}" if ttft_ms is not None else "N/A",
        )
        raise
    else:
        logger.info(
            "stream_pipeline_complete provider=%s model=%s request_id=%s latency_ms=%.2f ttft_ms=%s",
            provider_name,
            request.model,
            req_id,
            (perf_counter() - started) * 1000.0,
            f"{ttft_ms:.2f}" if ttft_ms is not None else "N/A",
        )
