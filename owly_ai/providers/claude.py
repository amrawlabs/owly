"""Anthropic Claude provider adapter."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from time import perf_counter
from typing import Any

from ..core.exceptions import ProviderError, ProviderTimeoutError
from ..core.interfaces import BaseProvider
from ..core.types import ProviderChunk, ProviderRequest
from ..infra.logging import get_logger

try:  # pragma: no cover - import boundary
    from anthropic import AsyncAnthropic
except Exception:  # pragma: no cover - import boundary
    AsyncAnthropic = None  # type: ignore[assignment]


class ClaudeProvider(BaseProvider):
    """Anthropic Claude adapter with async streaming and timeout controls."""

    def __init__(self, client: Any | None = None, api_key: str | None = None) -> None:
        if client is not None:
            self._client = client
            return
        if AsyncAnthropic is None:
            raise ProviderError("anthropic SDK is required for ClaudeProvider")
        self._client = AsyncAnthropic(api_key=api_key) if api_key else AsyncAnthropic()

    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        logger = get_logger()
        started = perf_counter()

        # connect_timeout: how long to wait for the stream object to be created.
        # first_token_timeout: how long to wait for the very first event.
        # request_timeout: per-event timeout after the first token arrives.
        connect_timeout = request.hints.request_timeout
        request_timeout = request.hints.request_timeout
        first_token_timeout = request.hints.first_token_timeout

        messages = []
        for message in request.messages:
            if message.role in {"user", "assistant"} and message.content:
                messages.append({"role": message.role, "content": message.content})

        params: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens or 1024,
        }

        stream_obj: Any | None = None
        got_first = False

        try:
            try:
                stream_obj = await asyncio.wait_for(
                    self._client.messages.create(**params),
                    timeout=connect_timeout,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                raise ProviderTimeoutError(
                    f"Claude connect timed out after {connect_timeout:.2f}s"
                ) from exc

            iterator = stream_obj.__aiter__()
            while True:
                timeout = first_token_timeout if not got_first else request_timeout
                try:
                    event = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
                except StopAsyncIteration:
                    break
                except asyncio.CancelledError:
                    raise
                except TimeoutError as exc:
                    stage = "first token" if not got_first else "stream iteration"
                    raise ProviderTimeoutError(
                        f"Claude {stage} timed out after {timeout:.2f}s"
                    ) from exc

                text = _extract_text_delta(event)
                if text:
                    got_first = True
                    yield ProviderChunk(text=text, is_final=False)

        except asyncio.CancelledError:
            if stream_obj is not None:
                await _aclose_stream(stream_obj)
            raise
        except ProviderTimeoutError:
            if stream_obj is not None:
                await _aclose_stream(stream_obj)
            logger.exception(
                "provider_timeout provider=claude model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
            raise
        except Exception as exc:
            if stream_obj is not None:
                await _aclose_stream(stream_obj)
            logger.exception(
                "provider_error provider=claude model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
            raise ProviderError(f"Claude streaming failed: {exc}") from exc
        else:
            # Only log success when no exception was raised (C4 fix)
            logger.info(
                "provider_complete provider=claude model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
        finally:
            # Always attempt stream cleanup on any exit (including CancelledError)
            if stream_obj is not None:
                await _aclose_stream(stream_obj)


def _extract_text_delta(event: Any) -> str:
    event_type = getattr(event, "type", "")

    if isinstance(event, dict):
        event_type = event.get("type", "")
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if isinstance(delta, dict):
                return str(delta.get("text", "") or "")
        return ""

    if event_type == "content_block_delta":
        delta = getattr(event, "delta", None)
        if delta is None:
            return ""
        text = getattr(delta, "text", None)
        if text:
            return str(text)

    delta = getattr(event, "delta", None)
    text = getattr(delta, "text", None) if delta is not None else None
    if text:
        return str(text)

    return ""


async def _aclose_stream(stream_obj: Any) -> None:
    aclose = getattr(stream_obj, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
        except Exception:
            pass

