"""Vertex AI Gemini provider adapter — uses google.genai (native async)."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from time import perf_counter
from typing import Any

from ..core.exceptions import ProviderError, ProviderTimeoutError
from ..core.interfaces import BaseProvider
from ..core.types import ProviderChunk, ProviderRequest
from ..infra.logging import get_logger

try:  # pragma: no cover - import boundary
    import google.genai as genai
except Exception:  # pragma: no cover - import boundary
    genai = None  # type: ignore[assignment]


class VertexProvider(BaseProvider):
    """Vertex AI Gemini adapter using the native async google.genai SDK.

    Uses ``client.aio.models.generate_content_stream`` for fully async,
    non-blocking streaming — no thread pool, no queue, no stop_event required.
    """

    def __init__(
        self,
        project_id: str | None = None,
        region: str | None = None,
        credentials: Any | None = None,
        # Kept for backward compatibility and test injection only
        _client_factory: Any | None = None,
    ) -> None:
        """Initialize the Vertex provider.

        Args:
            project_id: GCP project ID. Reads ``GOOGLE_CLOUD_PROJECT`` env var if absent.
            region: GCP region (e.g. ``us-central1``). Reads ``GOOGLE_CLOUD_LOCATION`` if absent.
            credentials: Optional authentication object (service account dict or oauth2 credentials).
            _client_factory: Optional callable returning a mock client for testing.
        """
        if genai is None and _client_factory is None:
            raise ProviderError(
                "google-genai SDK is required for VertexProvider. "
                "Install with: pip install google-genai"
            )
        self._project_id = project_id
        self._region = region
        self._credentials = credentials
        self._client_factory = _client_factory

    def _get_client(
        self,
        project_id: str | None,
        region: str | None,
        credentials: Any | None = None,
    ) -> Any:
        if self._client_factory is not None:
            return self._client_factory(
                project_id=project_id, region=region, credentials=credentials
            )

        auth_obj = credentials or self._credentials
        # If we received a raw dict (e.g. from json.load), convert to a credentials object.
        if isinstance(auth_obj, dict):
            try:
                from google.oauth2 import service_account
                # Vertex AI requires at least the cloud-platform scope.
                scopes = ["https://www.googleapis.com/auth/cloud-platform"]
                auth_obj = service_account.Credentials.from_service_account_info(
                    auth_obj, scopes=scopes
                )
            except ImportError:
                raise ProviderError(
                    "google-auth is required to use dictionary credentials. "
                    "Install with: pip install google-auth"
                )

        return genai.Client(  # type: ignore[union-attr]
            vertexai=True,
            project=project_id,
            location=region,
            credentials=auth_obj,
        )

    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        """Execute and stream a Vertex AI request via the native async SDK."""
        logger = get_logger()
        started = perf_counter()

        # D1: read from typed RuntimeHints, not magic metadata keys
        request_timeout = request.hints.request_timeout
        first_token_timeout = request.hints.first_token_timeout
        raw_stream: Any | None = None

        # Per-request project/region/credentials override via hints
        project_id = request.hints.project_id or self._project_id
        region = request.hints.region or self._region
        credentials = request.hints.credentials or self._credentials

        try:
            client = self._get_client(project_id, region, credentials)
        except Exception as exc:
            raise ProviderError(f"Vertex client setup failed: {exc}") from exc

        tools_config = None
        if request.tools:
            tool_declarations = [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "object",
                        "properties": t.parameters,
                        "required": t.required,
                    },
                }
                for t in request.tools
            ]
            tools_config = [{"function_declarations": tool_declarations}]

        # D5: structured multi-turn contents + system_instruction
        system_instruction, contents = _to_contents(request)

        config: dict[str, Any] = {"temperature": request.temperature}
        if request.max_tokens is not None:
            config["max_output_tokens"] = request.max_tokens
        if system_instruction:
            config["system_instruction"] = system_instruction
        if tools_config:
            config["tools"] = tools_config

        try:
            raw_stream = await asyncio.wait_for(
                client.aio.models.generate_content_stream(
                    model=request.model,
                    contents=contents,
                    config=config,
                ),
                timeout=request_timeout,
            )
            async for event in _iter_with_timeouts(raw_stream, first_token_timeout, request_timeout):
                text = _extract_text(event)
                if text:
                    yield ProviderChunk(text=text, is_final=False, raw=event)

                for fc in _extract_function_calls(event):
                    yield ProviderChunk(
                        tool_call_id=f"call_{fc['name']}_{uuid.uuid4().hex[:8]}",
                        tool_name=fc["name"],
                        tool_arguments=fc["args_json"],
                        raw=event,
                    )

        except asyncio.CancelledError:
            if raw_stream is not None:
                await _aclose_stream(raw_stream)
            raise
        except ProviderTimeoutError:
            if raw_stream is not None:
                await _aclose_stream(raw_stream)
            logger.exception(
                "provider_timeout provider=vertex model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
            raise
        except Exception as exc:
            if raw_stream is not None:
                await _aclose_stream(raw_stream)
            logger.exception(
                "provider_error provider=vertex model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
            raise ProviderError(f"Vertex streaming failed: {exc}") from exc
        else:
            logger.info(
                "provider_complete provider=vertex model=%s request_id=%s latency_ms=%.2f",
                request.model,
                request.hints.request_id,
                (perf_counter() - started) * 1000.0,
            )
        finally:
            if raw_stream is not None:
                await _aclose_stream(raw_stream)


async def _iter_with_timeouts(
    stream: Any,
    first_token_timeout: float,
    request_timeout: float,
) -> AsyncIterator[Any]:
    """Enforce first-token and per-event timeouts on any async iterable."""
    received_first = False
    aiter = stream.__aiter__()
    while True:
        timeout = first_token_timeout if not received_first else request_timeout
        try:
            event = await asyncio.wait_for(aiter.__anext__(), timeout=timeout)
        except StopAsyncIteration:
            return
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError) as exc:
            stage = "first token" if not received_first else "stream iteration"
            raise ProviderTimeoutError(
                f"Vertex {stage} timed out after {timeout:.2f}s"
            ) from exc
        received_first = True
        yield event


def _to_contents(request: ProviderRequest) -> tuple[str, list[dict[str, Any]]]:
    """D5: Convert structured messages to Gemini contents list + system_instruction."""
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []

    for msg in request.messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue

        role = "model" if msg.role == "assistant" else "user"
        parts: list[dict[str, Any]] = []

        if msg.content:
            parts.append({"text": msg.content})

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.get("function", {})
                raw_args = fn.get("arguments", "{}")
                try:
                    args_dict = json.loads(raw_args) if raw_args else {}
                except ValueError:
                    args_dict = {}
                parts.append({
                    "function_call": {
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "args": args_dict,
                    }
                })

        if msg.role == "tool" and msg.content:
            parts.append({
                "function_response": {
                    "id": msg.tool_call_id or "",
                    "name": msg.name or "",
                    "response": {"output": msg.content},
                }
            })

        if parts:
            contents.append({"role": role, "parts": parts})

    return "\n".join(system_parts), contents


def _extract_text(event: Any) -> str:
    # Manual extraction from candidates suppresses the SDK's built-in warning.
    candidates = getattr(event, "candidates", None)
    if not candidates:
        # Fallback for simple mocks/tests that don't have candidates
        return getattr(event, "text", "")

    parts_text: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            part_text = getattr(part, "text", None)
            if part_text:
                parts_text.append(str(part_text))
    return "".join(parts_text)


def _extract_function_calls(event: Any) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    candidates = getattr(event, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if not parts:
            continue
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            name = getattr(fc, "name", "")
            args: dict[str, Any] = {}
            if hasattr(fc, "args"):
                for k, v in fc.args.items():
                    args[k] = v
            results.append({"name": name, "args_json": json.dumps(args)})
    return results


async def _aclose_stream(stream: Any) -> None:
    aclose = getattr(stream, "aclose", None)
    if callable(aclose):
        try:
            await aclose()
            return
        except Exception:
            pass

    close = getattr(stream, "close", None)
    if callable(close):
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception:
            pass
