"""LLM runtime entrypoint.

This module provides the primary user-facing class `LLM` for streaming
and generating completions from various providers.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from .core.exceptions import ConfigurationError, ProviderError
from .core.interfaces import BaseProvider
from .core.types import Chunk, LLMRequest, Message, ProviderRequest, RuntimeHints, ToolCallChunk
from .infra.config import OwlyConfig
from .runtime.stream_pipeline import run_stream_pipeline


class LLM:
    """Streaming-first LLM runtime client."""

    def __init__(
        self,
        provider: str | BaseProvider,
        model: str,
        api_key: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        credentials: Any | None = None,
        config: OwlyConfig | None = None,
    ) -> None:
        """Initialize the LLM.

        Args:
            provider: Provider name (e.g. "openai", "gemini", "vertex").
            model: Model name.
            api_key: Optional API key.
            project_id: Optional GCP project ID (for Vertex).
            region: Optional GCP region (for Vertex).
            credentials: Optional authentication object (for Vertex).
            config: Optional global runtime configuration.
        """
        self.config = config or OwlyConfig()
        self.model = model
        self._project_id = project_id
        self._region = region
        self._credentials = credentials
        self._stream_semaphore = asyncio.Semaphore(self.config.max_concurrency)

        if isinstance(provider, str):
            self.provider = self._resolve_provider(
                provider, api_key, project_id, region, credentials
            )
        elif isinstance(provider, BaseProvider):
            self.provider = provider
        else:
            raise ConfigurationError("provider must be a provider name or BaseProvider")

    @staticmethod
    def _resolve_provider(
        provider: str,
        api_key: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        credentials: Any | None = None,
    ) -> BaseProvider:
        key = provider.strip().lower()
        if key == "openai":
            from .providers.openai import OpenAIProvider
            return OpenAIProvider(api_key=api_key)
        if key == "gemini":
            from .providers.gemini import GeminiProvider
            return GeminiProvider(api_key=api_key)
        if key == "vertex":
            from .providers.vertex import VertexProvider
            return VertexProvider(project_id=project_id, region=region, credentials=credentials)
        if key == "claude":
            from .providers.claude import ClaudeProvider
            return ClaudeProvider(api_key=api_key)
        raise ProviderError(f"Unknown provider '{provider}'")

    async def stream(self, request: LLMRequest) -> AsyncIterator[Chunk | ToolCallChunk]:
        # Build typed RuntimeHints from config — never touch LLMRequest.metadata
        hints_kwargs: dict[str, Any] = {
            "request_timeout": self.config.request_timeout,
            "first_token_timeout": self.config.first_token_timeout,
            "queue_maxsize": self.config.queue_maxsize,
            "project_id": self._project_id,
            "region": self._region,
            "credentials": self._credentials,
        }
        if request.request_id:
            hints_kwargs["request_id"] = request.request_id

        hints = RuntimeHints(**hints_kwargs)

        provider_request = ProviderRequest(
            model=self.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            metadata=request.metadata,  # forwarded verbatim; Owly does not write here
            hints=hints,
            request_id=request.request_id,
        )
        async for chunk in run_stream_pipeline(
            self.provider,
            provider_request,
            self.config,
            semaphore=self._stream_semaphore,
        ):
            yield chunk

    async def generate(self, request: LLMRequest) -> Message:
        content = ""
        tool_calls: dict[str, dict[str, str]] = {}
        async for chunk in self.stream(request):
            if isinstance(chunk, ToolCallChunk):
                if chunk.tool_call_id not in tool_calls:
                    tool_calls[chunk.tool_call_id] = {"name": chunk.name or "", "arguments": ""}
                tool_calls[chunk.tool_call_id]["arguments"] += chunk.arguments
            elif chunk.text:
                content += chunk.text

        tool_payloads = []
        if tool_calls:
            for call_id, call_data in tool_calls.items():
                tool_payloads.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": call_data["name"],
                            "arguments": call_data["arguments"],
                        },
                    }
                )

        return Message(
            role="assistant",
            content=content if content else None,
            tool_calls=tool_payloads if tool_payloads else None,
        )

    def generate_sync(self, request: LLMRequest) -> Message:
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.generate(request))).result()

        return asyncio.run(self.generate(request))
