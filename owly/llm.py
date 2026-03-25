"""LLM runtime entrypoint.

This module provides the primary user-facing class `LLM` for streaming
and generating completions from various providers.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from .core.exceptions import ConfigurationError, ProviderError
from .core.interfaces import BaseProvider
from .core.types import Chunk, LLMRequest, Message, ProviderRequest, ToolCallChunk
from .infra.config import OwlyConfig
from .runtime.stream_pipeline import run_stream_pipeline


class LLM:
    """Streaming-first LLM runtime client.
    
    This class handles provider resolution, request normalization, and 
    orchestrates the streaming pipeline.
    """

    def __init__(
        self,
        provider: str | BaseProvider,
        model: str,
        config: OwlyConfig | None = None,
        api_key: str | None = None,
    ) -> None:
        """Initialize the LLM client.
        
        Args:
            provider: Either a provider name string (e.g., "openai", "gemini")
                     or an instance of a class implementing BaseProvider.
            model: The name of the model to use (e.g., "gpt-4o", "gemini-1.5-flash").
            config: Optional runtime configuration for streaming.
            api_key: Optional API key for the provider. If not provided, the 
                    provider adapter will attempt to use environment variables.
            
        Raises:
            ConfigurationError: If provider type is invalid.
        """
        if isinstance(provider, str):
            self.provider = self._resolve_provider(provider, api_key)
        elif isinstance(provider, BaseProvider):
            self.provider = provider
        else:
            raise ConfigurationError("provider must be a provider name or BaseProvider")
        self.model = model
        self.config = config or OwlyConfig()

    @staticmethod
    def _resolve_provider(provider: str, api_key: str | None = None) -> BaseProvider:
        """Internal helper to instantiate provider adapters by name.
        
        Args:
            provider: The name of the provider.
            api_key: Optional API key to pass to the provider constructor.
            
        Returns:
            An instance of the resolved BaseProvider.
            
        Raises:
            ProviderError: If the provider name is unknown.
        """
        key = provider.strip().lower()
        if key == "openai":
            from .providers.openai import OpenAIProvider

            return OpenAIProvider(api_key=api_key)
        if key == "gemini":
            from .providers.gemini import GeminiProvider

            return GeminiProvider(api_key=api_key)
        raise ProviderError(f"Unknown provider '{provider}'")

    async def stream(self, request: LLMRequest) -> AsyncIterator[Chunk | ToolCallChunk]:
        """Stream normalized chunks from the configured provider.

        Pipeline: provider -> cancellable -> normalized -> user.
        
        Args:
            request: The LLMRequest containing messages and parameters.
            
        Yields:
            Chunk or ToolCallChunk objects as they arrive from the provider.
        """

        provider_request = ProviderRequest(
            model=self.model,
            messages=request.messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools,
            metadata=request.metadata,
        )
        async for chunk in run_stream_pipeline(self.provider, provider_request, self.config):
            yield chunk

    async def generate(self, request: LLMRequest) -> Message:
        """Generate a complete message completion by buffering the stream.
        
        This method is a helper for non-streaming use cases. It aggregates all
        text and tool call chunks into a single Message object.
        
        Args:
            request: The LLMRequest to execute.
            
        Returns:
            A Message object with the assistant role and accumulated content/tool_calls.
        """
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
                tool_payloads.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                    }
                })
                
        return Message(
            role="assistant", 
            content=content if content else None, 
            tool_calls=tool_payloads if tool_payloads else None
        )

    def generate_sync(self, request: LLMRequest) -> Message:
        """Synchronous wrapper around generate().
        
        Safe to call from any context including inside running event loops.
        
        Args:
            request: The LLMRequest to execute.
            
        Returns:
            A Message object representing the completion.
        """
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, self.generate(request)).result()

        return asyncio.run(self.generate(request))
