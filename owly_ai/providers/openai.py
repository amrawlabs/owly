"""OpenAI provider adapter."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from openai import AsyncOpenAI

from ..core.exceptions import ProviderError
from ..core.interfaces import BaseProvider
from ..core.types import ProviderChunk, ProviderRequest


class OpenAIProvider(BaseProvider):
    """OpenAI adapter with immediate token emission semantics.
    
    Handles communication with OpenAI's API, including tool/function calling support.
    """

    def __init__(self, client: AsyncOpenAI | None = None, api_key: str | None = None) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            client: Optional pre-configured AsyncOpenAI client.
            api_key: Optional API key. Uses environment variable if not provided.
        """
        if client:
            self._client = client
        elif api_key:
            self._client = AsyncOpenAI(api_key=api_key)
        else:
            # Let the SDK pick up OPENAI_API_KEY from the environment
            self._client = AsyncOpenAI()

    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        """Convert ProviderRequest to OpenAI params and stream responses.
        
        Maps Owly tool definitions into OpenAI's function calling format.
        """
        try:
            messages = []
            for message in request.messages:
                msg_dict = {"role": message.role}
                if message.content is not None:
                    msg_dict["content"] = message.content
                if message.tool_calls:
                    msg_dict["tool_calls"] = message.tool_calls
                if message.tool_call_id:
                    msg_dict["tool_call_id"] = message.tool_call_id
                if message.name:
                    msg_dict["name"] = message.name
                messages.append(msg_dict)
            params: dict[str, Any] = {
                "model": request.model,
                "messages": messages,
                "stream": True,
                "temperature": request.temperature,
            }
            if request.max_tokens is not None:
                params["max_tokens"] = request.max_tokens
            if request.tools:
                openai_tools = []
                for t in request.tools:
                    openai_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": {
                                "type": "object",
                                "properties": t.parameters,
                                "required": t.required,
                                "additionalProperties": False,
                            },
                        "strict": True,
                        }
                    })
                params["tools"] = openai_tools

            # Buffer for tool call deltas keyed by index.
            # OpenAI only sends id and name on the first delta for each index;
            # subsequent argument deltas have both as None. We must accumulate
            # across the full stream before emitting.
            tool_call_buffer: dict[int, dict[str, str]] = {}

            stream = await self._client.chat.completions.create(**params)
            async for event in stream:
                choice = event.choices[0] if event.choices else None
                delta = choice.delta if choice else None
                if delta is None:
                    continue

                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        idx = tool_call.index
                        if idx not in tool_call_buffer:
                            tool_call_buffer[idx] = {"id": "", "name": "", "arguments": ""}
                        if tool_call.id:
                            tool_call_buffer[idx]["id"] = tool_call.id
                        if tool_call.function and tool_call.function.name:
                            tool_call_buffer[idx]["name"] = tool_call.function.name
                        if tool_call.function and tool_call.function.arguments:
                            tool_call_buffer[idx]["arguments"] += tool_call.function.arguments
                    continue

                content = delta.content
                if not content:
                    continue

                if isinstance(content, str):
                    yield ProviderChunk(text=content, is_final=False, raw=event)
                    continue

                if isinstance(content, list):
                    for part in content:
                        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if text:
                            yield ProviderChunk(text=str(text), is_final=False, raw=event)

            # Emit one clean ProviderChunk per buffered tool call
            for tc in tool_call_buffer.values():
                yield ProviderChunk(
                    tool_call_id=tc["id"],
                    tool_name=tc["name"],
                    tool_arguments=tc["arguments"],
                )
        except Exception as exc:  # pragma: no cover - provider boundary
            raise ProviderError(f"OpenAI streaming failed: {exc}") from exc
