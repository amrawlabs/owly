"""Gemini provider adapter."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import google.generativeai as genai

from ..core.exceptions import ProviderError
from ..core.interfaces import BaseProvider
from ..core.types import ProviderChunk, ProviderRequest


class GeminiProvider(BaseProvider):
    """Gemini adapter with smooth async streaming output.
    
    Handles communication with Google's Generative AI API, bridging
    Gemini's thread-based streaming to Owly's async pipeline.
    """

    def __init__(self, api_key: str | None = None) -> None:
        """Initialize the Gemini provider.
        
        Args:
            api_key: Optional API key. Required if not configured globally.
        """
        if api_key:
            genai.configure(api_key=api_key)

    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        """Execute and stream a Gemini request.
        
        Translates messages and tool definitions into Gemini-compatible structures.
        """
        gemini_tools = None
        if request.tools:
            gemini_tools = []
            for t in request.tools:
                gemini_tools.append({
                    "function_declarations": [{
                        "name": t.name,
                        "description": t.description,
                        "parameters": {
                            "type": "object",
                            "properties": t.parameters,
                            "required": t.required,
                        }
                    }]
                })
        model = genai.GenerativeModel(model_name=request.model, tools=gemini_tools)
        queue: asyncio.Queue[ProviderChunk | Exception | object] = asyncio.Queue()
        sentinel = object()
        loop = asyncio.get_running_loop()

        def push(item: ProviderChunk | Exception | object) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def producer() -> None:
            try:
                stream = model.generate_content(
                    self._to_prompt(request),
                    stream=True,
                    generation_config={
                        "temperature": request.temperature,
                        **(
                            {"max_output_tokens": request.max_tokens}
                            if request.max_tokens is not None
                            else {}
                        ),
                    },
                )
                for event in stream:
                    candidates = getattr(event, "candidates", [])
                    for candidate in candidates:
                        content = getattr(candidate, "content", None)
                        parts = getattr(content, "parts", []) if content else []
                        for part in parts:
                            part_text = getattr(part, "text", None)
                            if part_text:
                                push(ProviderChunk(text=str(part_text), is_final=False, raw=event))
                            
                            function_call = getattr(part, "function_call", None)
                            if function_call:
                                import json
                                name = getattr(function_call, "name", "")
                                args_dict = {}
                                if hasattr(function_call, "args"):
                                    for k, v in function_call.args.items():
                                        args_dict[k] = v
                                args_str = json.dumps(args_dict)
                                push(ProviderChunk(
                                    tool_call_id=f"call_{name}_{uuid.uuid4().hex[:8]}",
                                    tool_name=name,
                                    tool_arguments=args_str,
                                    raw=event
                                ))
            except Exception as exc:  # pragma: no cover - provider boundary
                push(exc)
            finally:
                push(sentinel)

        thread_task = asyncio.create_task(asyncio.to_thread(producer))
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise ProviderError(f"Gemini streaming failed: {item}") from item
                yield item
        finally:
            await thread_task

    @staticmethod
    def _to_prompt(request: ProviderRequest) -> str:
        lines = []
        for message in request.messages:
            if message.content:
                lines.append(f"{message.role}: {message.content}")
            if message.tool_calls:
                lines.append(f"{message.role} evaluated tools: {[tc.get('function', {}).get('name') for tc in message.tool_calls]}")
        return "\n".join(lines)
