"""Lightweight agent orchestrator.

Defines the Agent class which manages the loop between LLM streaming, 
tool execution, and conversation memory.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator

from .core.types import Chunk, LLMRequest, Message, ToolCallChunk
from .llm import LLM
from .memory import InMemoryHistory, Memory
from .tools import Tool

# Maximum number of tool call roundtrips per agent turn.
# Prevents infinite loops if the LLM gets stuck in a tool-call cycle.
_MAX_TOOL_ROUNDS = 10


class Agent:
    """An agent orchestrating LLM streaming, memory, and synchronous/async tool execution.
    
    The Agent wraps an LLM instance and provides a simplified interface for 
    multi-turn interactions that can involve tool usage.
    """

    def __init__(
        self,
        llm: LLM,
        memory: Memory | None = None,
        tools: list[Tool] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """Initialize the Agent.
        
        Args:
            llm: The LLM runtime instance to use.
            memory: Optional custom memory implementation. Defaults to InMemoryHistory.
            tools: Optional list of Tool instances available to the agent.
            system_prompt: Optional system prompt to initialize memory with.
        """
        self.llm = llm
        self.memory = memory if memory is not None else InMemoryHistory()
        self.tools = tools or []
        
        # Initialize memory with system prompt if provided and memory is empty
        if system_prompt:
            if not self.memory.get_messages():
                self.memory.add_message(Message(role="system", content=system_prompt))
        
        self._tool_map = {tool.name: tool for tool in self.tools}
        self._tool_defs = [tool.definition for tool in self.tools]

    async def stream(self, prompt: str) -> AsyncIterator[Chunk | ToolCallChunk]:
        """Stream an agent interaction round.
        
        This is a generator that yields chunks (text or tool calls) from the LLM.
        It automatically handles tool execution loops internally, re-prompting 
        the LLM with tool results until a final text response is produced.
        
        Args:
            prompt: The user input string.
            
        Yields:
            Chunk objects for text or ToolCallChunk objects for tool call status.
        """
        self.memory.add_message(Message(role="user", content=prompt))

        for _tool_round in range(_MAX_TOOL_ROUNDS):
            request = LLMRequest(
                messages=self.memory.get_messages(),
                tools=self._tool_defs if self._tool_defs else None,
            )

            current_tool_calls: dict[str, dict[str, str]] = {}
            current_content = ""

            # Consume the stream from the LLM
            async for chunk in self.llm.stream(request):
                yield chunk

                # Aggregate tool calls if they appear in the stream
                if isinstance(chunk, ToolCallChunk):
                    if chunk.tool_call_id not in current_tool_calls:
                        current_tool_calls[chunk.tool_call_id] = {"name": chunk.name or "", "arguments": ""}
                    current_tool_calls[chunk.tool_call_id]["arguments"] += chunk.arguments
                else:
                    if chunk.text:
                        current_content += chunk.text

            # If no tool calls were made, the turn is complete
            if not current_tool_calls:
                if current_content:
                    self.memory.add_message(Message(role="assistant", content=current_content))
                return

            # If tool calls were made, record the assistant's request in memory
            tool_call_payloads = []
            for call_id, call_data in current_tool_calls.items():
                tool_call_payloads.append({
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": call_data["name"],
                        "arguments": call_data["arguments"],
                    }
                })

            assistant_msg = Message(role="assistant", tool_calls=tool_call_payloads)
            self.memory.add_message(assistant_msg)

            # Execute each tool call and record the results in memory
            for call_id, call_data in current_tool_calls.items():
                tool_name = call_data["name"]
                raw_args = call_data["arguments"]

                result_str = ""
                if tool_name not in self._tool_map:
                    result_str = f"Error: Tool '{tool_name}' not found."
                else:
                    tool = self._tool_map[tool_name]
                    try:
                        kwargs = json.loads(raw_args) if raw_args else {}
                        # Support both sync and async tool functions.
                        # Sync tools are dispatched via run_in_executor so they never
                        # block the event loop (C5 fix).
                        if inspect.iscoroutinefunction(tool.func):
                            result = await tool.func(**kwargs)
                        else:
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(
                                None, lambda: tool.func(**kwargs)
                            )
                        result_str = str(result)
                    except (ValueError, TypeError, KeyError, RuntimeError) as e:
                        result_str = f"Tool error: {type(e).__name__}: {e}"

                tool_msg = Message(
                    role="tool",
                    tool_call_id=call_id,
                    name=tool_name,
                    content=result_str,
                )
                self.memory.add_message(tool_msg)

            # Loop continues to re-prompt LLM with the tool results now in memory

        # Safety valve: agent exceeded max allowed tool-call rounds
        raise RuntimeError(
            f"Agent exceeded maximum tool call depth ({_MAX_TOOL_ROUNDS} rounds). "
            "Check for tool errors or an unresponsive model."
        )

    async def run(self, prompt: str) -> str:
        """Run an agent interaction and return the final accumulated text.
        
        This method buffers the stream internally.
        
        Args:
            prompt: The user input string.
            
        Returns:
            The complete assistant response string.
        """
        result = ""
        async for chunk in self.stream(prompt):
            if hasattr(chunk, "text") and chunk.text:
                result += chunk.text
        return result

    def run_sync(self, prompt: str) -> str:
        """Synchronous wrapper around run().
        
        Safe to call from any context including inside running event loops
        (e.g. Jupyter notebooks, FastAPI handlers).
        
        Args:
            prompt: The user input string.
            
        Returns:
            The complete assistant response string.
        """
        import asyncio
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async context — run in a separate thread to avoid deadlock.
            # The lambda defers coroutine creation to the worker thread to avoid passing
            # a coroutine object across event loop boundaries (undefined behaviour).
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(lambda: asyncio.run(self.run(prompt))).result()

        return asyncio.run(self.run(prompt))
