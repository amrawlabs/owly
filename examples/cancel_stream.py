"""Demonstrate robust stream cancellation with Owly."""

from __future__ import annotations

import asyncio

from owly_ai import LLM
from owly_ai.core import BaseProvider, ProviderChunk, ProviderRequest
from owly_ai.core.exceptions import CancellationError
from owly_ai.core.types import LLMRequest, Message


class SlowProvider(BaseProvider):
    async def stream(self, request: ProviderRequest):
        del request
        for token in ["This ", "is ", "a ", "long ", "stream ", "that ", "will ", "stop."]:
            await asyncio.sleep(0.1)
            yield ProviderChunk(text=token, is_final=False)


async def consume(llm: LLM, request: LLMRequest) -> None:
    async for chunk in llm.stream(request):
        print(chunk.text, end="", flush=True)


async def main() -> None:
    llm = LLM(provider=SlowProvider(), model="demo")
    request = LLMRequest(messages=[Message(role="user", content="start")])

    task = asyncio.create_task(consume(llm, request))
    await asyncio.sleep(0.28)
    task.cancel()

    try:
        await task
    except CancellationError:
        print("\n[stream cancelled cleanly]")


if __name__ == "__main__":
    asyncio.run(main())
