"""Real-time Gemini streaming example for Owly."""

from __future__ import annotations

import asyncio

from owly import LLM
from owly.core.types import LLMRequest, Message


async def main() -> None:
    llm = LLM(provider="gemini", model="gemini-1.5-flash")
    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="In one short paragraph, explain why streaming matters for chat UIs.",
            )
        ],
        temperature=0.2,
    )

    async for chunk in llm.stream(request):
        # Flush each piece immediately to preserve real-time feel.
        print(chunk.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
