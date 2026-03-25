"""Real-time OpenAI streaming example for Owly."""

from __future__ import annotations

import asyncio

from owly import LLM
from owly.core.types import LLMRequest, Message


async def main() -> None:
    llm = LLM(provider="openai", model="gpt-4o-mini")
    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="In one short paragraph, explain why streaming matters for voice assistants.",
            )
        ],
        temperature=0.2,
    )

    async for chunk in llm.stream(request):
        # Print each token immediately for low-latency output.
        print(chunk.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
