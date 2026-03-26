"""Stream text from Gemini (AI Studio) using owly_ai."""

from __future__ import annotations

import asyncio
import os

from owly_ai import LLM
from owly_ai.core.types import LLMRequest, Message


async def main() -> None:
    if "GEMINI_API_KEY" not in os.environ:
        print("Please set GEMINI_API_KEY")
        return

    llm = LLM(provider="gemini", model=os.environ.get("MODEL", "gemini-2.5-flash"))
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
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)

    print()


if __name__ == "__main__":
    asyncio.run(main())
