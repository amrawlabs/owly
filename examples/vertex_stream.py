"""Stream text from Vertex AI Gemini using owly_ai."""

from __future__ import annotations

import asyncio
import json
import os

from owly_ai import LLM
from owly_ai.core.types import LLMRequest, Message


async def main() -> None:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project_id:
        print("Please set GOOGLE_CLOUD_PROJECT")
        return

    credentials = None
    creds_path = os.environ.get("CREDENTIALS_PATH")
    if creds_path and os.path.exists(creds_path):
        with open(creds_path, encoding="utf-8") as f:
            credentials = json.load(f)

    llm = LLM(
        provider="vertex",
        model=os.environ.get("MODEL", "gemini-2.5-flash"),
        project_id=project_id,
        region=region,
        credentials=credentials,
    )
    request = LLMRequest(
        messages=[
            Message(
                role="user",
                content="Explain in one paragraph how streaming improves voice latency.",
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
