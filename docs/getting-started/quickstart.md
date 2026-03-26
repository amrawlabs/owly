# Quickstart

## Stream from OpenAI

```python
import asyncio

from owly_ai import LLM
from owly_ai.core.types import LLMRequest, Message


async def main() -> None:
    llm = LLM(provider="openai", model="gpt-4o-mini")
    request = LLMRequest(
        messages=[Message(role="user", content="Explain async IO in one paragraph.")],
        temperature=0.2,
        request_id="req-quickstart-1",
    )

    async for chunk in llm.stream(request):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)

    print()


asyncio.run(main())
```

## Stream from Vertex

```python
llm = LLM(
    provider="vertex",
    model="gemini-2.5-flash",
    project_id="your-gcp-project",
    region="us-central1",
)
```
