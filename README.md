# owly-ai

Streaming-first LLM runtime and lightweight agent loop for real-time systems.

[![PyPI version](https://img.shields.io/pypi/v/owly-ai.svg)](https://pypi.org/project/owly-ai/)
[![Python](https://img.shields.io/pypi/pyversions/owly-ai.svg)](https://pypi.org/project/owly-ai/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## New in this release

- Vertex AI Gemini provider (`provider="vertex"`)
- Anthropic Claude provider (`provider="claude"`)
- Unified timeout controls (`request_timeout`, `first_token_timeout`)
- Improved cancellation safety and stream cleanup
- Provider-level observability (provider, model, latency, request_id)

## Installation

```bash
pip install owly-ai
```

Use in other projects:

```toml
# pyproject.toml
dependencies = ["owly-ai"]
```

```txt
# requirements.txt
owly-ai
```

Import path:

```python
from owly_ai import LLM
```

## Supported providers

- OpenAI: `provider="openai"`
- Gemini (Google AI Studio): `provider="gemini"`
- Vertex AI Gemini: `provider="vertex"`
- Anthropic Claude: `provider="claude"`

## Credentials

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_CLOUD_PROJECT="my-project"      # Vertex
export GOOGLE_CLOUD_LOCATION="us-central1"    # Vertex (optional)
```

## Quickstart

```python
import asyncio

from owly_ai import LLM
from owly_ai.core.types import LLMRequest, Message


async def main() -> None:
    llm = LLM(provider="openai", model="gpt-4o-mini")
    request = LLMRequest(
        messages=[Message(role="user", content="Explain asyncio in one paragraph.")],
        temperature=0.2,
        request_id="req-123",
    )

    async for chunk in llm.stream(request):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)

    print()


asyncio.run(main())
```

## Agent quickstart

```python
import asyncio

from owly_ai import Agent, LLM, Tool


def get_weather(location: str) -> str:
    data = {"tokyo": "Clear, 22C", "london": "Rainy, 12C"}
    return data.get(location.lower(), "Sunny, 20C")


async def main() -> None:
    llm = LLM(provider="openai", model="gpt-4o-mini")
    agent = Agent(
        llm=llm,
        tools=[Tool.from_function(get_weather)],
        system_prompt="You are a concise weather assistant.",
    )

    async for chunk in agent.stream("What's the weather in Tokyo?"):
        if hasattr(chunk, "text") and chunk.text:
            print(chunk.text, end="", flush=True)

    print()


asyncio.run(main())
```

## Runtime configuration

```python
from owly_ai import LLM
from owly_ai.infra.config import OwlyConfig

cfg = OwlyConfig(
    request_timeout=30.0,
    first_token_timeout=5.0,
    max_concurrency=256,
    queue_maxsize=256,
)
llm = LLM(provider="claude", model="claude-3-5-sonnet-latest", config=cfg)
```

## Examples

See [examples/README.md](examples/README.md) for runnable commands:

- `examples/openai_stream.py`
- `examples/gemini_stream.py`
- `examples/vertex_stream.py`
- `examples/claude_stream.py`
- `examples/cancel_stream.py`
- `examples/agent_weather.py`

## Architecture

```text
owly_ai/
 ├── core/        # contracts + exceptions + core types
 ├── providers/   # provider adapters
 ├── runtime/     # cancellation + normalizer + pipeline
 ├── infra/       # config + logging
 ├── utils/       # async helpers
 ├── llm.py       # public runtime interface
 ├── agent.py     # optional agent loop
 ├── tools.py     # tool definition helpers
 └── memory.py    # memory protocol + in-memory impl
```

Pipeline:

`provider -> cancellable -> normalized -> user`

## Contributing

- Keep provider-specific logic inside `owly_ai/providers/*`
- Preserve streaming and cancellation guarantees
- Add tests for new behaviors in `tests/`

## License

Apache 2.0. See [LICENSE](LICENSE).
