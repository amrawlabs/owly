# Owly 🦉

> A lightweight, streaming-first LLM agent framework for Python.

[![PyPI version](https://img.shields.io/pypi/v/owly.svg)](https://pypi.org/project/owly/)
[![Python](https://img.shields.io/pypi/pyversions/owly.svg)](https://pypi.org/project/owly/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Owly is a minimal Python framework for building real-time, streaming-first LLM applications. It provides an agent orchestration loop, provider-agnostic tool calling, session memory, and both async streaming and synchronous interfaces — all without pulling in heavy dependencies.

---

## ✨ Features

- **Streaming-first** — every interface yields tokens as they arrive
- **Agent loop** — automatic tool detection, execution, and memory management
- **Provider-agnostic tools** — define tools once, works on all providers
- **OpenAI & Gemini** — built-in adapters, zero provider leakage in core logic
- **Sync + Async** — `stream()`, `run()`, and `run_sync()` out of the box
- **Custom providers** — implement one method, plug in any LLM backend
- **Custom memory** — bring your own persistent storage (Redis, SQLite, etc.)
- **No heavy dependencies** — no Pydantic, no LangChain, no DAG framework

---

## 📦 Installation

```bash
pip install owly
```

```bash
# Create a virtual environment
python -m venv .venv

# Activate it (Mac/Linux)
source .venv/bin/activate

# Install in editable mode
pip install -e .
```

**Requirements**: Python ≥ 3.11

Owly ships with adapters for OpenAI and Gemini. Install the providers you need:

```bash
pip install openai               # for OpenAI
pip install google-generativeai  # for Gemini
```

---

## 🔑 API Keys

Set your provider key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="AIza..."
```

Or pass it directly at initialization:

```python
llm = LLM(provider="openai", model="gpt-4o-mini", api_key="sk-...")
```

---

## 🚀 Quickstart

### Raw LLM Streaming

Stream tokens directly from any provider:

```python
import asyncio
from owly import LLM
from owly.core.types import LLMRequest, Message

async def main():
    llm = LLM(provider="openai", model="gpt-4o-mini")

    request = LLMRequest(
        messages=[Message(role="user", content="Explain async/await in Python.")],
        temperature=0.3,
    )

    async for chunk in llm.stream(request):
        print(chunk.text, end="", flush=True)

asyncio.run(main())
```

### Synchronous API

No async boilerplate required:

```python
from owly import LLM
from owly.core.types import LLMRequest, Message

llm = LLM(provider="gemini", model="gemini-1.5-flash")
msg = llm.generate_sync(LLMRequest(
    messages=[Message(role="user", content="What is 2 + 2?")]
))
print(msg.content)
```

---

## 🤖 Agent Quickstart

`Agent` manages the full loop: memory → LLM → tool call → result → LLM.

```python
import asyncio
from owly import LLM, Agent, Tool

def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    data = {"london": "Rainy, 12°C", "tokyo": "Clear, 22°C"}
    return data.get(location.lower(), "Sunny, 20°C")

async def main():
    llm = LLM(provider="openai", model="gpt-4o-mini")

    agent = Agent(
        llm=llm,
        tools=[Tool.from_function(get_weather)],
        system_prompt="You are a helpful weather assistant.",
    )

    # Multi-turn: memory is preserved automatically
    async for chunk in agent.stream("What's the weather in Tokyo?"):
        if hasattr(chunk, "text"):
            print(chunk.text, end="", flush=True)

    print()

    async for chunk in agent.stream("And in London?"):
        if hasattr(chunk, "text"):
            print(chunk.text, end="", flush=True)

asyncio.run(main())
```

### Synchronous Agent

Works from scripts, notebooks, and FastAPI handlers safely:

```python
agent = Agent(llm=llm, tools=[Tool.from_function(get_weather)])
answer = agent.run_sync("What's the weather in Paris?")
print(answer)
```

---

## 🛠 Defining Tools

Use `Tool.from_function` to wrap any Python function. Owly uses `inspect` to auto-generate the JSON schema from type hints and docstrings:

```python
from owly import Tool

def search_web(query: str, max_results: int = 5) -> str:
    """Search the web and return results."""
    ...

tool = Tool.from_function(search_web)
```

**Supported types**: `str`, `int`, `float`, `bool` (maps to JSON Schema types). Parameters without defaults are marked `required`.

**Async tools** are fully supported:

```python
async def fetch_data(url: str) -> str:
    """Fetch content from a URL."""
    ...

agent = Agent(llm=llm, tools=[Tool.from_function(fetch_data)])
```

---

## 🧠 Memory

`Agent` uses `InMemoryHistory` by default. Bring your own by implementing the `Memory` protocol:

```python
from owly.memory import Memory
from owly.core.types import Message

class RedisMemory:
    def get_messages(self) -> tuple[Message, ...]:
        # load from Redis
        ...

    def add_message(self, message: Message) -> None:
        # save to Redis
        ...

agent = Agent(llm=llm, memory=RedisMemory())
```

---

## 🔌 Supported Providers

| Provider | String key | Default env variable |
|---|---|---|
| OpenAI | `"openai"` | `OPENAI_API_KEY` |
| Google Gemini | `"gemini"` | `GEMINI_API_KEY` |

### Switching providers

```python
llm = LLM(provider="gemini", model="gemini-1.5-flash")
```

No changes needed to agent or tool code.

---

## 🧩 Custom Providers

Implement `BaseProvider` to plug in any LLM backend:

```python
from collections.abc import AsyncGenerator
from owly.core.interfaces import BaseProvider
from owly.core.types import ProviderChunk, ProviderRequest

class MyProvider(BaseProvider):
    async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]:
        # Call your LLM API and yield ProviderChunk objects
        yield ProviderChunk(text="Hello", is_final=False)
        yield ProviderChunk(text=" world", is_final=False)

# Use it directly
llm = LLM(provider=MyProvider(), model="my-model")
```

---

## 📐 Architecture

```
owly/
 ├── core/
 │    ├── types.py         # Data contracts (Message, Chunk, ToolDefinition, ...)
 │    ├── interfaces.py    # BaseProvider protocol
 │    └── exceptions.py    # ProviderError, ConfigurationError
 ├── providers/
 │    ├── openai.py        # OpenAI adapter
 │    └── gemini.py        # Gemini adapter
 ├── runtime/
 │    └── normalizer.py    # Stream normalization & chunk sizing
 ├── infra/
 │    └── config.py        # StreamConfig (chunk sizing)
 ├── utils/
 │    └── async_utils.py   # Async helpers
 ├── agent.py              # Agent orchestration loop
 ├── llm.py                # LLM public interface
 ├── tools.py              # Tool + ToolDefinition schema builder
 └── memory.py             # Memory protocol + InMemoryHistory
```

**Data pipeline:**

```
User prompt
    │
    ▼
Agent.stream()            ← appends to memory, loops on tool calls
    │
    ▼
LLM.stream()              ← builds ProviderRequest
    │
    ▼
run_stream_pipeline()     ← cancellable async wrapper
    │
    ▼
ProviderAdapter.stream()  ← OpenAI / Gemini / Custom
    │
    ▼
normalize_stream()        ← chunks buffered to target size, ToolCallChunks routed
    │
    ▼
Chunk | ToolCallChunk     ← yielded to caller
```

---

## 📖 API Reference

### `LLM`

```python
LLM(
    provider: str | BaseProvider,  # "openai", "gemini", or custom instance
    model: str,                    # e.g. "gpt-4o-mini", "gemini-1.5-flash"
    api_key: str | None = None,    # overrides env variable
    config: OwlyConfig | None = None,
)
```

| Method | Description |
|---|---|
| `stream(request)` | `AsyncIterator[Chunk \| ToolCallChunk]` |
| `generate(request)` | `async` → `Message` (full buffered response) |
| `generate_sync(request)` | Sync wrapper → `Message` |

### `Agent`

```python
Agent(
    llm: LLM,
    tools: list[Tool] | None = None,
    memory: Memory | None = None,       # default: InMemoryHistory
    system_prompt: str | None = None,
)
```

| Method | Description |
|---|---|
| `stream(prompt)` | `AsyncIterator[Chunk \| ToolCallChunk]` |
| `run(prompt)` | `async` → `str` (full buffered response) |
| `run_sync(prompt)` | Sync wrapper → `str` |

### `Tool`

```python
Tool.from_function(
    func: Callable,
    name: str | None = None,         # default: func.__name__
    description: str | None = None,  # default: func.__doc__
) -> Tool
```

### Core Types

| Type | Description |
|---|---|
| `Message(role, content, tool_calls, tool_call_id, name)` | A single chat message |
| `LLMRequest(messages, temperature, max_tokens, tools, metadata)` | Request to `LLM.stream` |
| `Chunk(text, is_final)` | A streamed text chunk |
| `ToolCallChunk(tool_call_id, name, arguments, is_final)` | A streamed tool call event |
| `ToolDefinition(name, description, parameters, required)` | Provider-agnostic tool schema |

---

## 🏃 Examples

```bash
# Run the weather agent (OpenAI)
OPENAI_API_KEY=sk-... python examples/agent_weather.py

# Run with Gemini
PROVIDER=gemini GEMINI_API_KEY=AIza... python examples/agent_weather.py

# Basic streaming
OPENAI_API_KEY=sk-... python examples/openai_stream.py

# Stream cancellation demo
OPENAI_API_KEY=sk-... python examples/cancel_stream.py
```

---

## 🧪 Running Tests

```bash
pip install pytest pytest-asyncio
pytest tests/
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository at [github.com/amrawlabs/owly](https://github.com/amrawlabs/owly)
2. **Create a branch** for your feature or fix: `git checkout -b feat/my-feature`
3. **Write tests** for any new behavior in `tests/`
4. **Keep it minimal** — Owly's core principle is zero unnecessary abstraction
5. **Open a pull request** with a clear description

### Adding a Provider

Copy the structure of `owly/providers/openai.py`:
- Implement `async def stream(self, request: ProviderRequest) -> AsyncGenerator[ProviderChunk, None]`
- Yield `ProviderChunk` with `text=` for text and `tool_call_id=`, `tool_name=`, `tool_arguments=` for tool calls
- Register it in `LLM._resolve_provider()`

### Design Principles

- **No provider leakage** — provider-specific code lives only in `providers/`
- **Streaming-first** — every public interface is an async generator
- **Minimal dependencies** — if it can be done with stdlib, do it with stdlib
- **Typed contracts** — all data crosses module boundaries as frozen dataclasses

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">Built with ❤️ by <a href="https://github.com/amrawlabs">amrawlabs</a></p>
