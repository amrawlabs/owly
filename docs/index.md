# owly-ai

`owly-ai` is a streaming-first LLM runtime for real-time systems.

It is designed for:

- low-latency token streaming
- cancellation-safe execution
- provider abstraction without leakage
- predictable behavior under concurrency

## What you get

- Unified runtime API: `LLM.stream()`, `LLM.generate()`, `LLM.generate_sync()`
- Built-in providers: OpenAI, Gemini, Vertex AI Gemini, Anthropic Claude
- Structured stream outputs: `Chunk` and `ToolCallChunk`
- Runtime safety controls: request timeout, first token timeout, max concurrency

## Package naming

- Install from PyPI: `pip install owly-ai`
- Import in Python: `from owly_ai import LLM`

## Next steps

- Start with [Installation](getting-started/installation.md)
- Run your first stream from [Quickstart](getting-started/quickstart.md)
- See provider details in [Providers](guides/providers.md)
