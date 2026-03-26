# Stream Pipeline

Owly stream execution flow:

1. Build `ProviderRequest` from `LLMRequest`
2. Route to provider adapter (`openai`, `gemini`, `vertex`, `claude`)
3. Wrap stream with cancellation safety
4. Normalize chunks for consistent output cadence
5. Yield `Chunk` / `ToolCallChunk` to caller

Pipeline shape:

`provider -> cancellable -> normalized -> user`

## Reliability notes

- semaphore limits active stream concurrency
- request + first-token timeouts are enforced
- provider errors are normalized to Owly exceptions
