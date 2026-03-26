# Providers

## Built-in provider keys

- `openai`
- `gemini`
- `vertex`
- `claude`

## OpenAI

```python
llm = LLM(provider="openai", model="gpt-4o-mini")
```

Env var: `OPENAI_API_KEY`

## Gemini (AI Studio)

```python
llm = LLM(provider="gemini", model="gemini-2.5-flash")
```

Env var: `GEMINI_API_KEY`

## Vertex AI Gemini

```python
llm = LLM(
    provider="vertex",
    model="gemini-2.5-flash",
    project_id="my-project",
    region="us-central1",
)
```

Env vars:

- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION` (optional)

## Anthropic Claude

```python
llm = LLM(provider="claude", model="claude-3-5-sonnet-latest")
```

Env var: `ANTHROPIC_API_KEY`
