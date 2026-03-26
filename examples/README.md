# Examples

All examples assume:

- Python 3.11+
- `pip install owly-ai`

## 1) OpenAI stream

```bash
export OPENAI_API_KEY="sk-..."
python examples/openai_stream.py
```

## 2) Gemini stream (AI Studio)

```bash
export GEMINI_API_KEY="AIza..."
python examples/gemini_stream.py
```

## 3) Vertex stream (Gemini on GCP)

```bash
export GOOGLE_CLOUD_PROJECT="my-project"
export GOOGLE_CLOUD_LOCATION="us-central1"
# optional: path to service account json
export CREDENTIALS_PATH="/path/to/sa.json"
python examples/vertex_stream.py
```

## 4) Claude stream

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python examples/claude_stream.py
```

## 5) Cancellation demo

```bash
python examples/cancel_stream.py
```

## 6) Agent + tool demo

```bash
export PROVIDER=openai   # or gemini / vertex
export OPENAI_API_KEY="sk-..."
python examples/agent_weather.py
```
