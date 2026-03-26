# Development

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install pytest pytest-asyncio mkdocs mkdocs-material mkdocstrings[python]
```

## Run tests

```bash
pytest tests/
```

## Build docs locally

```bash
mkdocs serve
```

Open: `http://127.0.0.1:8000`

## Contribution rules

- keep provider-specific logic inside `owly_ai/providers/`
- do not leak SDK-specific payloads past adapter boundaries
- preserve cancellation and timeout guarantees
- add tests for behavioral changes
