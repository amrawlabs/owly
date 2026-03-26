# Installation

## Basic install

```bash
pip install owly-ai
```

## Provider SDKs

Install provider SDKs you plan to use:

```bash
pip install openai
pip install google-genai
pip install anthropic
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Verify install

```bash
python -c "from owly_ai import LLM; print(LLM.__name__)"
```
