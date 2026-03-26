# Release Process

## 1) Bump version

Update `version` in `pyproject.toml`.

## 2) Run quality checks

```bash
pytest tests/
python -m compileall owly_ai examples
```

## 3) Build package

```bash
rm -rf dist build *.egg-info
python -m pip install --upgrade build twine
python -m build
```

## 4) Publish

```bash
python -m twine upload dist/*
```

Use PyPI token auth:

- Username: `__token__`
- Password: `pypi-...`

## 5) Verify

```bash
python -m pip install -U owly-ai
python -c "from owly_ai import LLM; print(LLM.__name__)"
```
