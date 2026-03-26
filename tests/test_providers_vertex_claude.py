"""Provider integration tests for Claude and Vertex (new google.genai SDK)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from owly_ai.core.exceptions import CancellationError, ProviderTimeoutError
from owly_ai.core.types import Message, ProviderRequest, RuntimeHints
from owly_ai.providers.claude import ClaudeProvider
from owly_ai.providers.vertex import VertexProvider
from owly_ai.runtime.cancellation import cancellable_stream


# ---------------------------------------------------------------------------
# Claude fakes
# ---------------------------------------------------------------------------

class _FakeClaudeStream:
    def __init__(self, events: list[Any], delay: float = 0.0) -> None:
        self._events = iter(events)
        self._delay = delay
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._delay:
            await asyncio.sleep(self._delay)
        try:
            return next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


class _FakeClaudeMessages:
    def __init__(self, stream: _FakeClaudeStream) -> None:
        self._stream = stream

    async def create(self, **_: Any):
        return self._stream


class _FakeClaudeClient:
    def __init__(self, stream: _FakeClaudeStream) -> None:
        self.messages = _FakeClaudeMessages(stream)


# ---------------------------------------------------------------------------
# Vertex / Gemini-newSDK fakes
#
# The new VertexProvider uses:
#   client.aio.models.generate_content_stream(model=..., contents=..., config=...)
# which returns an async iterator.  We inject via _client_factory.
# ---------------------------------------------------------------------------

class _FakeAsyncStream:
    """Async iterable over a list of events with optional per-event delay."""

    def __init__(self, events: list[Any], delay: float = 0.0) -> None:
        self._events = events
        self._delay = delay

    def __aiter__(self):
        return self._aiter()

    async def _aiter(self):
        for event in self._events:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield event


class _FakeAioModels:
    def __init__(self, events: list[Any], delay: float = 0.0) -> None:
        self._events = events
        self._delay = delay

    async def generate_content_stream(self, *, model: str, contents: Any, config: Any) -> _FakeAsyncStream:
        return _FakeAsyncStream(self._events, self._delay)


class _FakeAio:
    def __init__(self, events: list[Any], delay: float = 0.0) -> None:
        self.models = _FakeAioModels(events, delay)


class _FakeGenaiClient:
    def __init__(self, events: list[Any], delay: float = 0.0, credentials: Any = None) -> None:
        self.aio = _FakeAio(events, delay)
        self.credentials = credentials


def _make_vertex_provider(
    events: list[Any], delay: float = 0.0, credentials: Any = None
) -> VertexProvider:
    def factory(*args, **kwargs):
        return _FakeGenaiClient(events, delay, credentials=kwargs.get("credentials"))

    return VertexProvider(
        project_id="test-proj",
        region="test-region",
        credentials=credentials,
        _client_factory=factory,
    )


def _make_request(
    model: str = "vertex-test",
    content: str = "hi",
    request_timeout: float = 2.0,
    first_token_timeout: float = 2.0,
) -> ProviderRequest:
    return ProviderRequest(
        model=model,
        messages=[Message(role="user", content=content)],
        hints=RuntimeHints(
            request_timeout=request_timeout,
            first_token_timeout=first_token_timeout,
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_claude_streaming_output() -> None:
    events = [
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hel")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="lo")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="")),
    ]
    provider = ClaudeProvider(client=_FakeClaudeClient(_FakeClaudeStream(events)))

    request = ProviderRequest(
        model="claude-test",
        messages=[Message(role="user", content="hi")],
        hints=RuntimeHints(request_timeout=1.0, first_token_timeout=1.0),
    )

    texts = []
    async for chunk in provider.stream(request):
        if chunk.text:
            texts.append(chunk.text)

    assert texts == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_vertex_streaming_output() -> None:
    events = [
        SimpleNamespace(text=""),
        SimpleNamespace(text="Hello"),
        SimpleNamespace(text=" world"),
    ]
    provider = _make_vertex_provider(events)
    request = _make_request()

    texts = []
    async for chunk in provider.stream(request):
        if chunk.text:
            texts.append(chunk.text)

    assert "".join(texts) == "Hello world"


@pytest.mark.asyncio
async def test_vertex_credentials_propagation():
    """Verify that credentials from provider/hints reach the client factory."""
    events = [SimpleNamespace(text="Auth check")]
    captured_creds = []

    def factory(*args, **kwargs):
        creds = kwargs.get("credentials")
        captured_creds.append(creds)
        return _FakeGenaiClient(events, credentials=creds)

    # 1. Credentials from constructor
    provider = VertexProvider(
        project_id="test-proj",
        region="test-region",
        credentials="my-init-creds",
        _client_factory=factory,
    )

    req = ProviderRequest(
        model="gemini-1.5-flash",
        messages=[Message(role="user", content="hi")],
    )
    # Trigger stream to hit _get_client
    async for _ in provider.stream(req):
        break

    assert captured_creds[-1] == "my-init-creds"

    # 2. Credentials from hints (takes priority)
    hints = RuntimeHints(credentials="my-hint-creds")
    req_with_hints = ProviderRequest(
        model="gemini-1.5-flash",
        messages=[Message(role="user", content="hi")],
        hints=hints,
    )
    async for _ in provider.stream(req_with_hints):
        break

    assert captured_creds[-1] == "my-hint-creds"


@pytest.mark.asyncio
async def test_cancellation_mid_stream() -> None:
    events = [SimpleNamespace(text="token") for _ in range(200)]
    provider = _make_vertex_provider(events, delay=0.01)
    request = _make_request(model="vertex-cancel", request_timeout=5.0, first_token_timeout=5.0)

    async def consume() -> None:
        async for _ in cancellable_stream(provider.stream(request)):
            await asyncio.sleep(0)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()

    with pytest.raises(CancellationError):
        await task


@pytest.mark.asyncio
async def test_timeout_behavior() -> None:
    delayed_stream = _FakeClaudeStream(
        [SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="late"))],
        delay=0.2,
    )
    provider = ClaudeProvider(client=_FakeClaudeClient(delayed_stream))
    request = ProviderRequest(
        model="claude-timeout",
        messages=[Message(role="user", content="hi")],
        hints=RuntimeHints(request_timeout=1.0, first_token_timeout=0.01),
    )

    with pytest.raises(ProviderTimeoutError):
        async for _ in provider.stream(request):
            pass


@pytest.mark.asyncio
async def test_vertex_timeout_first_token() -> None:
    """Vertex provider must raise ProviderTimeoutError when first token is too slow."""
    events = [SimpleNamespace(text="late")]
    provider = _make_vertex_provider(events, delay=0.3)
    request = _make_request(first_token_timeout=0.01, request_timeout=5.0)

    with pytest.raises(ProviderTimeoutError):
        async for _ in provider.stream(request):
            pass


@pytest.mark.asyncio
async def test_vertex_system_and_multiturn() -> None:
    """Verify _to_contents correctly handles system prompt and multi-turn history."""
    from owly_ai.providers.vertex import _to_contents

    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi there!"),
        Message(role="user", content="How are you?"),
    ]
    request = ProviderRequest(
        model="x",
        messages=messages,
        hints=RuntimeHints(),
    )
    system_instruction, contents = _to_contents(request)

    assert system_instruction == "You are helpful."
    assert len(contents) == 3
    assert contents[0]["role"] == "user"
    assert contents[1]["role"] == "model"  # assistant → model
    assert contents[2]["role"] == "user"
    assert contents[1]["parts"][0]["text"] == "Hi there!"


@pytest.mark.asyncio
async def test_runtime_hints_not_in_metadata() -> None:
    """LLM layer must not write __owly_* keys into user metadata."""
    from owly_ai.core.interfaces import BaseProvider
    from owly_ai.core.types import LLMRequest, RuntimeHints
    from owly_ai.llm import LLM
    from unittest.mock import AsyncMock, MagicMock

    captured: list[ProviderRequest] = []

    class _CapturingProvider(BaseProvider):
        async def stream(self, request: ProviderRequest):
            captured.append(request)
            return
            yield  # make it a generator

    llm = LLM(provider=_CapturingProvider(), model="test-model")  # type: ignore[arg-type]
    req = LLMRequest(
        messages=[Message(role="user", content="hi")],
        metadata={"user_key": "user_value"},
    )

    # Consume to trigger the stream call
    async for _ in llm.stream(req):
        pass

    assert len(captured) == 1
    pr = captured[0]
    # User metadata must be untouched
    assert pr.metadata == {"user_key": "user_value"}
    # Owly config must be in hints, not metadata
    assert pr.hints.request_timeout == 30.0
    assert "__owly_request_timeout" not in pr.metadata
    assert "__owly_first_token_timeout" not in pr.metadata
