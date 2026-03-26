"""Core data contracts for Owly."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """Provider-agnostic normalized tool definition schema.

    This represents the contract for a tool that can be used by an LLM,
    independent of any specific provider's API format.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    required: list[str]


@dataclass(slots=True, frozen=True)
class Message:
    """A single chat message used by runtime requests.

    Represents a message from a user, assistant, system, or tool.
    """

    role: str
    content: str | None = None
    tool_calls: list[Mapping[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass(slots=True, frozen=True)
class LLMRequest:
    """Public request accepted by ``LLM.stream``.

    Contains all user-provided context and parameters for a generation.
    ``metadata`` is fully user-owned — Owly never reads or writes to it.
    """

    messages: Sequence[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    tools: Sequence[ToolDefinition] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    request_id: str | None = None


@dataclass(slots=True, frozen=True)
class RuntimeHints:
    """Internal runtime configuration passed by the LLM layer to provider adapters.

    Carries Owly-owned parameters (timeouts, queue limits, cloud project info)
    so they are never mixed with the user-facing ``LLMRequest.metadata`` dict.
    """

    request_timeout: float = 30.0
    first_token_timeout: float = 5.0
    queue_maxsize: int = 256
    credentials: Any | None = None  # For Vertex AI dynamic auth
    project_id: str | None = None
    region: str | None = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(slots=True, frozen=True)
class ProviderRequest:
    """Provider-facing request with resolved model identity and runtime hints.

    Internal type used by adapters to communicate with LLM providers.
    ``metadata`` is forwarded verbatim from the public ``LLMRequest`` and remains
    user-owned; providers must not write internal keys into it.
    """

    model: str
    messages: Sequence[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    stop: list[str] | None = None
    tools: Sequence[ToolDefinition] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    hints: RuntimeHints = field(default_factory=RuntimeHints)
    request_id: str | None = None


@dataclass(slots=True, frozen=True)
class ProviderChunk:
    """Provider-emitted chunk payload before runtime normalization.

    Raw output from a provider, potentially containing text or tool call data.
    """

    text: str | None = None
    is_final: bool = False
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: str | None = None
    raw: Any | None = None


@dataclass(slots=True, frozen=True)
class Chunk:
    """User-facing normalized chunk emitted by Owly.

    Represents a piece of generated text intended for the end user.
    """

    text: str
    is_final: bool = False


@dataclass(slots=True, frozen=True)
class ToolCallChunk:
    """User-facing chunk emitted when a tool call is streamed.

    Allows real-time observability into the agent's decision to use a tool.
    """

    tool_call_id: str
    name: str | None = None
    arguments: str = ""
    is_final: bool = False
