"""Core data contracts for Owly."""

from __future__ import annotations

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
    """Public request documentation accepted by ``LLM.stream``.
    
    Contains all user-provided context and parameters for a generation.
    """

    messages: Sequence[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    tools: Sequence[ToolDefinition] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProviderRequest:
    """Provider-facing request with resolved model identity.
    
    Internal type used by adapters to communicate with LLM providers.
    """

    model: str
    messages: Sequence[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    tools: Sequence[ToolDefinition] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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
