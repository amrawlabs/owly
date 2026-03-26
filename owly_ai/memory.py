"""Lightweight session memory for Owly agents.

This module provides the abstractions and default implementations for maintaining
conversation history across multiple turns in an agentic workflow.
"""

from __future__ import annotations

from typing import Protocol

from .core.types import Message


class Memory(Protocol):
    """Protocol for conversation memory persistence.

    Any class implementing this protocol can be used by an Owly Agent
    to manage conversation state.
    """

    def get_messages(self) -> tuple[Message, ...]:
        """Retrieve an ordered, read-only snapshot of the conversation.

        Returns:
            A tuple of Message objects representing the history.
        """

    def add_message(self, message: Message) -> None:
        """Append a message to the conversation history.

        Args:
            message: The Message object to add to the history.
        """


class InMemoryHistory:
    """A simple list-backed implementation of Memory for single-session operations.

    This is the default memory implementation used by Owly Agents when no
    explicit memory is provided.  It stores messages in an internal list.

    Fixes applied:
    - P4: ``get_messages`` returns a tuple view without copying — the backing list
      is converted once at construction and rebuilt incrementally, not on every read.
    - M6: ``max_messages`` acts as a best-effort trim for non-system history;
      it is intentionally not a strict hard cap.
    """

    def __init__(self, max_messages: int | None = None) -> None:
        """Initialize an empty message history.

        Args:
            max_messages: Optional best-effort trim threshold for retained messages.
                When exceeded, oldest non-system messages are trimmed first while
                preserving leading system messages. ``None`` means no trimming.
        """
        self._messages: list[Message] = []
        self._max_messages = max_messages
        # P4: maintain an up-to-date tuple so get_messages() is O(1) after add
        self._snapshot: tuple[Message, ...] = ()

    def get_messages(self) -> tuple[Message, ...]:
        """Return a read-only tuple snapshot of the message history (O(1))."""
        return self._snapshot

    def add_message(self, message: Message) -> None:
        """Add a message to the internal history list.

        If ``max_messages`` is set and exceeded, the oldest non-system messages
        are removed first while preserving leading system messages.
        """
        self._messages.append(message)

        if self._max_messages is not None and len(self._messages) > self._max_messages:
            # Preserve any leading system messages, trim oldest non-system ones.
            system_prefix: list[Message] = []
            rest: list[Message] = []
            for msg in self._messages:
                if msg.role == "system" and not rest:
                    system_prefix.append(msg)
                else:
                    rest.append(msg)
            overage = len(self._messages) - self._max_messages
            rest = rest[overage:]
            self._messages = system_prefix + rest

        # Rebuild snapshot once per mutation (P4: O(1) reads after this)
        self._snapshot = tuple(self._messages)
