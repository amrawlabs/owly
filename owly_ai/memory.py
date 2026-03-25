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
    explicit memory is provided. It stores messages in an internal list.
    """

    def __init__(self) -> None:
        """Initialize an empty message history."""
        self.messages: list[Message] = []

    def get_messages(self) -> tuple[Message, ...]:
        """Return a read-only tuple snapshot of the message history.
        
        This avoids copying the underlying list on each agent turn.
        """
        return tuple(self.messages)

    def add_message(self, message: Message) -> None:
        """Add a message to the internal history list."""
        self.messages.append(message)
