"""Low-dependency tool definitions for agent functions.

This module allows defining Python functions as tools that can be called by LLMs.
It uses native Python inspection to generate provider-neutral schemas.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from .core.types import ToolDefinition


@dataclass(slots=True, frozen=True)
class Tool:
    """A callable tool that can be executed by an agent.
    
    Combines a Python callable with its provider-agnostic definition schema.
    """

    name: str
    description: str
    func: Callable[..., Any]
    definition: ToolDefinition

    @classmethod
    def from_function(cls, func: Callable[..., Any], name: str | None = None, description: str | None = None) -> Tool:
        """Create a Tool from a Python function using inspect.
        
        This method automatically extracts parameter types and descriptions from the 
        function signature and docstring.
        
        Args:
            func: The Python function to wrap as a tool.
            name: Optional override for the tool name. Defaults to function name.
            description: Optional override for tool description. Defaults to docstring.
            
        Returns:
            A Tool instance ready to be used by an Owly Agent.
        """
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or f"Tool {tool_name}"

        sig = inspect.signature(func)
        properties = {}
        required = []

        # Map Python types to JSON Schema types
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # Skip *args and **kwargs — they don't map to JSON schema properties
            if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            param_type = "string"
            if param.annotation is int:
                param_type = "integer"
            elif param.annotation is float:
                param_type = "number"
            elif param.annotation is bool:
                param_type = "boolean"

            properties[param_name] = {"type": param_type}
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        # Build the neutral ToolDefinition
        definition = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=properties,
            required=required,
        )

        return cls(name=tool_name, description=tool_desc, func=func, definition=definition)
