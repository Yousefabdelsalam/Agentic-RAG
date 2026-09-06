"""Shared shape for the tools in this package.

The `Tool` protocol accepts arbitrary keyword arguments. Every tool here takes a
single string instead, because the planner produces tool calls as structured
output and one text field is the shape a model fills in reliably — nested
argument objects are where structured tool calls usually go wrong.

`ToolResult` is what reaches the graph state: never an exception, always a
result that says whether it succeeded. A failing tool must not end a run that
could still be answered from retrieval.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.core.base import Component
from app.core.logging import get_logger
from app.models.base import Schema

INPUT_ARGUMENT = "input"


class ToolResult(Schema):
    """The outcome of one tool call."""

    tool: str
    input: str = ""
    output: str = ""
    ok: bool = True
    error: str = ""

    def describe(self) -> str:
        """Render for a prompt, stating failure plainly rather than hiding it."""
        if not self.ok:
            return f"{self.tool}({self.input}) failed: {self.error}"
        return f"{self.tool}({self.input}) = {self.output}"


class TextTool(Component, ABC):
    """A tool that takes one string and returns one string."""

    name: str = "tool"
    description: str = ""
    argument_hint: str = "the input for this tool"

    def __init__(self) -> None:
        self.logger = get_logger(type(self).__module__)

    async def run(self, **kwargs: Any) -> str:
        """Protocol entry point; reads the single `input` argument."""
        return await self.execute(str(kwargs.get(INPUT_ARGUMENT, "")))

    @abstractmethod
    async def execute(self, text: str) -> str:
        """Do the work. Raise `ToolError` for anything the caller got wrong."""

    def describe(self) -> str:
        """One line for the planner's tool catalogue."""
        return f"- {self.name}: {self.description} (input: {self.argument_hint})"


class ToolError(Exception):
    """A tool could not produce a result for the input it was given."""
