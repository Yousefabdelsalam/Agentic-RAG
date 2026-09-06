"""Contracts and registry for agent-callable tools."""

from __future__ import annotations

from typing import Any, Protocol

from app.core.exceptions import NotFoundError


class Tool(Protocol):
    """A capability an agent may call, described well enough to be bound to an LLM."""

    name: str
    description: str

    async def run(self, **kwargs: Any) -> Any: ...


class ToolRegistry:
    """Name-to-tool lookup, populated at wiring time."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise NotFoundError(f"Unknown tool: {name}") from exc

    def all(self) -> list[Tool]:
        return list(self._tools.values())
