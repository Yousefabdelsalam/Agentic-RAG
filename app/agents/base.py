"""Contracts for LangGraph-backed agents."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol

from app.models.base import Schema


class AgentRequest(Schema):
    """Input to an agent run."""

    session_id: str
    query: str
    context: dict[str, Any] = {}


class AgentResponse(Schema):
    """Terminal output of an agent run."""

    session_id: str
    answer: str
    metadata: dict[str, Any] = {}


class Agent(Protocol):
    """A compiled graph exposed behind a stable invoke/stream interface."""

    name: str

    async def invoke(self, request: AgentRequest) -> AgentResponse: ...

    def stream(self, request: AgentRequest) -> AsyncIterator[str]: ...
