"""Contracts for conversational and long-term memory."""

from __future__ import annotations

from typing import Any, Protocol

from app.models.base import Schema


class MemoryRecord(Schema):
    """A single stored interaction or fact belonging to a session."""

    session_id: str
    role: str
    content: str
    metadata: dict[str, Any] = {}


class MemoryStore(Protocol):
    """Reads and writes memory records for a session."""

    async def append(self, record: MemoryRecord) -> None: ...

    async def history(self, session_id: str, *, limit: int = 20) -> list[MemoryRecord]: ...

    async def clear(self, session_id: str) -> None: ...
