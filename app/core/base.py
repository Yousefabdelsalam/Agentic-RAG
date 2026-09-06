"""Shared lifecycle contracts every long-lived component implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Startable(Protocol):
    """A component that acquires resources when the application boots."""

    async def start(self) -> None: ...


@runtime_checkable
class Closeable(Protocol):
    """A component that releases resources when the application shuts down."""

    async def close(self) -> None: ...


@runtime_checkable
class HealthCheckable(Protocol):
    """A component that can report whether its dependency is reachable."""

    async def healthy(self) -> bool: ...


class Component:
    """Default no-op lifecycle implementation for concrete components.

    Subclasses override only the hooks they need; composition is preferred for
    behaviour, so this base intentionally carries no domain logic.
    """

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def healthy(self) -> bool:
        return True
