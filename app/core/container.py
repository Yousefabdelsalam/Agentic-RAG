"""Dependency container: owns component instances and their lifecycle."""

from __future__ import annotations

from typing import TypeVar

from app.config.settings import Settings
from app.core.base import Component
from app.core.exceptions import ConfigurationError
from app.core.logging import get_logger

T = TypeVar("T", bound=Component)

logger = get_logger(__name__)


class Container:
    """Composition root holding singletons keyed by their protocol/class.

    Components are registered once at startup and resolved by type, so callers
    depend on abstractions rather than constructing collaborators themselves.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._components: dict[type, Component] = {}
        self._order: list[type] = []

    @property
    def settings(self) -> Settings:
        return self._settings

    def register(self, key: type[T], component: T) -> None:
        """Bind a component instance to the key used for resolution."""
        if key in self._components:
            raise ConfigurationError(f"Component already registered: {key.__name__}")
        self._components[key] = component
        self._order.append(key)

    def resolve(self, key: type[T]) -> T:
        """Return the component bound to `key`."""
        try:
            component = self._components[key]
        except KeyError as exc:
            raise ConfigurationError(f"Component not registered: {key.__name__}") from exc
        return component  # type: ignore[return-value]

    async def startup(self) -> None:
        """Start components in registration order."""
        for key in self._order:
            await self._components[key].start()
            logger.info("component.started", component=key.__name__)

    async def shutdown(self) -> None:
        """Close components in reverse registration order, never raising."""
        for key in reversed(self._order):
            try:
                await self._components[key].close()
            except Exception:
                logger.exception("component.close_failed", component=key.__name__)
            else:
                logger.info("component.closed", component=key.__name__)

    async def health(self) -> dict[str, bool]:
        """Report reachability of every registered component."""
        return {key.__name__: await self._components[key].healthy() for key in self._order}
