"""The backend a deployment actually gets, chosen at startup.

Which backend is right cannot be known when the container is built: deciding
means connecting to Redis, and the container is assembled synchronously before
anything has a running loop. So this stands in for the real one, delegating
every call to whatever it is currently holding — the in-memory backend until
`start()` runs, and Redis afterwards if Redis answered.

The indirection is what lets the gateway and the knowledge base version share
one backend without either of them knowing which it is, or having to be rebuilt
when the answer changes.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.cache.base import CacheBackend
from app.cache.memory import InMemoryCacheBackend
from app.cache.redis import open_backend
from app.config.settings import CacheSettings
from app.core.base import Component
from app.core.logging import get_logger


class ManagedCacheBackend(Component):
    """Resolves to the best available backend when the application starts."""

    def __init__(self, settings: CacheSettings) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._backend: CacheBackend = InMemoryCacheBackend(
            index_max_entries=settings.semantic_max_entries
        )

    @property
    def name(self) -> str:
        return self._backend.name

    @property
    def backend(self) -> CacheBackend:
        """The backend in use, for tests and diagnostics."""
        return self._backend

    async def start(self) -> None:
        self._backend = await open_backend(self._settings)
        self.logger.info("cache.ready", backend=self.name, enabled=self._settings.enabled)

    async def get(self, key: str) -> str | None:
        return await self._backend.get(key)

    async def set(self, key: str, value: str, *, ttl_seconds: float) -> None:
        await self._backend.set(key, value, ttl_seconds=ttl_seconds)

    async def delete(self, key: str) -> None:
        await self._backend.delete(key)

    async def index_add(
        self, namespace: str, key: str, embedding: Sequence[float], *, ttl_seconds: float
    ) -> None:
        await self._backend.index_add(namespace, key, embedding, ttl_seconds=ttl_seconds)

    async def index_entries(self, namespace: str) -> list[tuple[str, tuple[float, ...]]]:
        return await self._backend.index_entries(namespace)

    async def index_remove(self, namespace: str, key: str) -> None:
        await self._backend.index_remove(namespace, key)

    async def clear(self) -> None:
        await self._backend.clear()

    async def healthy(self) -> bool:
        return await self._backend.healthy()

    async def close(self) -> None:
        await self._backend.close()
