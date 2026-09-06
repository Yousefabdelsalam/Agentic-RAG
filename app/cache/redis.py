"""Redis cache backend, and the decision of whether to use one.

Redis is the production choice because the cache is only worth much when it is
shared: a hit rate computed per replica is a fraction of the real one, and a
knowledge base version that lives in one process cannot stop another from
serving answers from a corpus that no longer exists.

Nothing here is required to start. The driver is imported lazily and the
connection is verified once at boot; if the package is absent or the server
unreachable, `open_backend` returns the in-memory backend and says so. A cache
that refuses to start takes down a deployment over an optimisation, which is the
wrong trade in every direction.

A Redis that fails mid-flight raises `DependencyError`, which the gateway
catches, counts, and turns into a miss. The policy that a cache failure never
fails a request lives there, in one place, rather than being reimplemented by
every backend.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import orjson

from app.cache.memory import InMemoryCacheBackend
from app.config.settings import CacheSettings
from app.core.exceptions import DependencyError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Suffix of the sorted set holding insertion order for one semantic namespace.
#: Redis hashes are unordered, so trimming the index to a bound needs a second
#: structure to say which entry is oldest.
_ORDER_SUFFIX = ":order"


class RedisCacheBackend:
    """Cache backend over a shared Redis instance."""

    name = "redis"

    def __init__(self, client: Any, settings: CacheSettings) -> None:
        self.logger = get_logger(__name__)
        self._client = client
        self._index_max_entries = settings.semantic_max_entries

    async def get(self, key: str) -> str | None:
        value = await self._call("get", self._client.get(key))
        if value is None:
            return None
        return value.decode("utf-8") if isinstance(value, bytes) else str(value)

    async def set(self, key: str, value: str, *, ttl_seconds: float) -> None:
        expiry = int(ttl_seconds) if ttl_seconds > 0 else None
        await self._call("set", self._client.set(key, value, ex=expiry))

    async def delete(self, key: str) -> None:
        await self._call("delete", self._client.delete(key))

    async def index_add(
        self, namespace: str, key: str, embedding: Sequence[float], *, ttl_seconds: float
    ) -> None:
        """Add one vector to the index and trim the oldest entries past the bound."""
        order = namespace + _ORDER_SUFFIX
        payload = orjson.dumps([float(value) for value in embedding])
        pipeline = self._client.pipeline()
        pipeline.hset(namespace, key, payload)
        pipeline.zadd(order, {key: time.time()})
        if ttl_seconds > 0:
            # The index outlives individual entries only as long as the longest
            # of them; refreshing on write keeps a busy namespace alive and lets
            # an abandoned one expire on its own.
            pipeline.expire(namespace, int(ttl_seconds))
            pipeline.expire(order, int(ttl_seconds))
        await self._call("index_add", pipeline.execute())
        await self._trim(namespace, order)

    async def index_entries(self, namespace: str) -> list[tuple[str, tuple[float, ...]]]:
        raw = await self._call("index_entries", self._client.hgetall(namespace))
        if not raw:
            return []
        entries: list[tuple[str, tuple[float, ...]]] = []
        for field, payload in raw.items():
            key = field.decode("utf-8") if isinstance(field, bytes) else str(field)
            vector = _decode_vector(payload)
            if vector is not None:
                entries.append((key, vector))
        return entries

    async def index_remove(self, namespace: str, key: str) -> None:
        pipeline = self._client.pipeline()
        pipeline.hdel(namespace, key)
        pipeline.zrem(namespace + _ORDER_SUFFIX, key)
        await self._call("index_remove", pipeline.execute())

    async def clear(self) -> None:
        """Drop every key this cache owns, leaving the rest of the database alone."""
        await self._call("clear", self._client.flushdb())

    async def healthy(self) -> bool:
        """Report reachability. Unlike the data path, this answers rather than raises."""
        try:
            return await self._call("ping", self._client.ping()) is not None
        except DependencyError:
            return False

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:  # pragma: no cover - shutdown must not raise
            self.logger.warning("cache.redis_close_failed")

    async def _trim(self, namespace: str, order: str) -> None:
        """Evict oldest entries once the namespace exceeds its bound."""
        count = await self._call("zcard", self._client.zcard(order))
        excess = int(count or 0) - self._index_max_entries
        if excess <= 0:
            return
        stale = await self._call("zrange", self._client.zrange(order, 0, excess - 1))
        if not stale:
            return
        keys = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in stale]
        pipeline = self._client.pipeline()
        pipeline.hdel(namespace, *keys)
        pipeline.zrem(order, *keys)
        await self._call("trim", pipeline.execute())

    async def _call(self, operation: str, awaitable: Any) -> Any:
        """Await a Redis command, translating any driver failure into one error type.

        The gateway decides what a failed cache call means for a request. This
        only has to make the failure legible and uniform, because the driver
        raises connection, timeout, and protocol errors that share no base class
        worth catching on.
        """
        try:
            return await awaitable
        except Exception as exc:
            self.logger.warning("cache.redis_failed", operation=operation, error=type(exc).__name__)
            raise DependencyError(
                f"Redis {operation} failed", details={"operation": operation}
            ) from exc


async def open_backend(settings: CacheSettings) -> InMemoryCacheBackend | RedisCacheBackend:
    """Return the best backend available, preferring a reachable Redis.

    Verified with a real round trip at boot rather than assumed from the URL,
    so a misconfigured host is reported once at startup instead of once per
    request for the lifetime of the deployment.
    """
    fallback = InMemoryCacheBackend(index_max_entries=settings.semantic_max_entries)
    if not settings.redis_url:
        return fallback

    client = _connect(settings)
    if client is None:
        return fallback

    backend = RedisCacheBackend(client, settings)
    if not await backend.healthy():
        logger.warning(
            "cache.redis_unavailable",
            reason="ping failed",
            fallback=fallback.name,
        )
        await backend.close()
        return fallback

    logger.info("cache.backend_ready", backend=backend.name)
    return backend


def _connect(settings: CacheSettings) -> Any | None:
    """Build an async Redis client, or None if the driver is not installed.

    `redis` is an optional dependency: the in-memory backend is a complete
    implementation, so a deployment that does not want Redis should not be made
    to install it.
    """
    try:
        from redis.asyncio import Redis
    except ImportError:
        logger.warning(
            "cache.redis_unavailable",
            reason="the redis package is not installed",
            hint="uv sync --extra redis",
        )
        return None

    try:
        return Redis.from_url(
            settings.redis_url,
            socket_timeout=settings.redis_timeout_seconds,
            socket_connect_timeout=settings.redis_timeout_seconds,
        )
    except Exception as exc:
        logger.warning("cache.redis_unavailable", reason=type(exc).__name__)
        return None


def _decode_vector(payload: Any) -> tuple[float, ...] | None:
    """Read one stored embedding, skipping anything unreadable."""
    try:
        values = orjson.loads(payload)
        return tuple(float(value) for value in values)
    except (orjson.JSONDecodeError, TypeError, ValueError):
        return None
