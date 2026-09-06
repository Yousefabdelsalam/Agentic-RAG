"""In-process cache backend.

The development default, and the fallback whenever Redis is configured but
unreachable. Values live in this process: a restart empties the cache and
replicas do not share it, which is acceptable for something whose only job is to
avoid work that can always be redone.

Expiry is checked on read rather than swept on a timer, so an entry that is
never looked up again costs one dictionary slot until eviction reaches it.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import anyio

from app.core.logging import get_logger

#: Applied to the key/value half. The semantic index is bounded separately, by
#: the gateway's configured entry limit.
DEFAULT_MAX_ENTRIES = 4096


@dataclass
class _Entry:
    value: str
    expires_at: float


class InMemoryCacheBackend:
    """Bounded, expiring cache backend safe for concurrent async callers."""

    name = "memory"

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        index_max_entries: int = 500,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.logger = get_logger(__name__)
        self._max_entries = max_entries
        self._index_max_entries = index_max_entries
        self._clock = clock
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._index: dict[str, OrderedDict[str, tuple[float, ...]]] = {}
        self._lock = anyio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.expires_at and entry.expires_at <= self._clock():
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return entry.value

    async def set(self, key: str, value: str, *, ttl_seconds: float) -> None:
        async with self._lock:
            expires_at = self._clock() + ttl_seconds if ttl_seconds > 0 else 0.0
            self._entries[key] = _Entry(value=value, expires_at=expires_at)
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._entries.pop(key, None)

    async def index_add(
        self, namespace: str, key: str, embedding: Sequence[float], *, ttl_seconds: float
    ) -> None:
        async with self._lock:
            entries = self._index.setdefault(namespace, OrderedDict())
            entries[key] = tuple(float(value) for value in embedding)
            entries.move_to_end(key)
            while len(entries) > self._index_max_entries:
                entries.popitem(last=False)

    async def index_entries(self, namespace: str) -> list[tuple[str, tuple[float, ...]]]:
        async with self._lock:
            return list(self._index.get(namespace, {}).items())

    async def index_remove(self, namespace: str, key: str) -> None:
        async with self._lock:
            entries = self._index.get(namespace)
            if entries is not None:
                entries.pop(key, None)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()
            self._index.clear()

    async def healthy(self) -> bool:
        return True

    async def close(self) -> None:
        return None
