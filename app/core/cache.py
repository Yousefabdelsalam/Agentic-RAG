"""An async TTL cache with single-flight loading.

Sized for the thing it was built for: repeated query embeddings. The same
question asked twice — by a retrying client, a shared dashboard, or the second
pass of a run the critic sent back — should not be embedded twice.

Two properties make it safe under concurrency:

- entries expire, so a cache cannot serve a stale vector forever after a model
  change;
- concurrent misses on the same key wait for one in-flight load rather than each
  starting their own, which is the difference between one API call and fifty
  when a popular query expires under load.

Values live in the process. A restart empties it, and replicas do not share it —
acceptable for a cost optimisation, not for anything that must be consistent.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import anyio


@dataclass
class _Entry[V]:
    value: V
    expires_at: float


class AsyncTTLCache[K, V]:
    """Bounded, expiring cache safe for concurrent async callers."""

    def __init__(
        self,
        *,
        max_size: int,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[K, _Entry[V]] = OrderedDict()
        self._loading: dict[K, anyio.Event] = {}
        self._lock = anyio.Lock()
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        """A zero-sized cache is a disabled cache, not an error."""
        return self._max_size > 0

    @property
    def size(self) -> int:
        return len(self._entries)

    async def get(self, key: K) -> V | None:
        """Return a live value, or None if absent or expired."""
        async with self._lock:
            return self._read(key)

    async def set(self, key: K, value: V) -> None:
        """Store a value, evicting the least recently used entry if full."""
        if not self.enabled:
            return
        async with self._lock:
            self._write(key, value)

    async def get_or_load(self, key: K, loader: Callable[[], Awaitable[V]]) -> V:
        """Return the cached value, loading it once if several callers miss together.

        A caller that finds a load already in flight waits for it instead of
        starting a duplicate. If that load fails, the waiter retries the load
        itself rather than inheriting an error it cannot attribute.
        """
        if not self.enabled:
            return await loader()

        while True:
            async with self._lock:
                cached = self._read(key)
                if cached is not None:
                    return cached
                in_flight = self._loading.get(key)
                if in_flight is None:
                    self._loading[key] = anyio.Event()
                    break
            await in_flight.wait()

        try:
            value = await loader()
        finally:
            async with self._lock:
                waiters = self._loading.pop(key, None)
            if waiters is not None:
                waiters.set()

        await self.set(key, value)
        return value

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    def _read(self, key: K) -> V | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        if entry.expires_at <= self._clock():
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry.value

    def _write(self, key: K, value: V) -> None:
        self._entries[key] = _Entry(value=value, expires_at=self._clock() + self._ttl)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_size:
            self._entries.popitem(last=False)
