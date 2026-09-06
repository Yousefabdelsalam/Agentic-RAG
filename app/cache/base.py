"""Contracts and stored records for the answer cache.

A cached answer is only valid under the conditions that produced it, so the
record carries them: which documents were indexed, which model wrote it, and
which prompts it was written from. A lookup that cannot match all three is a
miss, not a hit — the alternative is answering today's question from last
month's corpus and reporting it as fresh.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

CacheType = Literal["exact", "semantic", "none"]

EXACT: CacheType = "exact"
SEMANTIC: CacheType = "semantic"
NONE: CacheType = "none"

#: Frozen like every other domain record, but with pydantic's `model_` guard
#: lifted: `model_version` is a name the cache contract fixes, not a choice.
_RECORD_CONFIG = ConfigDict(
    frozen=True,
    extra="forbid",
    populate_by_name=True,
    protected_namespaces=(),
)


class CacheVersions(BaseModel):
    """The three identities under which a cached answer stays valid."""

    model_config = _RECORD_CONFIG

    knowledge_base_version: str
    model_version: str
    prompt_version: str

    @property
    def fingerprint(self) -> str:
        """A short stable digest of all three, used to namespace keys.

        Folding the versions into the key means an incompatible entry is never
        even read, so the common case costs no comparison. The explicit check in
        `CachedAnswer.matches` still runs, because semantic lookup reaches
        entries by similarity rather than by key.
        """
        joined = "\x00".join((self.knowledge_base_version, self.model_version, self.prompt_version))
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class CachedAnswer(BaseModel):
    """One answer retained for reuse, with everything needed to judge it valid."""

    model_config = _RECORD_CONFIG

    query: str
    normalized_query: str
    answer: str
    embedding: tuple[float, ...] = ()

    citations: tuple[dict[str, Any], ...] = ()
    tools: tuple[dict[str, Any], ...] = ()
    trace: tuple[str, ...] = ()
    revisions: int = 0

    grounded: bool = False
    sufficient_context: bool = False
    confidence: float = 0.0

    knowledge_base_version: str
    model_version: str
    prompt_version: str

    created_at: float
    expires_at: float

    def age_seconds(self, now: float) -> float:
        """How long ago this answer was written, never negative."""
        return max(0.0, now - self.created_at)

    def is_live(self, now: float) -> bool:
        """Whether the entry is still within its TTL."""
        return now < self.expires_at

    def matches(self, versions: CacheVersions) -> bool:
        """Whether this answer was produced under the versions now in force."""
        return (
            self.knowledge_base_version == versions.knowledge_base_version
            and self.model_version == versions.model_version
            and self.prompt_version == versions.prompt_version
        )


class CacheHit(BaseModel):
    """A lookup that found a usable answer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry: CachedAnswer
    cache_type: CacheType
    cache_key: str
    age_seconds: float
    similarity: float = 1.0


@runtime_checkable
class CacheBackend(Protocol):
    """Where cache entries live.

    Deliberately a key/value store plus a small vector index, rather than
    anything richer: this is the whole surface a Redis, in-memory, or future
    backend has to implement, and the gateway holds all the policy.

    Implementations never raise for an unreachable backend on read — a cache
    that fails a request it could only ever have made faster is worse than no
    cache. Writes and reads return None/empty instead, and the gateway counts it.
    """

    @property
    def name(self) -> str:
        """Which backend this is, for logs and the stats endpoint."""
        ...

    async def get(self, key: str) -> str | None:
        """Return the stored payload, or None if absent."""
        ...

    async def set(self, key: str, value: str, *, ttl_seconds: float) -> None:
        """Store a payload. A `ttl_seconds` of zero or less means no expiry."""
        ...

    async def delete(self, key: str) -> None:
        """Remove a key, ignoring one that is already gone."""
        ...

    async def index_add(
        self, namespace: str, key: str, embedding: Sequence[float], *, ttl_seconds: float
    ) -> None:
        """Record `key`'s embedding in the semantic index for `namespace`."""
        ...

    async def index_entries(self, namespace: str) -> list[tuple[str, tuple[float, ...]]]:
        """Return every indexed `(key, embedding)` pair for `namespace`."""
        ...

    async def index_remove(self, namespace: str, key: str) -> None:
        """Drop a key from the semantic index."""
        ...

    async def clear(self) -> None:
        """Discard everything this backend holds for the cache."""
        ...

    async def healthy(self) -> bool:
        """Whether the backend is reachable."""
        ...

    async def close(self) -> None:
        """Release any connection the backend holds."""
        ...
