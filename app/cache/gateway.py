"""The cache gateway: the one place that decides hit, miss, or store.

Lookup order is exact, then semantic, then nothing — cheapest first. An exact
lookup is a hash and a single read; a semantic lookup costs an embedding call
and a scan, which is still an order of magnitude below a graph run but is not
free, so it only runs once the exact key has missed.

Two rules hold everywhere in this module:

- **A cache failure is never a request failure.** Anything the backend or the
  embedding service raises during a lookup is counted, logged, and treated as a
  miss. The caller runs the graph, exactly as it would have without a cache.
- **Only answers the critic stood behind are stored.** An answer that was not
  grounded, ran out of context, failed a tool call, or came back below the
  confidence floor is a bad answer that happens to be finished. Caching it would
  turn one bad answer into every future answer to that question.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from app.cache.base import (
    EXACT,
    SEMANTIC,
    CacheBackend,
    CachedAnswer,
    CacheHit,
    CacheVersions,
)
from app.cache.keys import answer_key, index_namespace, normalize_query
from app.cache.metrics import CacheMetrics
from app.cache.versions import VersionResolver
from app.config.settings import AgentSettings, CacheSettings
from app.core.base import Component
from app.core.logging import get_logger
from app.core.observability import record, traced
from app.retrieval.similarity import cosine_similarity
from app.services.embeddings import EmbeddingService


class CacheGateway(Component):
    """Answers the question "do we already know this?" and records new answers."""

    def __init__(
        self,
        settings: CacheSettings,
        backend: CacheBackend,
        versions: VersionResolver,
        agent: AgentSettings,
        embeddings: EmbeddingService | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._backend = backend
        self._versions = versions
        self._agent = agent
        self._embeddings = embeddings
        self._clock = clock
        self._metrics = CacheMetrics()

    @property
    def enabled(self) -> bool:
        return self._settings.enabled

    @property
    def metrics(self) -> CacheMetrics:
        return self._metrics

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def settings(self) -> CacheSettings:
        return self._settings

    async def versions(self) -> CacheVersions:
        """The version triple a lookup would run under right now."""
        return await self._versions.resolve()

    @property
    def semantic_enabled(self) -> bool:
        """Semantic lookup needs an embedding service; without one it is off."""
        return self._settings.semantic_enabled and self._embeddings is not None

    # ------------------------------------------------------------------ lookup

    @traced("cache.lookup")
    async def lookup(self, query: str) -> CacheHit | None:
        """Return a usable cached answer, or None to run the graph.

        A disabled cache returns None without counting anything: a lookup that
        was never attempted is not a miss, and counting it as one would make the
        hit rate of a disabled cache look like a failing one.
        """
        if not self._settings.enabled:
            return None

        normalized = normalize_query(query)
        if not normalized:
            return None

        try:
            versions = await self._versions.resolve()
            hit = await self._exact(normalized, versions) or await self._semantic(
                normalized, versions
            )
        except Exception as exc:
            # Includes an unreachable backend and a failed embedding call. Both
            # mean "we do not know", which is what a miss means.
            self._metrics.record_backend_error()
            self._metrics.record_miss()
            self.logger.warning("cache.lookup_failed", error=type(exc).__name__)
            record(cache_hit=False, cache_type="none", cache_error=type(exc).__name__)
            return None

        if hit is None:
            self._metrics.record_miss()
            self.logger.info("cache.miss", query=normalized[:120])
            record(cache_hit=False, cache_type="none")
            return None

        self._metrics.record_hit(hit.cache_type)
        self.logger.info(
            "cache.hit",
            cache_type=hit.cache_type,
            cache_key=hit.cache_key,
            cache_age=round(hit.age_seconds, 3),
            similarity=round(hit.similarity, 4),
        )
        record(
            cache_hit=True,
            cache_type=hit.cache_type,
            cache_key=hit.cache_key,
            cache_age=round(hit.age_seconds, 3),
            cache_similarity=round(hit.similarity, 4),
        )
        return hit

    async def _exact(self, normalized: str, versions: CacheVersions) -> CacheHit | None:
        """Look the normalised query up by its deterministic key."""
        key = answer_key(self._settings.namespace, versions, normalized)
        entry = await self._read(key, versions)
        if entry is None:
            return None
        return CacheHit(
            entry=entry,
            cache_type=EXACT,
            cache_key=key,
            age_seconds=entry.age_seconds(self._clock()),
        )

    async def _semantic(self, normalized: str, versions: CacheVersions) -> CacheHit | None:
        """Find the nearest cached question above the similarity threshold.

        Candidates are tried best-first rather than best-only: the closest match
        may have expired out of the key/value half while its vector lingers in
        the index, and the second-closest can still be a legitimate hit.
        """
        if self._embeddings is None or not self._settings.semantic_enabled:
            return None

        namespace = index_namespace(self._settings.namespace, versions)
        indexed = await self._backend.index_entries(namespace)
        if not indexed:
            return None

        embedding = await self._embeddings.embed_text(normalized)
        ranked = _ranked_candidates(
            embedding, indexed, threshold=self._settings.similarity_threshold
        )

        for key, similarity in ranked:
            entry = await self._read(key, versions)
            if entry is not None:
                return CacheHit(
                    entry=entry,
                    cache_type=SEMANTIC,
                    cache_key=key,
                    age_seconds=entry.age_seconds(self._clock()),
                    similarity=similarity,
                )
            # The vector outlived the answer it pointed at; drop it so the next
            # lookup does not pay to score it again.
            await self._backend.index_remove(namespace, key)
        return None

    async def _read(self, key: str, versions: CacheVersions) -> CachedAnswer | None:
        """Read and validate one entry, discarding anything unusable.

        Three things disqualify an entry, and all three are silent by design:
        an unreadable payload from an older schema, expiry, and a version triple
        that no longer matches. Expired and mismatched entries are deleted on
        the way past, so a corpus that has moved on cleans itself up as it is
        read rather than accumulating until something sweeps it.
        """
        raw = await self._backend.get(key)
        if raw is None:
            return None

        try:
            entry = CachedAnswer.model_validate_json(raw)
        except ValidationError:
            self.logger.warning("cache.entry_unreadable", cache_key=key)
            await self._backend.delete(key)
            return None

        if not entry.is_live(self._clock()):
            self.logger.debug("cache.entry_expired", cache_key=key)
            await self._backend.delete(key)
            return None

        if not entry.matches(versions):
            self.logger.info(
                "cache.version_mismatch",
                cache_key=key,
                stored_knowledge_base=entry.knowledge_base_version,
                current_knowledge_base=versions.knowledge_base_version,
                stored_model=entry.model_version,
                current_model=versions.model_version,
                stored_prompt=entry.prompt_version,
                current_prompt=versions.prompt_version,
            )
            await self._backend.delete(key)
            return None

        return entry

    # ------------------------------------------------------------------- store

    @traced("cache.store")
    async def store(self, query: str, answer: str, metadata: dict[str, Any]) -> bool:
        """Retain an answer for reuse, if it is one worth reusing.

        Returns whether it was stored, so the caller can report the decision
        without inspecting the cache.
        """
        if not self._settings.enabled:
            return False

        normalized = normalize_query(query)
        if not normalized:
            return False

        refusal = self._refuse(answer, metadata)
        if refusal is not None:
            self.logger.info("cache.not_stored", reason=refusal, query=normalized[:120])
            record(cache_stored=False, cache_skip_reason=refusal)
            return False

        try:
            versions = await self._versions.resolve()
            embedding = await self._embedding_for(normalized)
            entry = self._entry(normalized, query, answer, metadata, versions, embedding)
            key = answer_key(self._settings.namespace, versions, normalized)
            await self._backend.set(
                key, entry.model_dump_json(), ttl_seconds=self._settings.ttl_seconds
            )
            if embedding:
                await self._backend.index_add(
                    index_namespace(self._settings.namespace, versions),
                    key,
                    embedding,
                    ttl_seconds=self._settings.ttl_seconds,
                )
        except Exception as exc:
            self._metrics.record_backend_error()
            self.logger.warning("cache.store_failed", error=type(exc).__name__)
            record(cache_stored=False, cache_skip_reason="backend_error")
            return False

        self._metrics.record_write()
        self.logger.info("cache.stored", cache_key=key, ttl_seconds=self._settings.ttl_seconds)
        record(cache_stored=True, cache_key=key)
        return True

    def _refuse(self, answer: str, metadata: dict[str, Any]) -> str | None:
        """Name the reason this answer must not be cached, or None to cache it."""
        if not answer.strip():
            return "empty_answer"

        critique = metadata.get("critique")
        if not isinstance(critique, dict):
            # No verdict means nothing checked this answer against its sources.
            return "no_critique"
        if critique.get("sufficient_context") is not True:
            return "insufficient_context"
        if critique.get("grounded") is not True:
            return "not_grounded"

        confidence = float(critique.get("confidence") or 0.0)
        if confidence < self._agent.min_confidence:
            return "low_confidence"

        tools = metadata.get("tools") or []
        if any(not tool.get("ok", True) for tool in tools if isinstance(tool, dict)):
            return "failed_tool_call"

        return None

    async def _embedding_for(self, normalized: str) -> tuple[float, ...]:
        """Embed the normalised query, or return nothing if semantics are off."""
        if self._embeddings is None or not self._settings.semantic_enabled:
            return ()
        return tuple(await self._embeddings.embed_text(normalized))

    def _entry(
        self,
        normalized: str,
        query: str,
        answer: str,
        metadata: dict[str, Any],
        versions: CacheVersions,
        embedding: tuple[float, ...],
    ) -> CachedAnswer:
        critique: dict[str, Any] = metadata.get("critique") or {}
        now = self._clock()
        return CachedAnswer(
            query=query,
            normalized_query=normalized,
            answer=answer,
            embedding=embedding,
            citations=tuple(metadata.get("citations") or ()),
            tools=tuple(metadata.get("tools") or ()),
            trace=tuple(metadata.get("trace") or ()),
            revisions=int(metadata.get("revisions") or 0),
            grounded=bool(critique.get("grounded")),
            sufficient_context=bool(critique.get("sufficient_context")),
            confidence=float(critique.get("confidence") or 0.0),
            knowledge_base_version=versions.knowledge_base_version,
            model_version=versions.model_version,
            prompt_version=versions.prompt_version,
            created_at=now,
            expires_at=now + self._settings.ttl_seconds,
        )

    # --------------------------------------------------------------- lifecycle

    async def clear(self) -> None:
        """Discard every cached answer."""
        await self._backend.clear()
        self.logger.info("cache.cleared", backend=self._backend.name)

    async def healthy(self) -> bool:
        return await self._backend.healthy()

    async def close(self) -> None:
        await self._backend.close()


def _ranked_candidates(
    embedding: list[float],
    indexed: list[tuple[str, tuple[float, ...]]],
    *,
    threshold: float,
) -> list[tuple[str, float]]:
    """Score every indexed vector against the query, best first.

    Vectors of a different width are skipped rather than raising: they belong to
    a superseded embedding model whose entries have not yet aged out, and one
    stale dimension must not fail the lookup.
    """
    scored: list[tuple[str, float]] = []
    for key, vector in indexed:
        if len(vector) != len(embedding):
            continue
        similarity = cosine_similarity(embedding, vector)
        if similarity >= threshold:
            scored.append((key, similarity))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored
