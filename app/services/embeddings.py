"""Embedding service: turns text into vectors via the configured OpenAI model."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_openai import OpenAIEmbeddings
from openai import OpenAIError

from app.config.settings import (
    IngestionSettings,
    ObservabilitySettings,
    OpenAISettings,
    ResilienceSettings,
)
from app.core.cache import AsyncTTLCache
from app.core.costs import estimate_tokens
from app.core.exceptions import ConfigurationError, DependencyError
from app.core.observability import LLM, record, record_usage, traced
from app.services.base import Service

_HEALTH_PROBE = "ok"


class EmbeddingService(Service):
    """Async wrapper around the OpenAI embeddings endpoint.

    The client is built on `start()` so a missing API key fails at boot rather
    than halfway through an ingestion run. Batching and retries are delegated to
    the client; callers hand over whole lists of text.

    The endpoint reports no token usage, so traced runs carry an estimate derived
    from input length, flagged as estimated. Without it, embedding spend — which
    dominates a large ingestion — would simply be absent from cost reporting.
    """

    def __init__(
        self,
        settings: OpenAISettings,
        ingestion: IngestionSettings,
        observability: ObservabilitySettings | None = None,
        resilience: ResilienceSettings | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._batch_size = ingestion.embed_batch_size
        self._observability = observability or ObservabilitySettings()
        self._resilience = resilience or ResilienceSettings()
        self._client: OpenAIEmbeddings | None = None
        # Only query embeddings are cached. Document embedding happens once per
        # chunk during ingestion, so caching it would fill memory to serve a hit
        # rate of approximately zero.
        self._query_cache: AsyncTTLCache[str, list[float]] = AsyncTTLCache(
            max_size=self._resilience.embedding_cache_size,
            ttl_seconds=self._resilience.embedding_cache_ttl_seconds,
        )

    @property
    def model(self) -> str:
        return self._settings.embedding_model

    async def start(self) -> None:
        api_key = self._settings.api_key
        if api_key is None:
            raise ConfigurationError(
                "OPENAI__API_KEY is required to embed documents",
                details={"model": self.model},
            )
        self._client = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=api_key,
            openai_api_base=self._settings.base_url,
            request_timeout=self._settings.timeout_seconds,
            max_retries=self._settings.max_retries,
            chunk_size=self._batch_size,
        )
        self.logger.info("embeddings.ready", model=self.model, batch_size=self._batch_size)

    async def close(self) -> None:
        self._client = None

    @traced("embeddings.documents", run_type=LLM)
    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, in the order given."""
        if not texts:
            return []
        client = self._require_client()
        try:
            vectors = await client.aembed_documents(list(texts))
        except OpenAIError as exc:
            raise DependencyError(
                "Embedding request failed", details={"model": self.model, "texts": len(texts)}
            ) from exc
        self._record_usage(list(texts), vectors=len(vectors))
        return vectors

    @traced("embeddings.query", run_type=LLM)
    async def embed_text(self, text: str) -> list[float]:
        """Return the vector for a single text, such as a search query.

        Cached by text and model: the same question asked twice — by a retrying
        client, or by the second pass of a run the critic sent back — costs one
        embedding call, not two. The model is part of the key so changing it
        cannot serve vectors from the old one.
        """
        cached = await self._query_cache.get_or_load(
            f"{self.model}\x00{text}", lambda: self._embed_query(text)
        )
        record(cache_hits=self._query_cache.hits, cache_misses=self._query_cache.misses)
        return cached

    async def _embed_query(self, text: str) -> list[float]:
        client = self._require_client()
        try:
            vector = await client.aembed_query(text)
        except OpenAIError as exc:
            raise DependencyError(
                "Embedding request failed", details={"model": self.model}
            ) from exc
        self._record_usage([text], vectors=1)
        return vector

    def _record_usage(self, texts: list[str], *, vectors: int) -> None:
        tokens = estimate_tokens(texts)
        record(texts=len(texts), vectors=vectors, characters=sum(len(text) for text in texts))
        cost = record_usage(
            self.model,
            input_tokens=tokens,
            prices=self._observability.model_prices,
            estimated=True,
        )
        self.logger.debug(
            "embeddings.usage",
            model=self.model,
            texts=len(texts),
            estimated_tokens=tokens,
            estimated_cost_usd=round(cost, 8),
        )

    async def healthy(self) -> bool:
        """Report whether the service is configured and ready to be called.

        Cheap by design: readiness is polled, and embedding a probe string on
        every poll would bill for information the local state already has. Use
        `probe` for a real round trip.
        """
        return self._client is not None

    async def probe(self) -> bool:
        """Make a real round trip to the embeddings endpoint."""
        if self._client is None:
            return False
        try:
            await self.embed_texts([_HEALTH_PROBE])
        except (DependencyError, ConfigurationError):
            return False
        return True

    def _require_client(self) -> OpenAIEmbeddings:
        if self._client is None:
            raise ConfigurationError("Embedding service used before start()")
        return self._client
