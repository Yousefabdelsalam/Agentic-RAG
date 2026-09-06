"""Retriever construction.

This is the layer's front door. A caller — today application code, later a
planner — describes the retrieval it wants and gets back something satisfying
`BaseRetriever`, without naming a concrete class or knowing whether compression
or fusion is involved.

`describe()` is the introspection half of that contract: a planner can read what
strategies exist and what they support *before* committing to a query, which is
what keeps planning free of hard-coded knowledge about this package.
"""

from __future__ import annotations

from app.config.settings import ObservabilitySettings, RetrievalSettings
from app.core.base import Component
from app.core.logging import get_logger
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.compression import CompressionRetriever
from app.retrieval.hybrid import HybridRetriever, SparseRetriever
from app.retrieval.query import RetrievalQuery, RetrieverCapabilities, SearchType
from app.retrieval.retriever import BaseRetriever
from app.retrieval.vector import VectorRetriever
from app.services.embeddings import EmbeddingService


class RetrieverFactory(Component):
    """Builds and caches the retrievers available to this deployment.

    Retrievers are stateless with respect to a query, so one instance per
    configuration is reused rather than rebuilt per call.
    """

    def __init__(
        self,
        store: ChromaVectorStore,
        embeddings: EmbeddingService,
        settings: RetrievalSettings,
        sparse: SparseRetriever | None = None,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._observability = observability or ObservabilitySettings()
        self._vector = VectorRetriever(store, embeddings, settings, self._observability)
        self._hybrid = HybridRetriever(self._vector, settings, sparse, self._observability)
        self._embeddings = embeddings
        self._compressed: dict[str, CompressionRetriever] = {}

    @property
    def default_search_type(self) -> SearchType:
        return SearchType(self._settings.default_search_type)

    def build(
        self, search_type: SearchType | None = None, *, compressed: bool | None = None
    ) -> BaseRetriever:
        """Return a retriever for `search_type`, wrapped in compression if enabled."""
        strategy = search_type or self.default_search_type
        base = self._hybrid if strategy is SearchType.HYBRID else self._vector
        use_compression = self._settings.compression_enabled if compressed is None else compressed
        return self._compress(base) if use_compression else base

    def for_query(self, query: RetrievalQuery) -> BaseRetriever:
        """Return the retriever that serves `query` as the query itself describes it."""
        return self.build(query.search_type, compressed=query.compress)

    def describe(self) -> tuple[RetrieverCapabilities, ...]:
        """List every strategy this factory can produce."""
        return (
            self._vector.capabilities(),
            self._hybrid.capabilities(),
            self._compress(self._vector).capabilities(),
            self._compress(self._hybrid).capabilities(),
        )

    def _compress(self, base: BaseRetriever) -> CompressionRetriever:
        if base.name not in self._compressed:
            self._compressed[base.name] = CompressionRetriever(
                base, self._embeddings, self._settings, observability=self._observability
            )
        return self._compressed[base.name]
