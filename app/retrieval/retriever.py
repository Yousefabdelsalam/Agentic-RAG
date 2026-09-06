"""The retriever contract every strategy implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.config.settings import ObservabilitySettings, RetrievalSettings
from app.core.base import Component
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.retrieval.base import ScoredDocument
from app.retrieval.query import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverCapabilities,
    SearchType,
)


class BaseRetriever(Component, ABC):
    """Common behaviour for every retrieval strategy.

    Subclasses implement `search`, which takes the full declarative query and
    returns a traced result. `retrieve` is the narrow form that satisfies the
    `Retriever` protocol, so a caller holding only that contract never has to
    know which strategy it is talking to.

    Retrievers resolve query defaults through `prepare`, so the precedence rule —
    explicit query value beats configured default — lives in exactly one place.
    """

    name: str = "retriever"

    def __init__(
        self,
        settings: RetrievalSettings,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        self.logger = get_logger(type(self).__module__)
        self.settings = settings
        self.observability = observability or ObservabilitySettings()

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[ScoredDocument]:
        """Return the documents most relevant to `query`."""
        result = await self.search(RetrievalQuery(text=query, top_k=top_k))
        return list(result.documents)

    @abstractmethod
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        """Execute `query` and return the documents with a trace of how."""

    @abstractmethod
    def capabilities(self) -> RetrieverCapabilities:
        """Describe what this retriever supports, without executing anything."""

    def prepare(self, query: RetrievalQuery) -> RetrievalQuery:
        """Fill unset query fields from configuration and validate the result."""
        resolved = query.with_defaults(
            top_k=self.settings.top_k,
            search_type=SearchType(self.settings.default_search_type),
            fetch_k=self.settings.fetch_k,
            mmr_lambda=self.settings.mmr_lambda,
            score_threshold=self.settings.score_threshold,
        )
        search_type = resolved.search_type or SearchType.SIMILARITY
        if not self.capabilities().supports(search_type):
            raise ValidationError(
                f"{self.name} cannot serve a {search_type} search",
                details={"retriever": self.name, "search_type": str(search_type)},
            )
        # A candidate pool smaller than the requested page would silently cap
        # results, so widen it rather than under-serving the caller.
        top_k = resolved.top_k or self.settings.top_k
        if (resolved.fetch_k or 0) < top_k:
            resolved = resolved.model_copy(update={"fetch_k": top_k})
        return resolved
