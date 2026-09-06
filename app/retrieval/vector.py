"""Dense vector retrieval: similarity and MMR over the Chroma store."""

from __future__ import annotations

from collections.abc import Sequence

from app.config.settings import ObservabilitySettings, RetrievalSettings
from app.core.observability import RETRIEVER, traced
from app.retrieval.base import ScoredDocument
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.observability import record_query, record_result
from app.retrieval.query import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverCapabilities,
    SearchType,
    VectorMatch,
)
from app.retrieval.retriever import BaseRetriever
from app.retrieval.similarity import maximal_marginal_relevance
from app.services.embeddings import EmbeddingService

_SUPPORTED = (SearchType.SIMILARITY, SearchType.MMR)


class VectorRetriever(BaseRetriever):
    """Embeds the query once, then either ranks by similarity or diversifies by MMR.

    Similarity search asks the store for exactly `top_k` hits. MMR asks for the
    wider `fetch_k` pool *with* its vectors and re-selects locally, which is the
    only way to trade relevance for diversity without a second round trip.
    """

    name = "vector"

    def __init__(
        self,
        store: ChromaVectorStore,
        embeddings: EmbeddingService,
        settings: RetrievalSettings,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        super().__init__(settings, observability)
        self._store = store
        self._embeddings = embeddings

    def capabilities(self) -> RetrieverCapabilities:
        return RetrieverCapabilities(name=self.name, search_types=_SUPPORTED)

    @traced("retriever.vector", run_type=RETRIEVER)
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        resolved = self.prepare(query)
        record_query(resolved, self.observability)
        search_type = resolved.search_type or SearchType.SIMILARITY
        is_mmr = search_type is SearchType.MMR
        top_k = resolved.top_k or self.settings.top_k

        embedding = await self._embeddings.embed_text(resolved.text)
        candidates = await self._store.search_by_vector(
            embedding,
            top_k=(resolved.fetch_k or self.settings.fetch_k) if is_mmr else top_k,
            where=resolved.filters.to_chroma() if resolved.filters else None,
            where_document=_content_clause(resolved.content_contains),
            with_embeddings=is_mmr,
        )

        selected = self._rerank(embedding, candidates, resolved) if is_mmr else candidates
        documents = _above_threshold(selected, resolved.score_threshold)

        self.logger.info(
            "retrieval.vector",
            search_type=str(search_type),
            candidates=len(candidates),
            returned=len(documents),
        )
        result = RetrievalResult(
            documents=documents,
            search_type=search_type,
            retriever=self.name,
            candidates=len(candidates),
            stages=("embed", "mmr" if is_mmr else "similarity", "threshold"),
        )
        record_result(result, self.observability)
        return result

    def _rerank(
        self,
        embedding: Sequence[float],
        candidates: Sequence[VectorMatch],
        query: RetrievalQuery,
    ) -> list[VectorMatch]:
        """Re-select the candidate pool for diversity, preserving store scores.

        Candidates whose vectors the store did not return cannot participate in
        the diversity calculation, so they are dropped rather than silently
        ranked as if they were maximally dissimilar.
        """
        vectors = [match.embedding for match in candidates]
        usable = [index for index, vector in enumerate(vectors) if vector is not None]
        if not usable:
            return list(candidates[: query.top_k or self.settings.top_k])

        chosen = maximal_marginal_relevance(
            embedding,
            [vectors[index] or () for index in usable],
            k=query.top_k or self.settings.top_k,
            lambda_mult=query.mmr_lambda
            if query.mmr_lambda is not None
            else self.settings.mmr_lambda,
        )
        return [candidates[usable[position]] for position in chosen]


def _content_clause(contains: str | None) -> dict[str, str] | None:
    """Translate a substring requirement into Chroma's document filter."""
    return {"$contains": contains} if contains else None


def _above_threshold(
    matches: Sequence[VectorMatch], threshold: float | None
) -> tuple[ScoredDocument, ...]:
    minimum = threshold or 0.0
    return tuple(match.as_scored() for match in matches if match.score >= minimum)
