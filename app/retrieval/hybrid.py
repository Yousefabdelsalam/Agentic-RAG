"""Hybrid retrieval: the interface for fusing a dense and a sparse retriever.

No sparse backend ships with this layer — Chroma has no lexical index, and
choosing one (BM25, an external search engine) is a deployment decision. What is
defined here is the contract a sparse retriever must satisfy and the fusion that
combines it with dense results, so adding one is a registration rather than a
change to any retriever.

With no sparse side supplied the retriever degrades to dense-only and says so in
the result trace, rather than pretending a fusion happened.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.config.settings import ObservabilitySettings, RetrievalSettings
from app.core.observability import RETRIEVER, traced
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.observability import record_query, record_result
from app.retrieval.query import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverCapabilities,
    SearchType,
)
from app.retrieval.retriever import BaseRetriever
from app.retrieval.similarity import reciprocal_rank_fusion


@runtime_checkable
class SparseRetriever(Protocol):
    """A lexical retriever that ranks documents by term overlap with the query."""

    async def search(self, query: RetrievalQuery) -> RetrievalResult: ...


class HybridRetriever(BaseRetriever):
    """Fuses dense and sparse rankings with reciprocal rank fusion.

    RRF is used rather than a weighted sum of scores because the two sides are
    not on a comparable scale; only their orderings are meaningful together.
    """

    name = "hybrid"

    def __init__(
        self,
        dense: BaseRetriever,
        settings: RetrievalSettings,
        sparse: SparseRetriever | None = None,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        super().__init__(settings, observability)
        self._dense = dense
        self._sparse = sparse

    @property
    def has_sparse_side(self) -> bool:
        return self._sparse is not None

    def capabilities(self) -> RetrieverCapabilities:
        return RetrieverCapabilities(
            name=self.name,
            search_types=(SearchType.HYBRID, SearchType.SIMILARITY, SearchType.MMR),
        )

    @traced("retriever.hybrid", run_type=RETRIEVER)
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        resolved = self.prepare(query)
        record_query(resolved, self.observability)
        if resolved.search_type is not SearchType.HYBRID:
            return await self._dense.search(resolved)

        # The dense side runs its own default strategy; hybrid describes how the
        # rankings are combined, not how either side is produced.
        dense = await self._dense.search(resolved.model_copy(update={"search_type": None}))
        if self._sparse is None:
            return dense.model_copy(
                update={
                    "search_type": SearchType.HYBRID,
                    "retriever": self.name,
                    "stages": (*dense.stages, "fusion:skipped"),
                }
            )

        sparse = await self._sparse.search(resolved.model_copy(update={"search_type": None}))
        documents = self._fuse(dense.documents, sparse.documents, resolved)

        self.logger.info(
            "retrieval.hybrid",
            dense=len(dense.documents),
            sparse=len(sparse.documents),
            returned=len(documents),
        )
        result = RetrievalResult(
            documents=documents,
            search_type=SearchType.HYBRID,
            retriever=self.name,
            candidates=len(dense.documents) + len(sparse.documents),
            stages=(*dense.stages, "sparse", "fusion:rrf"),
        )
        record_result(result, self.observability)
        return result

    def _fuse(
        self,
        dense: Sequence[ScoredDocument],
        sparse: Sequence[ScoredDocument],
        query: RetrievalQuery,
    ) -> tuple[ScoredDocument, ...]:
        by_id: dict[str, Document] = {
            scored.document.id: scored.document for scored in (*sparse, *dense)
        }
        fused = reciprocal_rank_fusion(
            [[scored.document.id for scored in dense], [scored.document.id for scored in sparse]],
            weights=[self.settings.dense_weight, 1.0 - self.settings.dense_weight],
            k=self.settings.rrf_k,
        )
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        top_k = query.top_k or self.settings.top_k
        return tuple(
            ScoredDocument(document=by_id[identifier], score=score)
            for identifier, score in ranked[:top_k]
        )
