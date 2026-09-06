"""The compression retriever: retrieve wide, then compress down to what is used."""

from __future__ import annotations

from collections.abc import Sequence

from app.config.settings import ObservabilitySettings, RetrievalSettings
from app.core.observability import RETRIEVER, record, traced
from app.retrieval.base import ScoredDocument
from app.retrieval.compressors import (
    BudgetCompressor,
    CompressionContext,
    ContextCompressor,
    RedundancyCompressor,
    ScoreThresholdCompressor,
    SentenceCompressor,
)
from app.retrieval.observability import record_result
from app.retrieval.query import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverCapabilities,
)
from app.retrieval.retriever import BaseRetriever
from app.services.embeddings import EmbeddingService


class CompressionRetriever(BaseRetriever):
    """Wraps another retriever and runs its results through a compression pipeline.

    The base retriever is asked for more documents than the caller wants —
    `compression_fetch_multiplier` times more — because compression only removes
    material. Over-fetching is what lets the pipeline discard weak and redundant
    hits and still return a full page.

    A query may opt out with `compress=False`, in which case the base retriever's
    result is returned untouched; the wrapper stays transparent.
    """

    name = "compression"

    def __init__(
        self,
        base: BaseRetriever,
        embeddings: EmbeddingService,
        settings: RetrievalSettings,
        compressors: Sequence[ContextCompressor] | None = None,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        super().__init__(settings, observability)
        self._base = base
        self._embeddings = embeddings
        self._compressors = tuple(
            compressors if compressors is not None else default_pipeline(embeddings, settings)
        )

    @property
    def base(self) -> BaseRetriever:
        return self._base

    def capabilities(self) -> RetrieverCapabilities:
        return self._base.capabilities().model_copy(
            update={"name": f"{self.name}({self._base.name})", "compresses": True}
        )

    @traced("retriever.compression", run_type=RETRIEVER)
    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        top_k = query.top_k or self.settings.top_k
        if query.compress is False or not self._compressors:
            return await self._base.search(query)

        widened = query.model_copy(
            update={"top_k": top_k * self.settings.compression_fetch_multiplier}
        )
        result = await self._base.search(widened)
        if result.is_empty:
            return result.with_stage("compress:empty")

        context = CompressionContext(
            query=query.text, embedding=tuple(await self._embeddings.embed_text(query.text))
        )
        documents, stages = await self._run_pipeline(context, result.documents)

        before = sum(len(scored.document.content) for scored in result.documents)
        after = sum(len(scored.document.content) for scored in documents[:top_k])
        self.logger.info(
            "retrieval.compressed",
            retrieved=len(result.documents),
            returned=min(len(documents), top_k),
            characters=after,
        )
        record(
            characters_before=before,
            characters_after=after,
            compression_ratio=round(after / before, 4) if before else None,
        )
        compressed = result.model_copy(
            update={
                "documents": tuple(documents[:top_k]),
                "retriever": f"{self.name}({result.retriever})",
                "stages": (*result.stages, *stages),
            }
        )
        record_result(compressed, self.observability)
        return compressed

    async def _run_pipeline(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> tuple[list[ScoredDocument], tuple[str, ...]]:
        """Feed the documents through each stage, recording what each one did.

        A stage that empties the set ends the run: later stages have nothing to
        work on, and the trace records where the material was lost.
        """
        current = list(documents)
        stages: list[str] = []
        for compressor in self._compressors:
            before = len(current)
            current = await compressor.compress(context, current)
            stages.append(f"compress:{compressor.name}:{before}->{len(current)}")
            if not current:
                break
        return current, tuple(stages)


def default_pipeline(
    embeddings: EmbeddingService, settings: RetrievalSettings
) -> tuple[ContextCompressor, ...]:
    """Build the standard funnel: weak hits, then duplicates, then sentences, then budget.

    Ordering is deliberate. Dropping documents first means the expensive
    sentence-level pass embeds only text that survived, and the budget runs last
    so it measures what will actually be returned.
    """
    stages: list[ContextCompressor] = [
        ScoreThresholdCompressor(settings.compression_min_relevance),
        RedundancyCompressor(embeddings, settings.compression_redundancy_threshold),
    ]
    if settings.compression_sentence_level:
        stages.append(SentenceCompressor(embeddings, settings.compression_min_relevance))
    stages.append(BudgetCompressor(settings.compression_max_characters))
    return tuple(stages)
