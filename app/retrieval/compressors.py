"""Context compressors: stages that shrink a retrieved set before it is used.

Compression here is embedding-based and extractive, never generative. Nothing in
this module calls a chat model, so compressing a result set costs one embedding
round trip at most and can never introduce text that was not in the source
documents.

Each stage takes the documents the previous one kept, so a pipeline reads as a
funnel: drop weak hits, drop near-duplicates, trim to the relevant sentences,
then enforce a hard budget.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.models.base import Schema
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.similarity import cosine_similarity
from app.services.embeddings import EmbeddingService

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_MIN_SENTENCE_CHARACTERS = 2


class CompressionContext(Schema):
    """The query a compression pass is compressing *for*.

    The embedding is computed once by the retriever and shared across stages, so
    no stage re-embeds the query.
    """

    query: str
    embedding: tuple[float, ...] = ()


@runtime_checkable
class ContextCompressor(Protocol):
    """A single compression stage."""

    name: str

    async def compress(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]: ...


class ScoreThresholdCompressor:
    """Drops documents the retriever already scored as weakly relevant."""

    name = "threshold"

    def __init__(self, min_relevance: float) -> None:
        self._min_relevance = min_relevance

    async def compress(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        return [scored for scored in documents if scored.score >= self._min_relevance]


class RedundancyCompressor:
    """Drops documents that repeat content already kept.

    Chunk overlap during ingestion guarantees adjacent chunks share text, so a
    top-k window frequently contains the same passage twice. Documents are
    compared in score order, so the higher-scoring copy is the one retained.
    """

    name = "redundancy"

    def __init__(self, embeddings: EmbeddingService, threshold: float) -> None:
        self._embeddings = embeddings
        self._threshold = threshold

    async def compress(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        if len(documents) < 2:
            return list(documents)

        vectors = await self._embeddings.embed_texts(
            [scored.document.content for scored in documents]
        )
        kept: list[ScoredDocument] = []
        kept_vectors: list[Sequence[float]] = []
        for scored, vector in zip(documents, vectors, strict=True):
            if any(
                cosine_similarity(vector, existing) >= self._threshold for existing in kept_vectors
            ):
                continue
            kept.append(scored)
            kept_vectors.append(vector)
        return kept


class SentenceCompressor:
    """Keeps only the sentences of each document that bear on the query.

    Every sentence across every document is embedded in one batched call, then
    each document is rebuilt from the sentences scoring above `min_relevance`.
    A document whose sentences all fall below the bar keeps its single best
    sentence, so a document the retriever chose is never reduced to nothing.
    """

    name = "sentences"

    def __init__(self, embeddings: EmbeddingService, min_relevance: float) -> None:
        self._embeddings = embeddings
        self._min_relevance = min_relevance

    async def compress(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        if not documents or not context.embedding:
            return list(documents)

        split = [_sentences(scored.document.content) for scored in documents]
        flat = [sentence for group in split for sentence in group]
        if not flat:
            return list(documents)

        vectors = await self._embeddings.embed_texts(flat)
        scores = [cosine_similarity(context.embedding, vector) for vector in vectors]

        compressed: list[ScoredDocument] = []
        cursor = 0
        for scored, group in zip(documents, split, strict=True):
            window = scores[cursor : cursor + len(group)]
            cursor += len(group)
            compressed.append(self._rebuild(scored, group, window))
        return compressed

    def _rebuild(
        self, scored: ScoredDocument, sentences: Sequence[str], scores: Sequence[float]
    ) -> ScoredDocument:
        if not sentences:
            return scored
        kept = [
            sentence
            for sentence, score in zip(sentences, scores, strict=True)
            if score >= self._min_relevance
        ]
        if not kept:
            best = max(range(len(sentences)), key=scores.__getitem__)
            kept = [sentences[best]]
        return _replace_content(scored, " ".join(kept))


class BudgetCompressor:
    """Enforces a hard character budget across the whole result set.

    The budget is spent in rank order: documents that no longer fit are dropped
    whole rather than truncated mid-sentence, except the first, which is cut to
    the budget so a single oversized document still yields context.
    """

    name = "budget"

    def __init__(self, max_characters: int) -> None:
        self._max_characters = max_characters

    async def compress(
        self, context: CompressionContext, documents: Sequence[ScoredDocument]
    ) -> list[ScoredDocument]:
        kept: list[ScoredDocument] = []
        remaining = self._max_characters
        for scored in documents:
            length = len(scored.document.content)
            if length <= remaining:
                kept.append(scored)
                remaining -= length
            elif not kept:
                trimmed = scored.document.content[: self._max_characters]
                kept.append(_replace_content(scored, trimmed))
                remaining = 0
            else:
                break
        return kept


def _sentences(content: str) -> list[str]:
    """Split content into sentences, discarding fragments that carry no meaning."""
    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(content))
    return [part for part in parts if len(part) >= _MIN_SENTENCE_CHARACTERS]


def _replace_content(scored: ScoredDocument, content: str) -> ScoredDocument:
    """Return the same hit with rewritten content; id and metadata are provenance."""
    document = Document(
        id=scored.document.id, content=content, metadata=dict(scored.document.metadata)
    )
    return ScoredDocument(document=document, score=scored.score)
