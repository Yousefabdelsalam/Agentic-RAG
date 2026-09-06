from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config.settings import ChromaSettings, RetrievalSettings
from app.core.exceptions import ValidationError
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.compression import CompressionRetriever
from app.retrieval.compressors import (
    BudgetCompressor,
    CompressionContext,
    RedundancyCompressor,
    ScoreThresholdCompressor,
    SentenceCompressor,
)
from app.retrieval.factory import RetrieverFactory
from app.retrieval.filters import FilterOperator, MetadataFilter, MetadataFilters
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.query import (
    RetrievalQuery,
    RetrievalResult,
    RetrieverCapabilities,
    SearchType,
)
from app.retrieval.similarity import (
    cosine_similarity,
    distance_to_score,
    maximal_marginal_relevance,
    reciprocal_rank_fusion,
)
from app.retrieval.vector import VectorRetriever

_VOCABULARY = ("alpha", "beta", "gamma", "delta", "epsilon")


class FakeEmbeddings:
    """Bag-of-words embedder over a fixed vocabulary.

    Real enough for relevance to be meaningful in tests — documents sharing
    vocabulary land near each other — while staying entirely deterministic.
    """

    def __init__(self) -> None:
        self.text_calls = 0
        self.batch_calls = 0

    def _vector(self, text: str) -> list[float]:
        tokens = text.lower().split()
        counts = [
            float(sum(token.strip(".,!?") == word for token in tokens)) for word in _VOCABULARY
        ]
        norm = math.sqrt(sum(value * value for value in counts))
        return [value / norm for value in counts] if norm else [1.0] + [0.0] * (len(counts) - 1)

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.batch_calls += 1
        return [self._vector(text) for text in texts]

    async def embed_text(self, text: str) -> list[float]:
        self.text_calls += 1
        return self._vector(text)


def _settings(**overrides: object) -> RetrievalSettings:
    defaults: dict[str, object] = {"top_k": 3, "fetch_k": 10, "compression_enabled": False}
    return RetrievalSettings(**(defaults | overrides))  # type: ignore[arg-type]


async def _store(tmp_path: Path, embeddings: FakeEmbeddings) -> ChromaVectorStore:
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection="retrieval")
    store = ChromaVectorStore(settings, embeddings)  # type: ignore[arg-type]
    await store.start()
    return store


def _corpus() -> list[Document]:
    return [
        Document(
            id="a1",
            content="alpha alpha alpha",
            metadata={"filename": "a.pdf", "page": "1", "topic": "alpha"},
        ),
        Document(
            id="a2",
            content="alpha alpha alpha alpha",
            metadata={"filename": "a.pdf", "page": "2", "topic": "alpha"},
        ),
        Document(
            id="b1",
            content="beta beta gamma",
            metadata={"filename": "b.pdf", "page": "1", "topic": "beta"},
        ),
        Document(
            id="c1",
            content="delta epsilon delta",
            metadata={"filename": "c.pdf", "page": "1", "topic": "delta"},
        ),
    ]


@pytest.fixture
async def retriever(tmp_path: Path) -> VectorRetriever:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    await store.upsert(_corpus())
    return VectorRetriever(store, embeddings, _settings())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- filters


def test_single_filter_is_emitted_without_a_combinator() -> None:
    filters = MetadataFilters.equals(topic="alpha")
    assert filters.to_chroma() == {"topic": {"$eq": "alpha"}}


def test_multiple_filters_are_combined() -> None:
    filters = MetadataFilters(
        conditions=(
            MetadataFilter(field="topic", value="alpha"),
            MetadataFilter(field="page", value=("1", "2"), operator=FilterOperator.IN),
        ),
        combinator="or",
    )
    assert filters.to_chroma() == {
        "$or": [{"topic": {"$eq": "alpha"}}, {"page": {"$in": ["1", "2"]}}]
    }


def test_empty_filters_produce_no_clause() -> None:
    assert MetadataFilters().to_chroma() is None
    assert not MetadataFilters()


def test_operator_arity_is_validated() -> None:
    with pytest.raises(ValueError):
        MetadataFilter(field="page", value="1", operator=FilterOperator.IN)
    with pytest.raises(ValueError):
        MetadataFilter(field="page", value=("1", "2"), operator=FilterOperator.EQ)


# ------------------------------------------------------------------------ similarity


def test_cosine_similarity_bounds() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_distance_to_score_is_clamped() -> None:
    assert distance_to_score(0.0) == 1.0
    assert distance_to_score(2.0) == 0.0
    assert distance_to_score(-0.001) == 1.0


def test_mmr_prefers_diversity_over_a_near_duplicate() -> None:
    query = [1.0, 0.0]
    candidates = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]

    assert maximal_marginal_relevance(query, candidates, k=2, lambda_mult=1.0) == [0, 1]
    assert maximal_marginal_relevance(query, candidates, k=2, lambda_mult=0.2) == [0, 2]


def test_mmr_handles_degenerate_input() -> None:
    assert maximal_marginal_relevance([1.0], [], k=3, lambda_mult=0.5) == []
    assert maximal_marginal_relevance([1.0], [[1.0]], k=0, lambda_mult=0.5) == []


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=1)
    assert fused["a"] == pytest.approx(fused["b"])

    fused = reciprocal_rank_fusion([["a", "b"], ["a", "b"]], k=1)
    assert fused["a"] > fused["b"]


# -------------------------------------------------------------------- vector search


async def test_similarity_search_ranks_by_relevance(retriever: VectorRetriever) -> None:
    result = await retriever.search(RetrievalQuery(text="alpha"))

    assert result.search_type is SearchType.SIMILARITY
    # a1 and a2 are pure-alpha and tie at the top; ordering between them is
    # arbitrary, so assert the set rather than a spurious order.
    assert {scored.document.id for scored in result.documents[:2]} == {"a1", "a2"}
    assert result.documents[0].score > result.documents[-1].score
    assert result.stages == ("embed", "similarity", "threshold")


async def test_retrieve_satisfies_the_narrow_protocol(retriever: VectorRetriever) -> None:
    documents = await retriever.retrieve("alpha", top_k=2)

    assert len(documents) == 2
    assert all(isinstance(scored, ScoredDocument) for scored in documents)


async def test_metadata_filter_restricts_the_candidate_set(retriever: VectorRetriever) -> None:
    result = await retriever.search(
        RetrievalQuery(text="alpha", filters=MetadataFilters.equals(topic="beta"))
    )

    assert [scored.document.id for scored in result.documents] == ["b1"]


async def test_combined_metadata_filters(retriever: VectorRetriever) -> None:
    result = await retriever.search(
        RetrievalQuery(text="alpha", filters=MetadataFilters.equals(filename="a.pdf", page="2"))
    )

    assert [scored.document.id for scored in result.documents] == ["a2"]


async def test_content_filter_restricts_by_substring(retriever: VectorRetriever) -> None:
    result = await retriever.search(RetrievalQuery(text="alpha", content_contains="gamma"))

    assert [scored.document.id for scored in result.documents] == ["b1"]


async def test_score_threshold_drops_weak_hits(retriever: VectorRetriever) -> None:
    result = await retriever.search(RetrievalQuery(text="alpha", top_k=4, score_threshold=0.99))

    assert {scored.document.id for scored in result.documents} == {"a1", "a2"}


async def test_mmr_search_diversifies_results(retriever: VectorRetriever) -> None:
    similarity = await retriever.search(RetrievalQuery(text="alpha beta", top_k=2))
    mmr = await retriever.search(
        RetrievalQuery(text="alpha beta", top_k=2, search_type=SearchType.MMR, mmr_lambda=0.0)
    )

    assert "mmr" in mmr.stages
    assert mmr.candidates >= len(mmr.documents)
    # The two alpha documents are near-identical, so relevance-only ranking takes
    # both while diversity-only ranking must reach for a different topic.
    assert {scored.document.id for scored in similarity.documents} == {"a1", "a2"}
    assert len({scored.document.metadata["topic"] for scored in mmr.documents}) == 2


async def test_unsupported_search_type_is_rejected(retriever: VectorRetriever) -> None:
    with pytest.raises(ValidationError):
        await retriever.search(RetrievalQuery(text="alpha", search_type=SearchType.HYBRID))


async def test_empty_collection_returns_no_documents(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    retriever = VectorRetriever(store, embeddings, _settings())  # type: ignore[arg-type]

    result = await retriever.search(RetrievalQuery(text="alpha"))

    assert result.is_empty


# --------------------------------------------------------------------- compression


def _scored(identifier: str, content: str, score: float) -> ScoredDocument:
    return ScoredDocument(
        document=Document(id=identifier, content=content, metadata={"topic": "t"}), score=score
    )


async def test_threshold_compressor_drops_weak_documents() -> None:
    context = CompressionContext(query="alpha")
    documents = [_scored("a", "alpha", 0.9), _scored("b", "beta", 0.1)]

    kept = await ScoreThresholdCompressor(0.5).compress(context, documents)

    assert [scored.document.id for scored in kept] == ["a"]


async def test_redundancy_compressor_keeps_the_stronger_duplicate() -> None:
    context = CompressionContext(query="alpha")
    documents = [
        _scored("a", "alpha alpha", 0.9),
        _scored("a-copy", "alpha alpha", 0.8),
        _scored("b", "beta beta", 0.7),
    ]

    kept = await RedundancyCompressor(FakeEmbeddings(), 0.95).compress(context, documents)  # type: ignore[arg-type]

    assert [scored.document.id for scored in kept] == ["a", "b"]


async def test_sentence_compressor_keeps_only_relevant_sentences() -> None:
    embeddings = FakeEmbeddings()
    context = CompressionContext(
        query="alpha", embedding=tuple(await embeddings.embed_text("alpha"))
    )
    documents = [_scored("a", "alpha alpha alpha. beta beta beta. gamma gamma gamma.", 0.9)]

    kept = await SentenceCompressor(embeddings, 0.5).compress(context, documents)  # type: ignore[arg-type]

    assert kept[0].document.content == "alpha alpha alpha."
    assert kept[0].document.id == "a"
    assert kept[0].document.metadata == {"topic": "t"}


async def test_sentence_compressor_never_empties_a_document() -> None:
    embeddings = FakeEmbeddings()
    context = CompressionContext(
        query="alpha", embedding=tuple(await embeddings.embed_text("alpha"))
    )
    documents = [_scored("a", "beta beta. gamma gamma.", 0.9)]

    kept = await SentenceCompressor(embeddings, 0.9).compress(context, documents)  # type: ignore[arg-type]

    assert kept[0].document.content


async def test_budget_compressor_stops_at_the_character_limit() -> None:
    context = CompressionContext(query="alpha")
    documents = [_scored("a", "x" * 6, 0.9), _scored("b", "y" * 6, 0.8)]

    kept = await BudgetCompressor(8).compress(context, documents)

    assert [scored.document.id for scored in kept] == ["a"]


async def test_budget_compressor_trims_a_single_oversized_document() -> None:
    context = CompressionContext(query="alpha")
    kept = await BudgetCompressor(4).compress(context, [_scored("a", "x" * 10, 0.9)])

    assert kept[0].document.content == "xxxx"


async def test_compression_retriever_overfetches_then_trims(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    await store.upsert(_corpus())
    settings = _settings(
        top_k=2,
        compression_enabled=True,
        compression_fetch_multiplier=2,
        compression_min_relevance=0.0,
        compression_sentence_level=False,
    )
    base = VectorRetriever(store, embeddings, settings)  # type: ignore[arg-type]

    result = await CompressionRetriever(base, embeddings, settings).search(  # type: ignore[arg-type]
        RetrievalQuery(text="alpha")
    )

    assert len(result.documents) <= 2
    assert result.retriever == "compression(vector)"
    assert any(stage.startswith("compress:") for stage in result.stages)


async def test_compression_can_be_declined_per_query(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    await store.upsert(_corpus())
    settings = _settings(compression_enabled=True)
    base = VectorRetriever(store, embeddings, settings)  # type: ignore[arg-type]

    result = await CompressionRetriever(base, embeddings, settings).search(  # type: ignore[arg-type]
        RetrievalQuery(text="alpha", compress=False)
    )

    assert result.retriever == "vector"
    assert not any(stage.startswith("compress:") for stage in result.stages)


# -------------------------------------------------------------------------- hybrid


class FakeSparseRetriever:
    """Ranks documents by literal token overlap, the way a lexical index would."""

    def __init__(self, corpus: Sequence[Document]) -> None:
        self._corpus = list(corpus)

    async def search(self, query: RetrievalQuery) -> RetrievalResult:
        terms = set(query.text.lower().split())
        hits = [
            ScoredDocument(document=document, score=float(overlap))
            for document in self._corpus
            if (overlap := len(terms & set(document.content.lower().split())))
        ]
        hits.sort(key=lambda scored: scored.score, reverse=True)
        return RetrievalResult(documents=tuple(hits), retriever="sparse")


async def test_hybrid_degrades_to_dense_without_a_sparse_side(
    retriever: VectorRetriever,
) -> None:
    hybrid = HybridRetriever(retriever, _settings())

    result = await hybrid.search(RetrievalQuery(text="alpha", search_type=SearchType.HYBRID))

    assert hybrid.has_sparse_side is False
    assert result.search_type is SearchType.HYBRID
    assert "fusion:skipped" in result.stages
    assert result.documents


async def test_hybrid_fuses_both_rankings(retriever: VectorRetriever) -> None:
    hybrid = HybridRetriever(retriever, _settings(), FakeSparseRetriever(_corpus()))

    result = await hybrid.search(
        RetrievalQuery(text="delta", top_k=3, search_type=SearchType.HYBRID)
    )

    assert "fusion:rrf" in result.stages
    assert result.documents[0].document.id == "c1"
    assert len({scored.document.id for scored in result.documents}) == len(result.documents)


async def test_hybrid_delegates_non_hybrid_queries(retriever: VectorRetriever) -> None:
    hybrid = HybridRetriever(retriever, _settings(), FakeSparseRetriever(_corpus()))

    result = await hybrid.search(RetrievalQuery(text="alpha", search_type=SearchType.MMR))

    assert result.retriever == "vector"
    assert "mmr" in result.stages


# ------------------------------------------------------------------------- factory


async def test_factory_builds_the_retriever_a_query_asks_for(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    factory = RetrieverFactory(store, embeddings, _settings())  # type: ignore[arg-type]

    assert factory.build(SearchType.SIMILARITY).name == "vector"
    assert factory.build(SearchType.HYBRID).name == "hybrid"
    assert factory.build(SearchType.MMR, compressed=True).name == "compression"
    assert factory.for_query(RetrievalQuery(text="alpha", compress=True)).name == "compression"


async def test_factory_reuses_instances(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    factory = RetrieverFactory(store, embeddings, _settings())  # type: ignore[arg-type]

    assert factory.build(SearchType.SIMILARITY) is factory.build(SearchType.SIMILARITY)
    assert factory.build(SearchType.MMR, compressed=True) is factory.build(
        SearchType.SIMILARITY, compressed=True
    )


async def test_factory_describes_its_strategies_without_running_them(tmp_path: Path) -> None:
    embeddings = FakeEmbeddings()
    store = await _store(tmp_path, embeddings)
    factory = RetrieverFactory(store, embeddings, _settings())  # type: ignore[arg-type]

    described = factory.describe()

    assert all(isinstance(entry, RetrieverCapabilities) for entry in described)
    assert {SearchType.HYBRID} <= {
        search_type for entry in described for search_type in entry.search_types
    }
    assert any(entry.compresses for entry in described)
    assert all(entry.supports_metadata_filters for entry in described)


def test_query_defaults_do_not_override_explicit_values() -> None:
    query = RetrievalQuery(text="alpha", top_k=7)

    resolved = query.with_defaults(top_k=3, fetch_k=20)

    assert resolved.top_k == 7
    assert resolved.fetch_k == 20
