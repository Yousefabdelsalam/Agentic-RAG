"""Recording retrieval runs.

Retriever spans are the ones people actually read when an answer goes wrong, so
what a run records is decided here, once, rather than in each retriever: the
query and its resolved parameters going in, the documents that came back — with
scores and provenance — coming out.

Document text is bounded by `ObservabilitySettings`, because a top-20 retrieval
of full chunks turns a trace into something nobody scrolls through.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.config.settings import ObservabilitySettings
from app.core.observability import record, record_outputs, truncate
from app.retrieval.base import ScoredDocument
from app.retrieval.query import RetrievalQuery, RetrievalResult


def record_query(query: RetrievalQuery, settings: ObservabilitySettings) -> None:
    """Annotate the current retriever run with what was asked for."""
    filters = query.filters.to_chroma() if query.filters else None
    record(
        query=truncate(query.text, settings.max_captured_characters),
        top_k=query.top_k,
        search_type=str(query.search_type) if query.search_type else None,
        fetch_k=query.fetch_k,
        mmr_lambda=query.mmr_lambda,
        score_threshold=query.score_threshold,
        metadata_filter=filters,
        content_contains=query.content_contains,
    )


def record_result(result: RetrievalResult, settings: ObservabilitySettings) -> None:
    """Annotate the current retriever run with what came back.

    Documents are written to the run's outputs under the key LangSmith's
    retriever view expects, so hits render as documents rather than as an opaque
    blob of JSON.
    """
    record(
        retriever=result.retriever,
        documents_returned=len(result.documents),
        candidates_considered=result.candidates,
        stages=list(result.stages),
        top_score=result.documents[0].score if result.documents else None,
    )
    if settings.capture_documents:
        record_outputs(documents=describe_documents(result.documents, settings))


def describe_documents(
    documents: Sequence[ScoredDocument], settings: ObservabilitySettings
) -> list[dict[str, Any]]:
    """Render documents for a trace: provenance, score, and bounded content."""
    return [
        {
            "id": scored.document.id,
            "score": scored.score,
            "metadata": dict(scored.document.metadata),
            "page_content": truncate(scored.document.content, settings.max_captured_characters),
        }
        for scored in documents[: settings.max_captured_documents]
    ]
