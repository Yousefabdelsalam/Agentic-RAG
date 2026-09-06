"""The retrieval request and result types.

These are the planner-ready surface of the layer: a caller describes *what* it
wants retrieved — strategy, breadth, filters, whether to compress — and gets back
the documents plus a record of how they were produced. Nothing here knows about
Chroma, OpenAI, or any particular retriever.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from app.models.base import Schema
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.filters import MetadataFilters


class SearchType(StrEnum):
    """Retrieval strategies a caller may ask for."""

    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"


class RetrievalQuery(Schema):
    """A declarative retrieval request.

    Every tuning field is optional: `None` means "use the configured default",
    so a planner can specify only what it actually cares about.
    """

    text: str = Field(min_length=1)
    top_k: int | None = Field(default=None, gt=0)
    search_type: SearchType | None = None
    fetch_k: int | None = Field(default=None, gt=0)
    mmr_lambda: float | None = Field(default=None, ge=0.0, le=1.0)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: MetadataFilters | None = None
    content_contains: str | None = None
    compress: bool | None = None

    def with_defaults(self, **defaults: object) -> RetrievalQuery:
        """Fill unset fields from `defaults`, leaving explicit values untouched."""
        unset = {
            name: value
            for name, value in defaults.items()
            if getattr(self, name, None) is None and value is not None
        }
        return self.model_copy(update=unset)


class VectorMatch(Schema):
    """A store hit, optionally carrying the vector that produced it.

    The embedding travels with the match so MMR and redundancy filtering can run
    without a second round trip to embed what the store already holds.
    """

    document: Document
    score: float
    embedding: tuple[float, ...] | None = None

    def as_scored(self) -> ScoredDocument:
        return ScoredDocument(document=self.document, score=self.score)


class RetrievalResult(Schema):
    """Retrieved documents plus the provenance of the retrieval itself.

    `stages` names each step that ran, in order, which is what makes a result
    explainable to a planner deciding whether to re-query differently.
    """

    documents: tuple[ScoredDocument, ...] = ()
    search_type: SearchType = SearchType.SIMILARITY
    retriever: str = "unknown"
    candidates: int = 0
    stages: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        return not self.documents

    def with_stage(self, stage: str) -> RetrievalResult:
        """Return a copy with `stage` appended to the trace."""
        return self.model_copy(update={"stages": (*self.stages, stage)})


class RetrieverCapabilities(Schema):
    """What a retriever can do, so a caller can choose one without constructing it."""

    name: str
    search_types: tuple[SearchType, ...]
    supports_metadata_filters: bool = True
    supports_content_filter: bool = True
    compresses: bool = False

    def supports(self, search_type: SearchType) -> bool:
        return search_type in self.search_types
