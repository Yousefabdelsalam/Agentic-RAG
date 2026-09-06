"""Contracts for the ingestion pipeline. Implementations land here later."""

from __future__ import annotations

from typing import Protocol

from app.retrieval.base import Document


class SourceReader(Protocol):
    """Reads raw content from a source system into documents."""

    async def read(self, uri: str) -> list[Document]: ...


class Chunker(Protocol):
    """Splits documents into retrieval-sized chunks."""

    def chunk(self, documents: list[Document]) -> list[Document]: ...


class IngestionPipeline(Protocol):
    """Orchestrates read -> chunk -> index for a single source."""

    async def ingest(self, uri: str) -> int: ...


class KnowledgeBaseVersioner(Protocol):
    """Tracks the version of the indexed corpus.

    Declared here, where the change originates, and implemented by the cache:
    ingestion is what makes an answer stale, so ingestion is what has to say so.
    The dependency points inward — the pipeline knows there is a version to
    advance, not who is listening.
    """

    async def bump(self, *, reason: str = "") -> str: ...
