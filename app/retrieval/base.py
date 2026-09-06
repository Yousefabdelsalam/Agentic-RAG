"""Contracts for the retrieval layer. Implementations land here later."""

from __future__ import annotations

from typing import Protocol

from app.models.base import Schema


class Document(Schema):
    """A retrievable chunk of content with its provenance metadata."""

    id: str
    content: str
    metadata: dict[str, str] = {}


class ScoredDocument(Schema):
    """A document paired with its relevance score for a query."""

    document: Document
    score: float


class Retriever(Protocol):
    """Returns the documents most relevant to a query."""

    async def retrieve(self, query: str, *, top_k: int = 5) -> list[ScoredDocument]: ...


class VectorStore(Protocol):
    """Persists embedded documents and answers similarity queries."""

    async def upsert(self, documents: list[Document]) -> None: ...

    async def search(self, query: str, *, top_k: int = 5) -> list[ScoredDocument]: ...

    async def delete(self, ids: list[str]) -> None: ...
