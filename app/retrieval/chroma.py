"""Chroma-backed vector store.

Implements the `VectorStore` contract: `upsert`/`delete` for ingestion and
`search` for retrieval. Chroma exposes a natively async client over HTTP and a
synchronous one for on-disk persistence, so every collection call goes through
`_invoke`, which awaits the former and offloads the latter to a worker thread.

Retrieval strategies do not live here. The store answers one question — nearest
neighbours to a vector, optionally filtered — and the retrievers in this package
decide how to use the answer.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from typing import Any

import anyio
import chromadb

from app.config.settings import ChromaSettings, ResilienceSettings
from app.core.base import Component
from app.core.exceptions import ConfigurationError, DependencyError
from app.core.logging import get_logger
from app.core.observability import RETRIEVER, record, traced
from app.core.retry import with_retry
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.query import VectorMatch
from app.retrieval.similarity import distance_to_score
from app.services.embeddings import EmbeddingService

_DISTANCE_SPACE = "cosine"
_BASE_INCLUDE = ("documents", "metadatas", "distances")


class ChromaVectorStore(Component):
    """Persists embedded documents in a Chroma collection."""

    def __init__(
        self,
        settings: ChromaSettings,
        embeddings: EmbeddingService,
        resilience: ResilienceSettings | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._embeddings = embeddings
        self._resilience = resilience or ResilienceSettings()
        self._client: Any = None
        self._collection: Any = None
        self._remote = settings.host is not None

    @property
    def collection_name(self) -> str:
        return self._settings.collection

    async def start(self) -> None:
        """Connect to Chroma and ensure the target collection exists."""
        try:
            self._client = await self._connect()
            self._collection = await self._open_collection()
        except ConfigurationError:
            raise
        except Exception as exc:
            raise DependencyError(
                "Could not open the Chroma collection",
                details={"collection": self.collection_name, "remote": self._remote},
            ) from exc
        self.logger.info("chroma.ready", collection=self.collection_name, remote=self._remote)

    async def close(self) -> None:
        self._collection = None
        self._client = None

    async def upsert(self, documents: list[Document]) -> None:
        """Embed and write `documents`, replacing any rows with the same ids."""
        if not documents:
            return
        vectors = await self._embeddings.embed_texts([document.content for document in documents])
        await self._invoke(
            "upsert",
            ids=[document.id for document in documents],
            documents=[document.content for document in documents],
            metadatas=_metadatas(documents),
            embeddings=vectors,
        )
        self.logger.info("chroma.upserted", collection=self.collection_name, count=len(documents))

    async def delete(self, ids: list[str]) -> None:
        """Remove the given ids; unknown ids are ignored by Chroma."""
        if not ids:
            return
        await self._invoke("delete", ids=ids)
        self.logger.info("chroma.deleted", collection=self.collection_name, count=len(ids))

    async def search(self, query: str, *, top_k: int = 5) -> list[ScoredDocument]:
        """Return the `top_k` documents nearest to `query`, most relevant first."""
        matches = await self.search_by_text(query, top_k=top_k)
        return [match.as_scored() for match in matches]

    async def search_by_text(
        self,
        query: str,
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        with_embeddings: bool = False,
    ) -> list[VectorMatch]:
        """Embed `query` and return its nearest neighbours."""
        embedding = await self._embeddings.embed_text(query)
        return await self.search_by_vector(
            embedding,
            top_k=top_k,
            where=where,
            where_document=where_document,
            with_embeddings=with_embeddings,
        )

    @traced("vectorstore.query", run_type=RETRIEVER)
    async def search_by_vector(
        self,
        embedding: Sequence[float],
        *,
        top_k: int = 5,
        where: Mapping[str, Any] | None = None,
        where_document: Mapping[str, Any] | None = None,
        with_embeddings: bool = False,
    ) -> list[VectorMatch]:
        """Return nearest neighbours of `embedding`, honouring any filter clauses.

        `with_embeddings` asks Chroma to return the stored vectors alongside the
        hits, which callers doing MMR or redundancy filtering need and plain
        similarity search should not pay for.
        """
        if top_k <= 0:
            return []
        include = [*_BASE_INCLUDE, "embeddings"] if with_embeddings else list(_BASE_INCLUDE)
        response = await self._invoke(
            "query",
            query_embeddings=[list(embedding)],
            n_results=top_k,
            where=dict(where) if where else None,
            where_document=dict(where_document) if where_document else None,
            include=include,
        )
        matches = _to_matches(response)
        self.logger.debug(
            "chroma.searched",
            collection=self.collection_name,
            requested=top_k,
            returned=len(matches),
            filtered=where is not None or where_document is not None,
        )
        record(
            collection=self.collection_name,
            requested=top_k,
            returned=len(matches),
            metadata_filter=dict(where) if where else None,
            content_filter=dict(where_document) if where_document else None,
            with_embeddings=with_embeddings,
        )
        return matches

    async def count(self) -> int:
        """Return the number of vectors currently stored in the collection."""
        return int(await self._invoke("count"))

    async def healthy(self) -> bool:
        """Report whether the Chroma backend answers a heartbeat."""
        if self._client is None:
            return False
        try:
            await self._heartbeat()
        except Exception:
            self.logger.warning("chroma.unhealthy", collection=self.collection_name)
            return False
        return True

    async def _connect(self) -> Any:
        if self._remote:
            host = self._settings.host
            port = self._settings.port
            if port is None:
                raise ConfigurationError("CHROMA__PORT is required when CHROMA__HOST is set")
            return await chromadb.AsyncHttpClient(host=str(host), port=port)
        return await anyio.to_thread.run_sync(
            functools.partial(chromadb.PersistentClient, path=self._settings.persist_directory)
        )

    async def _open_collection(self) -> Any:
        create = functools.partial(
            self._client.get_or_create_collection,
            name=self.collection_name,
            metadata={"hnsw:space": _DISTANCE_SPACE},
            embedding_function=None,
        )
        return await create() if self._remote else await anyio.to_thread.run_sync(create)

    async def _heartbeat(self) -> None:
        if self._remote:
            await self._client.heartbeat()
        else:
            await anyio.to_thread.run_sync(self._client.heartbeat)

    async def _invoke(self, method: str, **kwargs: Any) -> Any:
        """Call a collection method, awaiting or offloading it as the client requires.

        Retried on transient failure: a store that drops a connection during a
        restart should cost a short backoff, not a failed answer. Chroma's own
        rejections (a bad filter, an unknown collection) arrive as the same
        exception type, so they are wrapped first and classified by `with_retry`
        — which does not retry them.
        """
        collection = self._require_collection()

        async def call() -> Any:
            bound = functools.partial(getattr(collection, method), **kwargs)
            try:
                return await bound() if self._remote else await anyio.to_thread.run_sync(bound)
            except Exception as exc:
                raise DependencyError(
                    f"Chroma {method} failed",
                    details={"collection": self.collection_name},
                ) from exc

        return await with_retry(call, settings=self._resilience, name=f"chroma.{method}")

    def _require_collection(self) -> Any:
        if self._collection is None:
            raise ConfigurationError("Vector store used before start()")
        return self._collection


def _metadatas(documents: list[Document]) -> list[dict[str, str] | None] | None:
    """Map documents to Chroma metadata, which rejects empty dicts but accepts None."""
    rows: list[dict[str, str] | None] = [dict(d.metadata) or None for d in documents]
    return rows if any(row is not None for row in rows) else None


def _to_matches(response: Mapping[str, Any]) -> list[VectorMatch]:
    """Unpack Chroma's parallel result arrays into matches.

    Every field comes back as a list-of-lists keyed by query; this store issues
    one query at a time, so row 0 is the whole answer. Absent keys mean the
    field was not requested.
    """
    ids = _row(response, "ids")
    if not ids:
        return []

    contents = _row(response, "documents")
    metadatas = _row(response, "metadatas")
    distances = _row(response, "distances")
    embeddings = _row(response, "embeddings")

    matches: list[VectorMatch] = []
    for index, identifier in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else None
        content = contents[index] if index < len(contents) else ""
        distance = distances[index] if index < len(distances) else 0.0
        vector = embeddings[index] if index < len(embeddings) else None
        matches.append(
            VectorMatch(
                document=Document(
                    id=str(identifier),
                    content=str(content or ""),
                    metadata={str(k): str(v) for k, v in (metadata or {}).items()},
                ),
                score=distance_to_score(float(distance)),
                embedding=tuple(float(value) for value in vector) if vector is not None else None,
            )
        )
    return matches


def _row(response: Mapping[str, Any], key: str) -> list[Any]:
    """Return the first (and only) result row for `key`, or an empty list."""
    rows = response.get(key)
    if rows is None or len(rows) == 0:
        return []
    row = rows[0]
    return list(row) if row is not None else []
