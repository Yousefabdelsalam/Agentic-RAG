"""Indexing. Writes chunks to the vector store in bounded, concurrent batches."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import anyio

from app.config.settings import IngestionSettings
from app.core.base import Component
from app.core.logging import get_logger
from app.retrieval.base import Document
from app.retrieval.chroma import ChromaVectorStore


class Indexer(Component):
    """Batches chunks and upserts them, embedding each batch as it goes.

    Concurrency is capped so a large document cannot open an unbounded number of
    in-flight embedding requests; batches are independent, so a failure in one
    cancels the run rather than leaving a partially embedded batch behind.
    """

    def __init__(self, store: ChromaVectorStore, settings: IngestionSettings) -> None:
        self.logger = get_logger(__name__)
        self._store = store
        self._batch_size = settings.upsert_batch_size
        self._limit = anyio.CapacityLimiter(settings.max_concurrent_batches)

    async def index(self, chunks: Sequence[Document]) -> int:
        """Write every chunk to the store and return how many were written."""
        if not chunks:
            return 0

        batches = list(_batched(chunks, self._batch_size))
        async with anyio.create_task_group() as group:
            for batch in batches:
                group.start_soon(self._index_batch, batch)

        self.logger.info("ingestion.indexed", chunks=len(chunks), batches=len(batches))
        return len(chunks)

    async def _index_batch(self, batch: list[Document]) -> None:
        async with self._limit:
            await self._store.upsert(batch)


def _batched(documents: Sequence[Document], size: int) -> Iterator[list[Document]]:
    for start in range(0, len(documents), size):
        yield list(documents[start : start + size])
