"""Ingestion orchestration: read -> chunk -> index."""

from __future__ import annotations

import anyio

from app.core.base import Component
from app.core.logging import get_logger
from app.ingestion.base import KnowledgeBaseVersioner
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import PdfDocumentLoader
from app.retrieval.base import Document


class DocumentIngestionPipeline(Component):
    """Runs a single source through the pipeline and reports chunks indexed.

    The pipeline owns no domain logic of its own: it sequences the three stages
    and keeps the event loop free while the synchronous chunker runs.
    """

    def __init__(
        self,
        loader: PdfDocumentLoader,
        chunker: RecursiveChunker,
        indexer: Indexer,
        versioner: KnowledgeBaseVersioner | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._loader = loader
        self._chunker = chunker
        self._indexer = indexer
        self._versioner = versioner

    async def ingest(self, uri: str) -> int:
        """Ingest the document at `uri`, returning the number of chunks indexed."""
        self.logger.info("ingestion.started", uri=uri)

        documents = await self._loader.read(uri)
        chunks = await self._chunk(documents)
        indexed = await self._indexer.index(chunks)
        await self._announce(uri, indexed)

        self.logger.info("ingestion.completed", uri=uri, pages=len(documents), chunks=indexed)
        return indexed

    async def _announce(self, uri: str, indexed: int) -> None:
        """Advance the knowledge base version once the index has actually changed.

        After indexing, not before: a version bumped by a run that then failed
        would discard cached answers the corpus still supports. A run that
        indexed nothing changed nothing, so it bumps nothing.
        """
        if self._versioner is None or indexed == 0:
            return
        try:
            await self._versioner.bump(reason=uri)
        except Exception:
            # The documents are indexed either way. A failure here means cached
            # answers may outlive the corpus that produced them until the TTL
            # catches them, which is worth a loud log and not a failed ingestion.
            self.logger.exception("ingestion.version_bump_failed", uri=uri)

    async def _chunk(self, documents: list[Document]) -> list[Document]:
        """Run the synchronous splitter off the event loop."""
        if not documents:
            return []
        return await anyio.to_thread.run_sync(self._chunker.chunk, documents)
