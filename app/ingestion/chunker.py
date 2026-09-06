"""Chunking. Splits page documents into retrieval-sized, semantically-identified chunks."""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config.settings import IngestionSettings
from app.core.base import Component
from app.core.logging import get_logger
from app.ingestion.metadata import ChunkMetadata, PageMetadata
from app.retrieval.base import Document


class RecursiveChunker(Component):
    """Recursive character splitter that preserves and extends page provenance.

    `chunk` is synchronous because splitting is pure CPU work with no I/O to
    await; callers that must not block the event loop offload it to a thread.
    """

    def __init__(self, settings: IngestionSettings) -> None:
        self.logger = get_logger(__name__)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            keep_separator=False,
            strip_whitespace=True,
        )

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split every document, carrying its metadata onto each resulting chunk."""
        chunks = [chunk for document in documents for chunk in self._split(document)]
        self.logger.info("ingestion.chunked", documents=len(documents), chunks=len(chunks))
        return chunks

    def _split(self, document: Document) -> list[Document]:
        page = PageMetadata.from_mapping(document.metadata)
        texts = [text for text in self._splitter.split_text(document.content) if text.strip()]
        return [self._to_document(page, index, text) for index, text in enumerate(texts)]

    def _to_document(self, page: PageMetadata, index: int, text: str) -> Document:
        metadata = ChunkMetadata.for_page(page, chunk=index)
        return Document(
            id=metadata.chunk_id(text),
            content=text,
            metadata=metadata.as_mapping(),
        )
