"""Document loading. Reads a source into one `Document` per page."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import anyio
from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.base import Component
from app.core.exceptions import DependencyError, NotFoundError, ValidationError
from app.core.logging import get_logger
from app.ingestion.metadata import PageMetadata, utc_now
from app.retrieval.base import Document

_PDF_SUFFIX = ".pdf"


class PdfDocumentLoader(Component):
    """Reads a PDF from the local filesystem into page-level documents.

    pypdf is synchronous and CPU-bound, so extraction runs on a worker thread
    and the loader stays awaitable like every other pipeline stage.
    """

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    async def read(self, uri: str) -> list[Document]:
        """Return one document per page of the PDF at `uri`, skipping empty pages."""
        path = self._resolve(uri)
        created_at = utc_now()
        pages = await anyio.to_thread.run_sync(self._extract, path)

        documents = [
            self._to_document(path, page_number, text, created_at)
            for page_number, text in pages
            if text.strip()
        ]
        self.logger.info(
            "ingestion.loaded",
            source=str(path),
            pages=len(pages),
            documents=len(documents),
        )
        return documents

    def _resolve(self, uri: str) -> Path:
        path = Path(uri).expanduser()
        if path.suffix.lower() != _PDF_SUFFIX:
            raise ValidationError(
                "Only PDF sources are supported", details={"uri": uri, "suffix": path.suffix}
            )
        if not path.is_file():
            raise NotFoundError("Source file does not exist", details={"uri": uri})
        return path.resolve()

    def _extract(self, path: Path) -> list[tuple[int, str]]:
        """Extract `(page_number, text)` pairs with 1-based, human page numbers."""
        try:
            reader = PdfReader(path)
            return [
                (number, page.extract_text() or "")
                for number, page in enumerate(reader.pages, start=1)
            ]
        except PdfReadError as exc:
            raise DependencyError("Unreadable PDF source", details={"uri": str(path)}) from exc

    def _to_document(self, path: Path, page: int, text: str, created_at: datetime) -> Document:
        metadata = PageMetadata.for_path(path, page=page, created_at=created_at)
        return Document(id=metadata.document_id, content=text, metadata=metadata.as_mapping())
