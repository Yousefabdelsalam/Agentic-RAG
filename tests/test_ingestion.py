from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from app.config.settings import ChromaSettings, IngestionSettings
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import PdfDocumentLoader
from app.ingestion.metadata import ChunkMetadata, PageMetadata, utc_now
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval.base import Document
from app.retrieval.chroma import ChromaVectorStore

_METADATA_KEYS = {"filename", "page", "chunk", "created_at", "source"}


def _pdf_bytes(pages: Sequence[str]) -> bytes:
    """Build a minimal multi-page PDF with one text line per page."""
    objects: list[bytes] = []
    page_ids = [4 + index * 2 for index in range(len(pages))]

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = b" ".join(b"%d 0 R" % page_id for page_id in page_ids)
    objects.append(b"<< /Type /Pages /Kids [%s] /Count %d >>" % (kids, len(pages)))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for index, text in enumerate(pages):
        stream = b"BT /F1 12 Tf 72 720 Td (%s) Tj ET" % text.encode("ascii")
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 3 0 R >> >> /Contents %d 0 R >>" % (page_ids[index] + 1)
        )
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)

    xref_at = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_at,
    )
    return bytes(out)


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "Quarterly Report.pdf"
    path.write_bytes(_pdf_bytes(["Alpha page one content", "Beta page two content"]))
    return path.resolve()


class StubEmbeddingService:
    """Deterministic stand-in for the OpenAI-backed embedding service."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0, 0.0] for text in texts]


def _store(tmp_path: Path, embeddings: StubEmbeddingService) -> ChromaVectorStore:
    settings = ChromaSettings(persist_directory=str(tmp_path / "chroma"), collection="test")
    return ChromaVectorStore(settings, embeddings)  # type: ignore[arg-type]


async def test_loader_emits_one_document_per_page(pdf_path: Path) -> None:
    documents = await PdfDocumentLoader().read(str(pdf_path))

    assert len(documents) == 2
    assert "Alpha page one" in documents[0].content
    assert [document.metadata["page"] for document in documents] == ["1", "2"]
    assert documents[0].metadata["filename"] == "Quarterly Report.pdf"
    assert documents[0].metadata["source"] == str(pdf_path)
    assert documents[0].id == "quarterly-report-p0001"


async def test_loader_rejects_non_pdf(tmp_path: Path) -> None:
    from app.core.exceptions import ValidationError

    notes = tmp_path / "notes.txt"
    notes.write_text("not a pdf")
    with pytest.raises(ValidationError):
        await PdfDocumentLoader().read(str(notes))


def test_chunker_carries_metadata_and_semantic_ids() -> None:
    page = PageMetadata.for_path(Path("Quarterly Report.pdf"), page=3, created_at=utc_now())
    document = Document(id=page.document_id, content="word " * 400, metadata=page.as_mapping())

    chunks = RecursiveChunker(IngestionSettings(chunk_size=200, chunk_overlap=20)).chunk([document])

    assert len(chunks) > 1
    for index, chunk in enumerate(chunks):
        assert set(chunk.metadata) == _METADATA_KEYS
        assert chunk.metadata["page"] == "3"
        assert chunk.metadata["chunk"] == str(index)
        assert chunk.id.startswith(f"quarterly-report-p0003-c{index:04d}-")


def test_chunk_ids_are_deterministic_and_content_addressed() -> None:
    created_at = utc_now()
    metadata = ChunkMetadata.for_page(
        PageMetadata.for_path(Path("doc.pdf"), page=1, created_at=created_at), chunk=0
    )

    assert metadata.chunk_id("same text") == metadata.chunk_id("same text")
    assert metadata.chunk_id("same text") != metadata.chunk_id("other text")


async def test_indexer_batches_upserts(tmp_path: Path) -> None:
    embeddings = StubEmbeddingService()
    store = _store(tmp_path, embeddings)
    await store.start()
    try:
        chunks = [
            Document(id=f"chunk-{index}", content=f"content {index}", metadata={})
            for index in range(5)
        ]
        indexed = await Indexer(store, IngestionSettings(upsert_batch_size=2)).index(chunks)

        assert indexed == 5
        assert await store.count() == 5
        assert sorted(len(call) for call in embeddings.calls) == [1, 2, 2]
    finally:
        await store.close()


async def test_pipeline_ingests_pdf_into_chroma(tmp_path: Path, pdf_path: Path) -> None:
    embeddings = StubEmbeddingService()
    store = _store(tmp_path, embeddings)
    await store.start()
    try:
        settings = IngestionSettings(chunk_size=200, chunk_overlap=20)
        pipeline = DocumentIngestionPipeline(
            PdfDocumentLoader(), RecursiveChunker(settings), Indexer(store, settings)
        )

        indexed = await pipeline.ingest(str(pdf_path))

        assert indexed == 2
        assert await store.count() == 2
        assert await pipeline.ingest(str(pdf_path)) == 2
        assert await store.count() == 2, "re-ingesting the same file must not duplicate rows"
    finally:
        await store.close()


async def test_store_healthy_reports_backend_state(tmp_path: Path) -> None:
    store = _store(tmp_path, StubEmbeddingService())

    assert await store.healthy() is False
    await store.start()
    try:
        assert await store.healthy() is True
    finally:
        await store.close()
