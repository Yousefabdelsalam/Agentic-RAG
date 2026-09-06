"""Chunk provenance: the metadata carried through ingestion and the ids derived from it.

`Document.metadata` is a flat `dict[str, str]`, so the typed models here are the
authoritative shape and `as_mapping` / `from_mapping` are the only places that
serialise it. Ids are derived, never random: re-ingesting an unchanged page
produces the same ids, which makes upserts idempotent.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from app.models.base import Schema

_NON_SLUG = re.compile(r"[^a-z0-9]+")
_DIGEST_LENGTH = 8
_ORDINAL_WIDTH = 4


def slugify(value: str) -> str:
    """Return a lowercase, hyphen-separated form of `value` safe for ids."""
    slug = _NON_SLUG.sub("-", value.lower()).strip("-")
    return slug or "document"


def utc_now() -> datetime:
    """Return the current instant; the single clock the pipeline reads."""
    return datetime.now(UTC)


class PageMetadata(Schema):
    """Provenance of a single source page, assigned by the loader."""

    filename: str
    page: int
    created_at: datetime
    source: str

    @classmethod
    def for_path(cls, path: Path, *, page: int, created_at: datetime) -> Self:
        """Describe page `page` of the document at `path`."""
        return cls(
            filename=path.name,
            page=page,
            created_at=created_at,
            source=str(path.resolve()),
        )

    @classmethod
    def from_mapping(cls, metadata: dict[str, str]) -> Self:
        """Rebuild the typed model from a document's flat metadata."""
        fields = cls.model_fields
        return cls.model_validate({key: metadata[key] for key in fields if key in metadata})

    def as_mapping(self) -> dict[str, str]:
        """Flatten to the string-only mapping `Document.metadata` accepts."""
        return {key: _stringify(value) for key, value in self}

    @property
    def document_id(self) -> str:
        """Semantic id of the page document, e.g. `annual-report-p0003`."""
        return f"{slugify(Path(self.filename).stem)}-p{self.page:0{_ORDINAL_WIDTH}d}"


class ChunkMetadata(PageMetadata):
    """Provenance of a retrieval-sized chunk, assigned by the chunker.

    `chunk` is the zero-based index of the chunk within its page, so a chunk's
    identity depends only on the page it came from.
    """

    chunk: int

    @classmethod
    def for_page(cls, page: PageMetadata, *, chunk: int) -> Self:
        """Extend page provenance with the chunk's position on that page."""
        return cls(**page.model_dump(), chunk=chunk)

    def chunk_id(self, content: str) -> str:
        """Semantic, content-addressed id, e.g. `annual-report-p0003-c0002-1f4b9ac2`.

        The digest suffix makes the id change when the text changes, so edited
        content is written as a new chunk instead of silently replacing an old
        one under a reused id.
        """
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:_DIGEST_LENGTH]
        return f"{self.document_id}-c{self.chunk:0{_ORDINAL_WIDTH}d}-{digest}"


def _stringify(value: Any) -> str:
    return value.isoformat() if isinstance(value, datetime) else str(value)
