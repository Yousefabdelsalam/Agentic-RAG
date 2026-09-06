"""The upload endpoint: ingest a PDF into the vector store."""

from __future__ import annotations

import tempfile
from http import HTTPStatus
from pathlib import Path
from typing import Annotated

import anyio
from fastapi import APIRouter, File, UploadFile

from app.api.dependencies import PipelineDep, SettingsDep
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.models.chat import UploadResponse
from app.models.common import ErrorResponse

router = APIRouter(tags=["documents"])
logger = get_logger(__name__)

PDF_SUFFIX = ".pdf"
PDF_MEDIA_TYPE = "application/pdf"
PDF_MAGIC = b"%PDF-"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
_READ_CHUNK = 1024 * 1024


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=HTTPStatus.CREATED,
    summary="Ingest a PDF",
    description=(
        "Uploads a PDF, splits it into chunks, embeds them, and writes them to the "
        "vector store. Ingestion is idempotent: re-uploading an unchanged file "
        "rewrites the same chunk ids rather than duplicating them."
    ),
    responses={
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        HTTPStatus.REQUEST_ENTITY_TOO_LARGE: {"model": ErrorResponse},
        HTTPStatus.BAD_GATEWAY: {"model": ErrorResponse, "description": "Not configured"},
    },
)
async def upload(
    pipeline: PipelineDep,
    settings: SettingsDep,
    file: Annotated[UploadFile, File(description="The PDF to ingest.")],
) -> UploadResponse:
    """Ingest one uploaded PDF and report how many chunks it produced."""
    filename = _validated_name(file)

    # The loader reads from a path, and the upload may be larger than memory, so
    # the body is streamed to a temporary file rather than buffered whole. The
    # directory is removed on the way out whatever happens.
    with tempfile.TemporaryDirectory(prefix="upload-") as workspace:
        target = Path(workspace) / filename
        size = await _spool(file, target)
        await _reject_if_not_a_pdf(target)

        chunks = await pipeline.ingest(str(target))

    logger.info("api.uploaded", filename=filename, bytes=size, chunks=chunks)
    return UploadResponse(filename=filename, chunks_indexed=chunks, bytes_received=size)


def _validated_name(file: UploadFile) -> str:
    """Return a safe basename, rejecting anything that is not a PDF.

    Only the basename survives: an uploaded name is client-controlled, and
    `../../etc/passwd` must not be able to steer where the spooled file lands.
    """
    raw = (file.filename or "").strip()
    if not raw:
        raise ValidationError("A filename is required")

    name = Path(raw).name
    if not name or name in {".", ".."}:
        raise ValidationError("Invalid filename", details={"filename": raw})
    if Path(name).suffix.lower() != PDF_SUFFIX:
        raise ValidationError("Only PDF files can be ingested", details={"filename": name})
    if file.content_type and file.content_type != PDF_MEDIA_TYPE:
        raise ValidationError(
            "Content type must be application/pdf",
            details={"content_type": file.content_type},
        )
    return name


async def _spool(file: UploadFile, target: Path) -> int:
    """Stream the upload to disk, stopping if it exceeds the size limit."""
    size = 0
    async with await anyio.open_file(target, "wb") as sink:
        while chunk := await file.read(_READ_CHUNK):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise _too_large()
            await sink.write(chunk)
    if size == 0:
        raise ValidationError("The uploaded file is empty")
    return size


async def _reject_if_not_a_pdf(target: Path) -> None:
    """Check the file's magic bytes.

    The extension and the declared content type are both client-supplied. This
    reads what was actually sent, so a mislabelled file fails as a 422 here
    rather than as an opaque parser error downstream.
    """
    async with await anyio.open_file(target, "rb") as source:
        header = await source.read(len(PDF_MAGIC))
    if header != PDF_MAGIC:
        raise ValidationError("The uploaded file is not a PDF", details={"header": repr(header)})


def _too_large() -> ValidationError:
    error = ValidationError(
        f"The file exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
        details={"max_bytes": MAX_UPLOAD_BYTES},
    )
    error.status_code = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
    return error
