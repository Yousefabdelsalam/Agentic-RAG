"""Exception handlers mapping application errors to the error envelope."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.models.common import ErrorResponse

logger = get_logger(__name__)


def _envelope(request: Request, status_code: int, payload: ErrorResponse) -> JSONResponse:
    # `details` carries arbitrary values (validation error context, upstream
    # payloads), so encode rather than hand raw objects to json.dumps.
    return JSONResponse(status_code=status_code, content=jsonable_encoder(payload))


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    payload = ErrorResponse(
        code=exc.code,
        message=exc.message,
        request_id=_request_id(request),
        details=exc.details,
    )
    return _envelope(request, exc.status_code, payload)


async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    payload = ErrorResponse(
        code=HTTPStatus(exc.status_code).name.lower(),
        message=str(exc.detail),
        request_id=_request_id(request),
    )
    return _envelope(request, exc.status_code, payload)


async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    payload = ErrorResponse(
        code="validation_error",
        message="Request validation failed.",
        request_id=_request_id(request),
        details={"errors": exc.errors()},
    )
    return _envelope(request, HTTPStatus.UNPROCESSABLE_ENTITY.value, payload)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error")
    payload = ErrorResponse(
        code="internal_error",
        message="Internal server error.",
        request_id=_request_id(request),
    )
    return _envelope(request, HTTPStatus.INTERNAL_SERVER_ERROR.value, payload)


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every handler to the application instance."""
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, validation_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_error_handler)
