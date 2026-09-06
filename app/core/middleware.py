"""HTTP middleware: request correlation and access logging."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.utils.ids import new_id

REQUEST_ID_HEADER = "X-Request-ID"

Handler = Callable[[Request], Awaitable[Response]]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, bind it to the log context, and log the outcome."""

    async def dispatch(self, request: Request, call_next: Handler) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or new_id()
        request.state.request_id = request_id

        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        logger = structlog.stdlib.get_logger("app.request")
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request.failed", duration_ms=_elapsed_ms(started))
            raise
        else:
            response.headers[REQUEST_ID_HEADER] = request_id
            logger.info(
                "request.completed",
                status_code=response.status_code,
                duration_ms=_elapsed_ms(started),
            )
            return response
        finally:
            structlog.contextvars.clear_contextvars()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)
