"""Request timeout middleware.

A request that will never finish should stop consuming a worker and tell the
client so, rather than holding a connection open until something upstream gives
up first.

Written as pure ASGI rather than on `BaseHTTPMiddleware`, which is not a style
preference. `BaseHTTPMiddleware` runs the downstream application in a task group
it owns, so cancelling the `await call_next(...)` does not cancel the handler:
the timeout fires, the 504 is built, and then the middleware blocks until the
runaway handler finishes anyway. That produces the right status code with none
of the benefit. Calling the app directly puts the work in this task, where a
cancel scope can actually stop it.

The budget covers time-to-first-byte only. Once the handler starts responding,
the deadline is lifted, so a Server-Sent Events run streams for as long as it
needs: this protects against a handler that never answers, not against an answer
that takes a while to deliver.
"""

from __future__ import annotations

import math
from http import HTTPStatus
from typing import Any

import anyio
import orjson
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.logging import get_logger
from app.models.common import ErrorResponse

TIMEOUT_CODE = "timeout"
TIMEOUT_MESSAGE = "The request took too long and was abandoned."

logger = get_logger(__name__)


class TimeoutMiddleware:
    """Abandon a request whose handler does not begin responding in time."""

    def __init__(self, app: ASGIApp, timeout_seconds: float) -> None:
        self.app = app
        self._timeout = timeout_seconds

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        with anyio.move_on_after(self._timeout) as cancel_scope:

            async def guarded_send(message: Message) -> None:
                nonlocal started
                if message["type"] == "http.response.start":
                    started = True
                    # The handler has answered; streaming its body is not the
                    # thing this middleware is guarding against.
                    cancel_scope.deadline = math.inf
                await send(message)

            await self.app(scope, receive, guarded_send)

        if cancel_scope.cancelled_caught and not started:
            logger.warning(
                "request.timed_out",
                path=scope.get("path", ""),
                timeout_seconds=self._timeout,
            )
            await _send_timeout(scope, send, self._timeout)


async def _send_timeout(scope: Scope, send: Send, timeout_seconds: float) -> None:
    """Emit the same error envelope every other failure uses."""
    payload = ErrorResponse(
        code=TIMEOUT_CODE,
        message=TIMEOUT_MESSAGE,
        request_id=_request_id(scope),
        details={"timeout_seconds": timeout_seconds},
    )
    body = orjson.dumps(payload.model_dump())
    await send(
        {
            "type": "http.response.start",
            "status": HTTPStatus.GATEWAY_TIMEOUT,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def _request_id(scope: Scope) -> str | None:
    """Read the correlation id the request-context middleware put on the scope."""
    state: dict[str, Any] = scope.get("state") or {}
    return state.get("request_id")
