"""Cross-cutting API payloads: errors and health."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from app.models.base import ResponseSchema


class ErrorResponse(ResponseSchema):
    """Uniform error envelope returned for every failed request."""

    code: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class HealthResponse(ResponseSchema):
    """Liveness payload."""

    status: Literal["ok"] = "ok"
    app: str
    environment: str
    version: str


class ReadinessResponse(ResponseSchema):
    """Readiness payload with per-dependency status."""

    ready: bool
    dependencies: dict[str, bool] = Field(default_factory=dict)
