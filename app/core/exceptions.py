"""Application error taxonomy and their HTTP translation."""

from __future__ import annotations

from http import HTTPStatus
from typing import Any


class AppError(Exception):
    """Base class for every error raised by application code."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(
        self, message: str | None = None, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(AppError):
    """A required setting is missing or invalid."""

    code = "configuration_error"


class NotFoundError(AppError):
    """The requested resource does not exist."""

    status_code = HTTPStatus.NOT_FOUND
    code = "not_found"


class ValidationError(AppError):
    """The request payload is semantically invalid."""

    status_code = HTTPStatus.UNPROCESSABLE_ENTITY
    code = "validation_error"


class DependencyError(AppError):
    """An upstream dependency failed or is unreachable."""

    status_code = HTTPStatus.BAD_GATEWAY
    code = "dependency_error"
