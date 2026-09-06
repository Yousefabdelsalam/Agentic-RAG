"""LangSmith tracing setup.

LangChain, LangGraph, and the langsmith SDK all read their configuration from the
process environment, so this module translates typed settings into the variables
they expect.

Both naming generations are written. `LANGCHAIN_*` is the documented, widely
deployed set and is what the langsmith SDK still honours; `LANGSMITH_*` is the
newer set that takes precedence in current releases. Writing only one leaves
tracing at the mercy of which version is installed, so both are set together and
cleared together.
"""

from __future__ import annotations

import os

from app.config.settings import LangSmithSettings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TRACING_FLAGS = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
_API_KEY_VARS = ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
_PROJECT_VARS = ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT")
_ENDPOINT_VARS = ("LANGCHAIN_ENDPOINT", "LANGSMITH_ENDPOINT")


def configure_tracing(settings: LangSmithSettings) -> None:
    """Enable or disable LangSmith tracing for the current process."""
    if not settings.enabled or settings.api_key is None:
        _set(_TRACING_FLAGS, "false")
        logger.info("tracing.disabled", reason="no_api_key" if settings.enabled else "disabled")
        return

    _set(_TRACING_FLAGS, "true")
    _set(_API_KEY_VARS, settings.api_key.get_secret_value())
    _set(_PROJECT_VARS, settings.project)
    _set(_ENDPOINT_VARS, settings.endpoint)
    logger.info("tracing.enabled", project=settings.project, endpoint=settings.endpoint)


def tracing_enabled() -> bool:
    """Whether tracing is currently switched on for this process."""
    return any(os.environ.get(flag, "").lower() == "true" for flag in _TRACING_FLAGS)


def _set(names: tuple[str, ...], value: str) -> None:
    for name in names:
        os.environ[name] = value
