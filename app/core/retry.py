"""Retrying calls that leave the process.

Networks fail transiently. A vector store that refuses one connection during a
rolling restart should cost a few hundred milliseconds, not a failed answer.

What is *not* retried matters as much as what is. A rejected filter, a missing
collection, or a malformed request will be rejected identically on the second
attempt, so retrying them turns a fast error into a slow one and multiplies load
on a dependency that is already unhappy. Only errors classified as transient are
retried; everything else propagates on the first attempt.

Backoff is exponential with jitter. Without jitter, every caller that failed
during the same outage retries in lockstep and re-creates the thundering herd
that caused it.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable

import anyio

from app.config.settings import ResilienceSettings
from app.core.exceptions import AppError, DependencyError
from app.core.logging import get_logger

logger = get_logger(__name__)

#: Errors worth a second attempt: the dependency was unreachable or unhappy,
#: not the request malformed.
TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    DependencyError,
    ConnectionError,
    TimeoutError,
    OSError,
)


def is_transient(error: BaseException) -> bool:
    """Whether `error` might succeed on a retry.

    Application errors other than `DependencyError` describe something wrong
    with the request itself, so they are never transient regardless of what they
    inherit from.
    """
    if isinstance(error, AppError):
        return isinstance(error, DependencyError)
    return isinstance(error, TRANSIENT_ERRORS)


async def with_retry[T](
    operation: Callable[[], Awaitable[T]],
    *,
    settings: ResilienceSettings,
    name: str = "operation",
) -> T:
    """Call `operation`, retrying transient failures with backoff.

    Re-raises the final error unchanged once attempts are exhausted, so the
    caller sees the real failure rather than a wrapper describing the retrying.
    """
    attempts = settings.retry_attempts
    for attempt in range(1, attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            if attempt >= attempts or not is_transient(exc):
                raise
            delay = backoff_delay(attempt, settings)
            logger.warning(
                "retry.attempt",
                operation=name,
                attempt=attempt,
                of=attempts,
                delay_seconds=round(delay, 3),
                error=type(exc).__name__,
            )
            await anyio.sleep(delay)

    # Unreachable: the loop either returns or raises on its final attempt.
    raise DependencyError(f"{name} exhausted its retries")


def backoff_delay(attempt: int, settings: ResilienceSettings) -> float:
    """Return the delay before the attempt after `attempt`, jittered and capped."""
    exponential = settings.retry_initial_backoff_seconds * (2 ** (attempt - 1))
    capped = float(min(exponential, settings.retry_max_backoff_seconds))
    if not settings.retry_jitter:
        return capped
    # Jitter spreads retries either side of the capped delay, never negative, so
    # callers that failed together do not retry together.
    spread = capped * settings.retry_jitter
    return max(0.0, capped + random.uniform(-spread, spread))
