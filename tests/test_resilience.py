from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from fastapi.responses import StreamingResponse
from httpx import ASGITransport, AsyncClient

from app.config.settings import ResilienceSettings, Settings
from app.core.cache import AsyncTTLCache
from app.core.exceptions import DependencyError, NotFoundError, ValidationError
from app.core.retry import backoff_delay, is_transient, with_retry
from app.core.timeout import TIMEOUT_CODE
from app.main import create_app


def _settings(**overrides: Any) -> ResilienceSettings:
    defaults: dict[str, Any] = {
        "retry_attempts": 3,
        "retry_initial_backoff_seconds": 0.001,
        "retry_max_backoff_seconds": 0.002,
        "retry_jitter": 0.0,
    }
    return ResilienceSettings(**(defaults | overrides))


# ------------------------------------------------------------------------------ retry


async def test_a_successful_call_is_not_retried() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    assert await with_retry(operation, settings=_settings()) == "ok"
    assert calls == 1


async def test_a_transient_failure_is_retried_until_it_succeeds() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise DependencyError("store unreachable")
        return "ok"

    assert await with_retry(operation, settings=_settings()) == "ok"
    assert calls == 3


async def test_retries_are_bounded_and_the_real_error_survives() -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise DependencyError("store unreachable")

    with pytest.raises(DependencyError, match="store unreachable"):
        await with_retry(operation, settings=_settings(retry_attempts=3))

    assert calls == 3


@pytest.mark.parametrize(
    "error",
    [ValidationError("bad filter"), NotFoundError("no such collection"), ValueError("bug")],
)
async def test_a_permanent_failure_is_not_retried(error: Exception) -> None:
    calls = 0

    async def operation() -> str:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        await with_retry(operation, settings=_settings())

    # Retrying a rejected request turns a fast error into a slow one.
    assert calls == 1


def test_transient_classification() -> None:
    assert is_transient(DependencyError("unreachable"))
    assert is_transient(ConnectionError())
    assert is_transient(TimeoutError())
    assert not is_transient(ValidationError("bad input"))
    assert not is_transient(NotFoundError("missing"))
    assert not is_transient(ValueError("bug"))


def test_backoff_grows_exponentially_and_is_capped() -> None:
    settings = ResilienceSettings(
        retry_initial_backoff_seconds=1.0, retry_max_backoff_seconds=4.0, retry_jitter=0.0
    )

    assert [backoff_delay(attempt, settings) for attempt in (1, 2, 3, 4)] == [1.0, 2.0, 4.0, 4.0]


def test_jitter_spreads_delays_without_going_negative() -> None:
    settings = ResilienceSettings(
        retry_initial_backoff_seconds=1.0, retry_max_backoff_seconds=1.0, retry_jitter=1.0
    )

    delays = {backoff_delay(1, settings) for _ in range(50)}

    assert len(delays) > 1
    assert all(0.0 <= delay <= 2.0 for delay in delays)


# ------------------------------------------------------------------------------ cache


def _cache(**overrides: Any) -> AsyncTTLCache[str, int]:
    defaults: dict[str, Any] = {"max_size": 3, "ttl_seconds": 60.0}
    return AsyncTTLCache(**(defaults | overrides))


async def test_a_value_survives_a_round_trip() -> None:
    cache = _cache()
    await cache.set("k", 1)

    assert await cache.get("k") == 1


async def test_a_missing_key_reads_as_none() -> None:
    assert await _cache().get("absent") is None


async def test_entries_expire() -> None:
    now = 0.0
    cache: AsyncTTLCache[str, int] = AsyncTTLCache(max_size=3, ttl_seconds=10.0, clock=lambda: now)
    await cache.set("k", 1)

    now = 9.0
    assert await cache.get("k") == 1

    now = 11.0
    assert await cache.get("k") is None


async def test_the_least_recently_used_entry_is_evicted() -> None:
    cache = _cache(max_size=2)
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.get("a")  # "a" becomes the most recent
    await cache.set("c", 3)

    assert await cache.get("b") is None
    assert await cache.get("a") == 1
    assert await cache.get("c") == 3


async def test_get_or_load_calls_the_loader_once_per_key() -> None:
    cache = _cache()
    calls = 0

    async def loader() -> int:
        nonlocal calls
        calls += 1
        return 42

    assert await cache.get_or_load("k", loader) == 42
    assert await cache.get_or_load("k", loader) == 42
    assert calls == 1


async def test_concurrent_misses_share_one_load() -> None:
    cache = _cache()
    calls = 0

    async def loader() -> int:
        nonlocal calls
        calls += 1
        await anyio.sleep(0.05)
        return 42

    results: list[int] = []

    async def fetch() -> None:
        results.append(await cache.get_or_load("k", loader))

    async with anyio.create_task_group() as group:
        for _ in range(10):
            group.start_soon(fetch)

    # Ten concurrent misses on the same key are one API call, not ten.
    assert calls == 1
    assert results == [42] * 10


async def test_a_failed_load_is_not_cached() -> None:
    cache = _cache()
    calls = 0

    async def loader() -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise DependencyError("upstream down")
        return 7

    with pytest.raises(DependencyError):
        await cache.get_or_load("k", loader)

    assert await cache.get_or_load("k", loader) == 7


async def test_a_zero_sized_cache_is_transparent() -> None:
    cache = _cache(max_size=0)
    calls = 0

    async def loader() -> int:
        nonlocal calls
        calls += 1
        return 1

    await cache.get_or_load("k", loader)
    await cache.get_or_load("k", loader)

    assert cache.enabled is False
    assert calls == 2


async def test_hits_and_misses_are_counted() -> None:
    cache = _cache()
    await cache.get("absent")
    await cache.set("k", 1)
    await cache.get("k")

    assert (cache.hits, cache.misses) == (1, 1)


# ---------------------------------------------------------------------------- timeout


@pytest.fixture
async def slow_api() -> AsyncIterator[AsyncClient]:
    """An app with handlers that are slow in two different ways."""
    settings = Settings(environment="local", resilience={"request_timeout_seconds": 0.05})  # type: ignore[arg-type]
    app = create_app(settings)

    @app.get("/api/v1/slow")
    async def slow() -> dict[str, str]:
        await anyio.sleep(5)
        return {"status": "never reached"}

    @app.get("/api/v1/slow-stream")
    async def slow_stream() -> StreamingResponse:
        async def body() -> AsyncIterator[bytes]:
            for index in range(3):
                await anyio.sleep(0.05)
                yield f"chunk {index}\n".encode()

        return StreamingResponse(body(), media_type="text/event-stream")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            yield client


async def test_a_slow_handler_is_abandoned_with_an_envelope(slow_api: AsyncClient) -> None:
    response = await slow_api.get("/api/v1/slow")

    assert response.status_code == 504
    body = response.json()
    assert body["code"] == TIMEOUT_CODE
    assert body["details"]["timeout_seconds"] == 0.05
    assert body["request_id"]


async def test_a_fast_handler_is_untouched(slow_api: AsyncClient) -> None:
    response = await slow_api.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_the_timeout_is_lifted_once_a_response_starts(slow_api: AsyncClient) -> None:
    """A stream whose body outlasts the budget must not be cut off.

    The whole point of the deadline being time-to-first-byte: an SSE chat run
    legitimately takes longer than any sane time-to-respond limit.
    """
    response = await slow_api.get("/api/v1/slow-stream")

    assert response.status_code == 200
    assert response.text == "chunk 0\nchunk 1\nchunk 2\n"


async def test_an_abandoned_request_does_not_wait_for_its_handler(
    slow_api: AsyncClient,
) -> None:
    """The 504 must arrive at the deadline, not when the handler happens to end.

    Returning the right status only after the runaway work completes would give
    the client a correct-looking error while still holding the worker for the
    full five seconds.
    """
    started = anyio.current_time()
    response = await slow_api.get("/api/v1/slow")
    elapsed = anyio.current_time() - started

    assert response.status_code == 504
    assert elapsed < 1.0
