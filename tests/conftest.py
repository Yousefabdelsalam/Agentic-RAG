from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.config.settings import CacheSettings, Settings, get_settings
from app.main import create_app

#: Settings groups that must never leak in from the machine running the tests.
#: A developer with a real OPENAI__API_KEY in .env would otherwise get a fully
#: configured app in tests written for an unconfigured one — and, worse, real
#: billed API calls from a unit test.
_ISOLATED_PREFIXES = (
    "OPENAI",
    "LANGSMITH",
    "LANGCHAIN",
    "CHROMA",
    "AGENT",
    "MEMORY",
    "CACHE",
    "RAG_CACHE",
)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run every test against declared settings only, never the developer's.

    Two sources are cut off: the process environment, and the `.env` file that
    `Settings` reads by default. Tests that want a configured deployment say so
    explicitly by passing values to `Settings(...)`, which keeps what a test
    exercises visible in the test rather than dependent on who runs it.
    """
    for name in list(os.environ):
        if name.upper().startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(name, raising=False)

    # `CacheSettings` is a settings object in its own right — it reads the
    # `RAG_CACHE_*` names directly — so silencing the root one is not enough to
    # cut it off from the developer's `.env`.
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    monkeypatch.setitem(CacheSettings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    return Settings(environment="local", debug=True)


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http_client:
        async with app.router.lifespan_context(app):
            yield http_client
