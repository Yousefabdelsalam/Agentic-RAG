"""Application factory and ASGI entrypoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.v1.router import api_router
from app.config.settings import Settings, get_settings
from app.core.bootstrap import build_container
from app.core.handlers import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.core.middleware import RequestContextMiddleware
from app.core.timeout import TimeoutMiddleware
from app.core.tracing import configure_tracing

logger = get_logger(__name__)

#: What `_lifespan` hands to FastAPI: a factory producing the startup/shutdown
#: context manager for an application instance.
Lifespan = Callable[[FastAPI], AbstractAsyncContextManager[None]]

DESCRIPTION = """
Agentic RAG over your own documents.

Ingest PDFs, then ask questions about them. Each question runs through a graph
that analyses it, plans how to gather evidence, calls tools and retrieval only
where they are needed, drafts an answer, and has a critic check that answer
against its sources before it is returned.

Answers carry citations. The `trace` field on a response records which nodes ran,
and `grounded` / `sufficient_context` report the critic's verdict, so a confident
answer can be told apart from one that was accepted reluctantly.
"""

TAGS = [
    {"name": "chat", "description": "Ask questions and stream answers."},
    {"name": "documents", "description": "Add documents to the index."},
    {"name": "memory", "description": "Inspect and clear conversation state."},
    {"name": "cache", "description": "What the answer cache has avoided."},
    {"name": "health", "description": "Liveness and readiness probes."},
]


def _lifespan(settings: Settings) -> Lifespan:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = build_container(settings)
        app.state.container = container
        await container.startup()
        logger.info("app.started", environment=settings.environment, version=__version__)
        try:
            yield
        finally:
            await container.shutdown()
            logger.info("app.stopped")

    return lifespan


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a fully configured FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.logging)
    configure_tracing(settings.langsmith)

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=DESCRIPTION,
        openapi_tags=TAGS,
        debug=settings.debug,
        root_path=settings.server.root_path,
        docs_url=None if settings.is_production else "/docs",
        redoc_url=None,
        openapi_url=None if settings.is_production else "/openapi.json",
        lifespan=_lifespan(settings),
    )

    # Order matters: middleware added later runs first, so the request context
    # is established before the timeout can produce an envelope that needs the
    # correlation id.
    app.add_middleware(
        TimeoutMiddleware, timeout_seconds=settings.resilience.request_timeout_seconds
    )
    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_prefix)
    return app


app = create_app()
