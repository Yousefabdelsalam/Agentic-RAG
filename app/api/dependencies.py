"""FastAPI dependency providers.

Routers depend on the annotated aliases declared here; nothing in the API layer
constructs its own collaborators.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.cache.agent import CachedRagAgent
from app.cache.gateway import CacheGateway
from app.config.settings import Settings, get_settings
from app.core.base import Component
from app.core.container import Container
from app.core.exceptions import ConfigurationError, DependencyError
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.memory.store import InMemoryStore

_UNCONFIGURED = (
    "This endpoint needs a language model, which is not configured. "
    "Set OPENAI__API_KEY and restart."
)


def get_container(request: Request) -> Container:
    """Return the container attached to the application state at startup."""
    return request.app.state.container  # type: ignore[no-any-return]


def get_request_id(request: Request) -> str | None:
    """Return the correlation id assigned by the request middleware."""
    return getattr(request.state, "request_id", None)


def _require[T: Component](container: Container, key: type[T]) -> T:
    """Resolve a component, turning an absent one into a 502 the client can read.

    A component missing from the container means the deployment was started
    without the credentials it needs. That is not a bug in the request, so it
    must not surface as a 500 with no explanation.
    """
    try:
        return container.resolve(key)
    except ConfigurationError as exc:
        raise DependencyError(_UNCONFIGURED, details={"component": key.__name__}) from exc


def get_agent(request: Request) -> CachedRagAgent:
    """Return the RAG agent behind its cache, or fail with a stated reason.

    The cached wrapper, not the bare graph: every caller wants the answer, and
    which of the two produced it is the wrapper's business.
    """
    return _require(get_container(request), CachedRagAgent)


def get_cache(request: Request) -> CacheGateway:
    """Return the cache gateway, or fail with a stated reason."""
    return _require(get_container(request), CacheGateway)


def get_pipeline(request: Request) -> DocumentIngestionPipeline:
    """Return the ingestion pipeline, or fail with a stated reason."""
    return _require(get_container(request), DocumentIngestionPipeline)


def get_memory(request: Request) -> InMemoryStore:
    """Return the conversation store; always available, credentials or not."""
    return _require(get_container(request), InMemoryStore)


SettingsDep = Annotated[Settings, Depends(get_settings)]
ContainerDep = Annotated[Container, Depends(get_container)]
RequestIdDep = Annotated[str | None, Depends(get_request_id)]
AgentDep = Annotated[CachedRagAgent, Depends(get_agent)]
CacheDep = Annotated[CacheGateway, Depends(get_cache)]
PipelineDep = Annotated[DocumentIngestionPipeline, Depends(get_pipeline)]
MemoryDep = Annotated[InMemoryStore, Depends(get_memory)]
