"""Wiring for the retrieval layer.

Mirrors the ingestion wiring: retrieval owns its own container so that starting
it stays an explicit act. The registrations move into `build_container` unchanged
once an API surface needs them.
"""

from __future__ import annotations

from app.config.settings import Settings
from app.core.container import Container
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.factory import RetrieverFactory
from app.retrieval.hybrid import SparseRetriever
from app.services.embeddings import EmbeddingService


def build_retrieval_container(
    settings: Settings, *, sparse: SparseRetriever | None = None
) -> Container:
    """Create a container holding the retrieval layer's components.

    `sparse` is the seam for a lexical backend: supply one and hybrid search
    fuses, omit it and hybrid degrades to dense-only.
    """
    container = Container(settings)

    embeddings = EmbeddingService(
        settings.openai, settings.ingestion, settings.observability, settings.resilience
    )
    store = ChromaVectorStore(settings.chroma, embeddings, settings.resilience)

    container.register(EmbeddingService, embeddings)
    container.register(ChromaVectorStore, store)
    container.register(
        RetrieverFactory,
        RetrieverFactory(store, embeddings, settings.retrieval, sparse, settings.observability),
    )
    return container
