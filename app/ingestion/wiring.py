"""Wiring for the ingestion pipeline.

Ingestion is a batch workload with its own lifecycle, so it builds its own
container rather than being registered in the request-serving one: the API does
not resolve embeddings or the vector store yet, and starting them would make the
readiness probe depend on credentials the HTTP app has no use for. The container
class itself is shared, so these registrations move into `build_container`
unchanged once retrieval lands.
"""

from __future__ import annotations

from app.cache.backend import ManagedCacheBackend
from app.cache.versions import KnowledgeBaseVersion
from app.config.settings import Settings
from app.core.container import Container
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import PdfDocumentLoader
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.retrieval.chroma import ChromaVectorStore
from app.services.embeddings import EmbeddingService


def build_ingestion_container(settings: Settings) -> Container:
    """Create a container holding every component the ingestion pipeline needs.

    Registration order is startup order: dependencies are registered before the
    components that use them, so shutdown tears them down in reverse.
    """
    container = Container(settings)

    embeddings = EmbeddingService(
        settings.openai, settings.ingestion, settings.observability, settings.resilience
    )
    store = ChromaVectorStore(settings.chroma, embeddings, settings.resilience)
    loader = PdfDocumentLoader()
    chunker = RecursiveChunker(settings.ingestion)
    indexer = Indexer(store, settings.ingestion)

    # Offline ingestion changes the same corpus the API answers from, so it has
    # to advance the same version. Without this, `python -m app.ingestion` would
    # add documents that every cached answer continues to ignore.
    backend = ManagedCacheBackend(settings.cache)
    knowledge_base = KnowledgeBaseVersion(backend, settings.cache)

    container.register(EmbeddingService, embeddings)
    container.register(ChromaVectorStore, store)
    container.register(ManagedCacheBackend, backend)
    container.register(KnowledgeBaseVersion, knowledge_base)
    container.register(PdfDocumentLoader, loader)
    container.register(RecursiveChunker, chunker)
    container.register(Indexer, indexer)
    container.register(
        DocumentIngestionPipeline,
        DocumentIngestionPipeline(loader, chunker, indexer, knowledge_base),
    )
    return container
