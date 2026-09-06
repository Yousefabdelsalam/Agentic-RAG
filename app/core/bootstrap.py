"""Wiring: builds the container from settings.

This is the single place where concrete implementations are chosen for the HTTP
application. Feature modules keep their own container builders for batch and
offline use; this one assembles the set the API serves from, so both paths share
component classes without sharing a lifecycle.

An API key is required to build the model-backed half of the stack. Without one
the process still starts — a container that refuses to boot makes `/health`
unreachable precisely when someone is trying to find out what is wrong — but the
endpoints that need it answer 503 with a stated reason, and the gap is logged at
startup rather than discovered at the first request.
"""

from __future__ import annotations

from app.agents.nodes.analyzer import QueryAnalyzerNode
from app.agents.nodes.critic import CriticNode
from app.agents.nodes.generator import AnswerGeneratorNode
from app.agents.nodes.memory_loader import MemoryLoaderNode
from app.agents.nodes.memory_writer import MemoryWriterNode
from app.agents.nodes.planner import PlannerNode
from app.agents.nodes.retriever import RetrieverNode
from app.agents.nodes.tools import ToolNode
from app.agents.rag import AgenticRagAgent
from app.agents.wiring import build_tool_registry
from app.cache.agent import CachedRagAgent
from app.cache.backend import ManagedCacheBackend
from app.cache.gateway import CacheGateway
from app.cache.versions import KnowledgeBaseVersion, build_resolver
from app.config.settings import Settings
from app.core.container import Container
from app.core.logging import get_logger
from app.ingestion.chunker import RecursiveChunker
from app.ingestion.indexer import Indexer
from app.ingestion.loader import PdfDocumentLoader
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.memory.store import InMemoryStore
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.factory import RetrieverFactory
from app.services.embeddings import EmbeddingService
from app.services.llm import ChatService

logger = get_logger(__name__)


def build_container(settings: Settings) -> Container:
    """Create the container and register every application component."""
    container = Container(settings)

    # Memory needs no credentials, so a conversation can be reset even on an
    # otherwise unconfigured deployment.
    memory = InMemoryStore(settings.memory)
    container.register(InMemoryStore, memory)

    if settings.openai.api_key is None:
        logger.warning(
            "bootstrap.degraded",
            reason="OPENAI__API_KEY is not set",
            unavailable=["chat", "upload"],
        )
        return container

    observability = settings.observability
    embeddings = EmbeddingService(
        settings.openai, settings.ingestion, observability, settings.resilience
    )
    chat = ChatService(settings.openai, settings.agent, observability)
    store = ChromaVectorStore(settings.chroma, embeddings, settings.resilience)
    factory = RetrieverFactory(store, embeddings, settings.retrieval, None, observability)
    tools = build_tool_registry()

    container.register(EmbeddingService, embeddings)
    container.register(ChatService, chat)
    container.register(ChromaVectorStore, store)
    container.register(RetrieverFactory, factory)

    # The cache is registered before ingestion and the agent because both hold a
    # reference to it: ingestion to advance the knowledge base version, the
    # agent to answer from it. Registration order is startup order, so the
    # backend is connected before anything asks it a question.
    backend = ManagedCacheBackend(settings.cache)
    knowledge_base = KnowledgeBaseVersion(backend, settings.cache)
    container.register(ManagedCacheBackend, backend)
    container.register(KnowledgeBaseVersion, knowledge_base)

    loader = PdfDocumentLoader()
    chunker = RecursiveChunker(settings.ingestion)
    indexer = Indexer(store, settings.ingestion)
    container.register(PdfDocumentLoader, loader)
    container.register(RecursiveChunker, chunker)
    container.register(Indexer, indexer)
    container.register(
        DocumentIngestionPipeline,
        DocumentIngestionPipeline(loader, chunker, indexer, knowledge_base),
    )

    agent = AgenticRagAgent(
        MemoryLoaderNode(memory, settings.memory),
        QueryAnalyzerNode(chat),
        PlannerNode(chat, settings.agent, tools),
        ToolNode(tools, settings.tools),
        RetrieverNode(factory),
        AnswerGeneratorNode(chat, settings.agent, observability),
        CriticNode(chat, settings.agent),
        MemoryWriterNode(memory, chat, settings.memory),
        settings.agent,
        observability,
    )
    gateway = CacheGateway(
        settings.cache,
        backend,
        build_resolver(knowledge_base, settings.cache, settings.openai, settings.agent),
        settings.agent,
        embeddings,
    )
    container.register(AgenticRagAgent, agent)
    container.register(CacheGateway, gateway)
    container.register(CachedRagAgent, CachedRagAgent(agent, gateway, memory, settings.memory))
    return container
