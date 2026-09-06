"""Wiring for the agentic RAG graph.

Follows the ingestion and retrieval wiring: the agent owns its container, so the
components it needs — chat, embeddings, store, retrievers, memory, tools — start
and stop as one unit.
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
from app.config.settings import Settings
from app.core.container import Container
from app.memory.store import InMemoryStore
from app.retrieval.chroma import ChromaVectorStore
from app.retrieval.factory import RetrieverFactory
from app.retrieval.hybrid import SparseRetriever
from app.services.embeddings import EmbeddingService
from app.services.llm import ChatService
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.clock import CurrentTimeTool
from app.tools.web_search import UnavailableWebSearch, WebSearchBackend, WebSearchTool


def build_tool_registry(search_backend: WebSearchBackend | None = None) -> ToolRegistry:
    """Register the tools this deployment exposes.

    The registry is also the planner's catalogue, so registering a tool is the
    single act that makes it both callable and plannable. Web search falls back
    to a backend that fails loudly rather than being omitted, so a plan that
    reaches for it produces a stated reason instead of an unknown-tool error.
    """
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    registry.register(WebSearchTool(search_backend or UnavailableWebSearch()))
    return registry


def build_agent_container(
    settings: Settings,
    *,
    sparse: SparseRetriever | None = None,
    search_backend: WebSearchBackend | None = None,
) -> Container:
    """Create a container holding the agent and everything it depends on."""
    container = Container(settings)

    observability = settings.observability
    embeddings = EmbeddingService(
        settings.openai, settings.ingestion, observability, settings.resilience
    )
    chat = ChatService(settings.openai, settings.agent, observability)
    store = ChromaVectorStore(settings.chroma, embeddings, settings.resilience)
    factory = RetrieverFactory(store, embeddings, settings.retrieval, sparse, observability)
    memory = InMemoryStore(settings.memory)
    tools = build_tool_registry(search_backend)

    container.register(EmbeddingService, embeddings)
    container.register(ChatService, chat)
    container.register(ChromaVectorStore, store)
    container.register(RetrieverFactory, factory)
    container.register(InMemoryStore, memory)
    container.register(
        AgenticRagAgent,
        AgenticRagAgent(
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
        ),
    )
    return container
