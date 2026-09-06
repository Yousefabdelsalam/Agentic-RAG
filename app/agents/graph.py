"""The agentic RAG graph.

    START -> memory_loader -> query_analyzer -> planner -> ... -> memory_writer -> END

Between the planner and the generator sit two optional stages. The planner
decides which of them the question actually needs, and the routers below skip
whichever it did not ask for:

    planner --tools--> tools --retrieval--> retriever --> generator
       |                 |                                    ^
       |                 +----------------- no retrieval -----+
       +--------- no tools, no retrieval ---------------------+

    generator -> critic --accepted--> memory_writer -> END
                    |                       ^
                    +-- insufficient --> planner

Skipping is routing, not a no-op call: a question needing neither capability
never enters those nodes, so the trace shows what the run actually did rather
than a uniform pipeline with empty steps in it.

Assembly is the whole responsibility of this module: it knows the shape, the
nodes know the work, and neither knows the other's business. Every routing
decision reads state the nodes already wrote — no router calls a model, and no
node decides where control goes next.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes.analyzer import QueryAnalyzerNode
from app.agents.nodes.critic import CriticNode
from app.agents.nodes.generator import AnswerGeneratorNode
from app.agents.nodes.memory_loader import MemoryLoaderNode
from app.agents.nodes.memory_writer import MemoryWriterNode
from app.agents.nodes.planner import PlannerNode
from app.agents.nodes.retriever import RetrieverNode
from app.agents.nodes.tools import ToolNode
from app.agents.state import RAGState
from app.config.settings import AgentSettings
from app.core.logging import get_logger

MEMORY_LOADER = "memory_loader"
ANALYZER = "query_analyzer"
PLANNER = "planner"
TOOLS = "tools"
RETRIEVER = "retriever"
GENERATOR = "generator"
CRITIC = "critic"
MEMORY_WRITER = "memory_writer"

logger = get_logger(__name__)


def build_rag_graph(
    memory_loader: MemoryLoaderNode,
    analyzer: QueryAnalyzerNode,
    planner: PlannerNode,
    tools: ToolNode,
    retriever: RetrieverNode,
    generator: AnswerGeneratorNode,
    critic: CriticNode,
    memory_writer: MemoryWriterNode,
    settings: AgentSettings,
) -> CompiledStateGraph[RAGState, Any, RAGState, RAGState]:
    """Wire the nodes into a compiled graph."""
    graph = StateGraph(RAGState)

    graph.add_node(MEMORY_LOADER, memory_loader)
    graph.add_node(ANALYZER, analyzer)
    graph.add_node(PLANNER, planner)
    graph.add_node(TOOLS, tools)
    graph.add_node(RETRIEVER, retriever)
    graph.add_node(GENERATOR, generator)
    graph.add_node(CRITIC, critic)
    graph.add_node(MEMORY_WRITER, memory_writer)

    graph.add_edge(START, MEMORY_LOADER)
    graph.add_edge(MEMORY_LOADER, ANALYZER)
    graph.add_edge(ANALYZER, PLANNER)

    graph.add_conditional_edges(
        PLANNER,
        _plan_router,
        {TOOLS: TOOLS, RETRIEVER: RETRIEVER, GENERATOR: GENERATOR},
    )
    graph.add_conditional_edges(
        TOOLS,
        _after_tools_router,
        {RETRIEVER: RETRIEVER, GENERATOR: GENERATOR},
    )
    graph.add_edge(RETRIEVER, GENERATOR)
    graph.add_edge(GENERATOR, CRITIC)
    graph.add_conditional_edges(
        CRITIC,
        _critic_router(settings),
        {PLANNER: PLANNER, MEMORY_WRITER: MEMORY_WRITER},
    )
    graph.add_edge(MEMORY_WRITER, END)

    return graph.compile()


def _plan_router(state: RAGState) -> str:
    """Send the run to the first stage the plan actually calls for.

    Tools run before retrieval so a tool that reshapes the question — resolving
    "today" into a date, for instance — does so before anything is searched.
    """
    plan = state.get("plan")
    if plan is None:
        return RETRIEVER
    if plan.wants_tools:
        return TOOLS
    return RETRIEVER if plan.retrieval_needed else GENERATOR


def _after_tools_router(state: RAGState) -> str:
    """Retrieve after tools only if the plan asked for retrieval too."""
    plan = state.get("plan")
    return RETRIEVER if plan is None or plan.retrieval_needed else GENERATOR


def _critic_router(settings: AgentSettings) -> Callable[[RAGState], str]:
    """Build the router that decides whether the critic's verdict ends the run.

    Returns either `PLANNER` or `MEMORY_WRITER`. The loop policy lives here
    rather than in the critic so that the critic stays a pure evaluator: it
    reports what it found, this decides what to do about it.

    Both outcomes pass through the memory writer, so a run that gave up after
    exhausting its revisions is still remembered — the next turn needs to know
    what was already tried.
    """

    def route(state: RAGState) -> str:
        critique = state.get("critique")
        revisions = state.get("revisions", 0)

        if critique is None or critique.accepts(min_confidence=settings.min_confidence):
            return MEMORY_WRITER
        if not critique.sufficient_context and revisions < settings.max_revisions:
            return PLANNER

        # A grounding failure is not fixed by retrieving again — the context was
        # there and the answer strayed from it — so replanning is reserved for
        # insufficient context, and the budget is finite either way.
        logger.info(
            "agent.loop_exhausted" if revisions >= settings.max_revisions else "agent.not_grounded",
            revisions=revisions,
            grounded=critique.grounded,
            sufficient_context=critique.sufficient_context,
        )
        return MEMORY_WRITER

    return route
