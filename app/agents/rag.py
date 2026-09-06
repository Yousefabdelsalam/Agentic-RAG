"""The agent facade: the compiled graph behind the `Agent` contract."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from pydantic import Field

from app.agents.base import AgentRequest, AgentResponse
from app.agents.graph import GENERATOR, build_rag_graph
from app.agents.models import Critique, RetrievalPlan
from app.agents.nodes.analyzer import QueryAnalyzerNode
from app.agents.nodes.critic import CriticNode
from app.agents.nodes.generator import AnswerGeneratorNode
from app.agents.nodes.memory_loader import MemoryLoaderNode
from app.agents.nodes.memory_writer import MemoryWriterNode
from app.agents.nodes.planner import PlannerNode
from app.agents.nodes.retriever import RetrieverNode
from app.agents.nodes.tools import ToolNode
from app.agents.state import RAGState, initial_state
from app.config.settings import AgentSettings, ObservabilitySettings
from app.core.base import Component
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced, truncate
from app.models.base import Schema
from app.retrieval.base import ScoredDocument

STAGE = "stage"
ANSWER = "answer"
RESULT = "result"

UPDATES_MODE: Literal["updates"] = "updates"
VALUES_MODE: Literal["values"] = "values"


class AgentEvent(Schema):
    """One observable moment in a streaming run.

    `stage` marks a node finishing, `answer` carries a draft as soon as it
    exists, and exactly one `result` closes the stream with the final answer and
    its citations.
    """

    type: str
    stage: str = ""
    answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgenticRagAgent(Component):
    """Runs the RAG graph and reports the run alongside the answer.

    The response metadata is the run's own account of itself — plan, verdict,
    citations, node trace — so a caller can tell a confident answer from one the
    critic accepted reluctantly, without re-running anything.
    """

    name = "agentic_rag"

    def __init__(
        self,
        memory_loader: MemoryLoaderNode,
        analyzer: QueryAnalyzerNode,
        planner: PlannerNode,
        tools: ToolNode,
        retriever: RetrieverNode,
        generator: AnswerGeneratorNode,
        critic: CriticNode,
        memory_writer: MemoryWriterNode,
        settings: AgentSettings,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._observability = observability or ObservabilitySettings()
        self._graph = build_rag_graph(
            memory_loader,
            analyzer,
            planner,
            tools,
            retriever,
            generator,
            critic,
            memory_writer,
            settings,
        )

    @property
    def graph(self) -> Any:
        """The compiled graph, exposed for inspection and visualisation."""
        return self._graph

    @traced("agentic_rag.invoke")
    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """Run the graph to completion and return the final answer."""
        record(session_id=request.session_id, question=request.query)
        final = await self._graph.ainvoke(initial_state(request.session_id, request.query))
        state = cast(RAGState, final)
        self.logger.info(
            "agent.completed",
            session_id=request.session_id,
            revisions=state.get("revisions", 0),
            documents=len(state.get("documents", [])),
        )
        metadata = _metadata(state)
        self._record_run(state, metadata)
        return AgentResponse(
            session_id=request.session_id,
            answer=state.get("answer", ""),
            metadata=metadata,
        )

    def _record_run(self, state: RAGState, metadata: dict[str, Any]) -> None:
        """Put the run's verdict on the root span.

        The root run is what a reviewer opens first, so it carries the answer and
        the critic's judgement of it — enough to triage without descending into
        the node spans.
        """
        critique = state.get("critique")
        record(
            revisions=state.get("revisions", 0),
            documents_used=len(state.get("documents", [])),
            trace=list(state.get("trace", [])),
            sufficient_context=critique.sufficient_context if critique else None,
            grounded=critique.grounded if critique else None,
            confidence=critique.confidence if critique else None,
        )
        outputs: dict[str, Any] = {"citations": metadata["citations"]}
        if self._observability.capture_answers:
            outputs["final_answer"] = truncate(
                state.get("answer", ""), self._observability.max_captured_characters
            )
        record_outputs(**outputs)

    async def stream(self, request: AgentRequest) -> AsyncIterator[str]:
        """Yield each answer the run produces, in order.

        A revised run generates more than once, so this can yield twice: the
        draft the critic rejected, then its replacement. It is answer-level, not
        token-level — token streaming would have to come from the generator
        node's model call, not from graph updates.
        """
        async for event in self.stream_events(request):
            if event.type == ANSWER and event.answer:
                yield event.answer

    async def stream_events(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        """Yield the run as it happens: each stage, each answer, then the result.

        Two LangGraph stream modes run together. `updates` says which node just
        finished, which is what makes progress visible while a slow run is still
        working. `values` carries the reduced state, so the terminal event can
        report citations and the critic's verdict without this method
        reimplementing the state reducers to accumulate them itself.
        """
        state: RAGState = initial_state(request.session_id, request.query)
        async for mode, payload in self._graph.astream(
            state, stream_mode=[UPDATES_MODE, VALUES_MODE]
        ):
            if mode == VALUES_MODE:
                state = cast(RAGState, payload)
                continue
            for node, update in cast(dict[str, Any], payload).items():
                yield AgentEvent(type=STAGE, stage=node)
                answer = (update or {}).get("answer", "") if isinstance(update, dict) else ""
                if node == GENERATOR and answer:
                    yield AgentEvent(type=ANSWER, stage=node, answer=answer)

        self._record_run(state, _metadata(state))
        yield AgentEvent(
            type=RESULT,
            answer=state.get("answer", ""),
            metadata=_metadata(state),
        )


def _metadata(state: RAGState) -> dict[str, Any]:
    """Summarise the run for the caller."""
    plan = state.get("plan")
    critique = state.get("critique")
    documents = state.get("documents", [])
    return {
        "revisions": state.get("revisions", 0),
        "trace": list(state.get("trace", [])),
        "plan": _plan_summary(plan),
        "critique": _critique_summary(critique),
        "citations": [_citation(index, doc) for index, doc in enumerate(documents, start=1)],
        "tools": [
            {"tool": result.tool, "input": result.input, "ok": result.ok, "error": result.error}
            for result in state.get("tool_results", [])
        ],
    }


def _plan_summary(plan: RetrievalPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "retrieval_needed": plan.retrieval_needed,
        "search_type": str(plan.search_type),
        "top_k": plan.top_k,
        "filters": [
            {"field": f.field, "operator": str(f.operator), "values": list(f.values)}
            for f in plan.filters
        ],
        "tools_needed": plan.wants_tools,
        "tool_calls": [{"tool": c.tool, "input": c.input} for c in plan.tool_calls],
        "reasoning": plan.reasoning,
    }


def _critique_summary(critique: Critique | None) -> dict[str, Any] | None:
    if critique is None:
        return None
    return {
        "sufficient_context": critique.sufficient_context,
        "grounded": critique.grounded,
        "confidence": critique.confidence,
        "unsupported_claims": list(critique.unsupported_claims),
        "feedback": critique.feedback,
    }


def _citation(index: int, scored: ScoredDocument) -> dict[str, Any]:
    metadata = scored.document.metadata
    return {
        "marker": index,
        "id": scored.document.id,
        "filename": metadata.get("filename", ""),
        "page": metadata.get("page", ""),
        "score": scored.score,
    }
