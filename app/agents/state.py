"""The graph state: the only channel through which nodes communicate.

Every node reads what it needs from this mapping and returns a partial update.
Nothing is passed node-to-node directly, so a node can be tested, replaced, or
run in isolation with a hand-built state.

`trace` is the one accumulating channel — each node appends the step it ran, and
LangGraph concatenates the fragments — so a completed run carries its own audit
trail without any node knowing what came before it.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.agents.models import Critique, QueryAnalysis, RetrievalPlan
from app.memory.models import MemorySnapshot
from app.retrieval.base import ScoredDocument
from app.tools.common import ToolResult


class RAGState(TypedDict, total=False):
    """State threaded through the agentic RAG graph."""

    session_id: str
    query: str

    memory: MemorySnapshot | None
    analysis: QueryAnalysis | None
    plan: RetrievalPlan | None
    tool_results: list[ToolResult]
    documents: list[ScoredDocument]
    answer: str
    critique: Critique | None

    revisions: int
    trace: Annotated[list[str], operator.add]


def initial_state(session_id: str, query: str) -> RAGState:
    """Build the state a run starts from."""
    return RAGState(
        session_id=session_id,
        query=query,
        memory=None,
        analysis=None,
        plan=None,
        tool_results=[],
        documents=[],
        answer="",
        critique=None,
        revisions=0,
        trace=[],
    )
