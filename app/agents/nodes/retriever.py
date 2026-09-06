"""Node 3 — Retriever: execute the plan against the retrieval layer."""

from __future__ import annotations

from app.agents.state import RAGState
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.retrieval.factory import RetrieverFactory
from app.retrieval.query import RetrievalQuery

NODE = "retriever"


class RetrieverNode(Component):
    """Runs the planned retrieval and puts the documents into state.

    Reads `plan`; writes `documents`. This is the only node with no model call:
    it makes no decisions of its own, it executes the plan it was given. A plan
    that says retrieval is not needed produces an empty document set rather than
    a skipped step, so the graph shape stays the same either way.
    """

    def __init__(self, factory: RetrieverFactory) -> None:
        self.logger = get_logger(__name__)
        self._factory = factory

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        plan = state.get("plan")
        record(node=NODE, retrieval_needed=plan.retrieval_needed if plan else False)
        if plan is None or not plan.retrieval_needed:
            self.logger.info("agent.retrieval_skipped")
            return RAGState(documents=[], trace=[f"{NODE}:skipped"])

        query = RetrievalQuery(
            text=plan.search_text or state.get("query", ""),
            top_k=plan.top_k,
            search_type=plan.search_type,
            filters=plan.to_metadata_filters(),
        )
        try:
            result = await self._factory.for_query(query).search(query)
        except DependencyError:
            # An unreachable store is not a reason to abandon the run: the
            # generator will report that it has no context, and the critic will
            # mark the context insufficient.
            self.logger.warning("agent.retrieval_failed", search_type=str(plan.search_type))
            return RAGState(documents=[], trace=[f"{NODE}:failed"])

        documents = list(result.documents)
        self.logger.info(
            "agent.retrieved",
            search_type=str(result.search_type),
            retriever=result.retriever,
            documents=len(documents),
        )
        record_outputs(
            documents_returned=len(documents),
            retriever=result.retriever,
            stages=list(result.stages),
        )
        return RAGState(
            documents=documents,
            trace=[f"{NODE}:{result.retriever}:{len(documents)}"],
        )
