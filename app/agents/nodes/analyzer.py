"""Node 1 — Query Analyzer: understand the question before deciding anything."""

from __future__ import annotations

from app.agents.models import QueryAnalysis
from app.agents.state import RAGState
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.prompts import registry
from app.services.llm import ChatService

NODE = "query_analyzer"
_TEMPLATE = "query_analyzer"


class QueryAnalyzerNode(Component):
    """Classifies intent and normalises the question for retrieval.

    Reads `query`; writes `analysis`. It deliberately makes no retrieval
    decisions — that is the planner's job — so that changing retrieval strategy
    never requires touching query understanding.
    """

    def __init__(self, chat: ChatService) -> None:
        self.logger = get_logger(__name__)
        self._chat = chat

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        query = state.get("query", "")
        memory = state.get("memory")
        record(node=NODE, query=query, has_memory=bool(memory and not memory.is_empty))
        try:
            analysis = await self._chat.structured(
                QueryAnalysis,
                # The conversation is what makes "what about the second one?"
                # resolvable into a standalone query, so it is the analyzer's
                # primary input, not decoration.
                registry.render(
                    _TEMPLATE,
                    query=query,
                    memory=memory.describe() if memory else "No prior conversation.",
                ),
                query,
            )
        except DependencyError:
            # Analysis is an optimisation, not a precondition: the planner can
            # work from the raw question, so a failure here degrades rather than
            # ends the run.
            self.logger.warning("agent.analyzer_failed", query=query)
            return RAGState(
                analysis=QueryAnalysis(normalised_query=query), trace=[f"{NODE}:fallback"]
            )

        self.logger.info("agent.analyzed", intent=analysis.intent, ambiguous=analysis.is_ambiguous)
        record_outputs(
            intent=analysis.intent,
            normalised_query=analysis.normalised_query,
            keywords=list(analysis.keywords),
            is_ambiguous=analysis.is_ambiguous,
        )
        return RAGState(analysis=analysis, trace=[f"{NODE}:{analysis.intent}"])
