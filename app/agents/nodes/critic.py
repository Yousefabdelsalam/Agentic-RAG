"""Node 5 — Critic: audit the answer for sufficient context and grounding."""

from __future__ import annotations

from app.agents.context import format_context, format_tool_results
from app.agents.models import Critique
from app.agents.state import RAGState
from app.config.settings import AgentSettings
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.prompts import registry
from app.services.llm import ChatService

NODE = "critic"
_TEMPLATE = "critic"
_EMPTY_CONTEXT_FEEDBACK = (
    "Retrieval returned nothing. Broaden the search: drop metadata filters, "
    "raise top_k, or search with different terms."
)


class CriticNode(Component):
    """Judges whether the answer is supported by the context it cites.

    Reads `query`, `documents`, and `answer`; writes `critique`. It only reports
    a verdict — whether that verdict sends the run back to the planner is the
    router's decision, so the critic stays a pure evaluator and the loop policy
    lives in one place.
    """

    def __init__(self, chat: ChatService, settings: AgentSettings) -> None:
        self.logger = get_logger(__name__)
        self._chat = chat
        self._settings = settings

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        documents = state.get("documents", [])
        tool_results = state.get("tool_results", [])
        usable_tools = [result for result in tool_results if result.ok]
        record(node=NODE, context_documents=len(documents), tool_results=len(usable_tools))
        if not documents and not usable_tools:
            # No context is an unambiguous verdict; spending a model call to
            # confirm it would only add a way to get it wrong.
            return RAGState(
                critique=Critique(
                    sufficient_context=False, feedback=_EMPTY_CONTEXT_FEEDBACK, confidence=1.0
                ),
                trace=[f"{NODE}:no_context"],
            )

        prompt = registry.render(
            _TEMPLATE,
            query=state.get("query", ""),
            context=format_context(documents, limit=self._settings.max_context_documents),
            tools=format_tool_results(tool_results),
            answer=state.get("answer", ""),
        )
        try:
            critique = await self._chat.structured(Critique, prompt, state.get("query", ""))
        except DependencyError:
            # A critic that cannot run must not silently approve. Accept the
            # answer, but record low confidence so the caller can see why.
            self.logger.warning("agent.critic_failed")
            return RAGState(
                critique=Critique(confidence=0.0, feedback="critic unavailable"),
                trace=[f"{NODE}:failed"],
            )

        self.logger.info(
            "agent.critiqued",
            sufficient_context=critique.sufficient_context,
            grounded=critique.grounded,
            confidence=critique.confidence,
            unsupported=len(critique.unsupported_claims),
        )
        record_outputs(
            sufficient_context=critique.sufficient_context,
            grounded=critique.grounded,
            confidence=critique.confidence,
            unsupported_claims=list(critique.unsupported_claims),
            feedback=critique.feedback,
        )
        return RAGState(
            critique=critique,
            trace=[f"{NODE}:{'accept' if critique.accepts(min_confidence=0.0) else 'reject'}"],
        )
