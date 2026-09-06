"""Node 2 — Planner: decide whether, what, and how to retrieve."""

from __future__ import annotations

from app.agents.models import Critique, QueryAnalysis, RetrievalPlan
from app.agents.state import RAGState
from app.config.settings import AgentSettings
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.prompts import registry
from app.services.llm import ChatService
from app.tools.base import ToolRegistry

NODE = "planner"
_TEMPLATE = "planner"
_NO_FEEDBACK = "This is the first attempt."
_FEEDBACK_HEADER = (
    "A previous attempt was judged inadequate. Plan differently — do not repeat "
    "the same search. Critic feedback:"
)


class PlannerNode(Component):
    """Turns an analysed question into a retrieval plan.

    Reads `query`, `analysis`, and — on a second pass — `critique`; writes `plan`
    and the revision counter. The planner is the graph's only re-entry point, so
    it is also where a retry is counted.

    Critic feedback is passed through verbatim rather than interpreted here: the
    planner's job is to act on the verdict, not to re-derive it.
    """

    def __init__(
        self,
        chat: ChatService,
        settings: AgentSettings,
        tools: ToolRegistry | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._chat = chat
        self._settings = settings
        self._tools = tools or ToolRegistry()

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        query = state.get("query", "")
        analysis = state.get("analysis") or QueryAnalysis(normalised_query=query)
        critique = state.get("critique")
        revisions = state.get("revisions", 0)
        is_retry = critique is not None
        record(node=NODE, query=query, attempt=revisions + 1, is_replan=is_retry)

        memory = state.get("memory")
        prompt = registry.render(
            _TEMPLATE,
            query=query,
            analysis=_describe(analysis),
            memory=memory.describe() if memory else "No prior conversation.",
            tools=self._tool_catalogue(),
            feedback=_describe_feedback(critique),
        )
        try:
            plan = await self._chat.structured(RetrievalPlan, prompt, query)
        except DependencyError:
            # Without a plan the graph cannot proceed usefully, but a default
            # plan is strictly better than no retrieval at all.
            self.logger.warning("agent.planner_failed", query=query)
            plan = RetrievalPlan(search_text=analysis.search_text(query), reasoning="fallback plan")

        plan = self._ensure_search_text(plan, analysis, query)
        self.logger.info(
            "agent.planned",
            retrieval_needed=plan.retrieval_needed,
            search_type=str(plan.search_type),
            top_k=plan.top_k,
            filters=len(plan.filters),
            tools=len(plan.tool_calls) if plan.wants_tools else 0,
            retry=is_retry,
        )
        record_outputs(
            retrieval_needed=plan.retrieval_needed,
            search_type=str(plan.search_type),
            top_k=plan.top_k,
            search_text=plan.search_text,
            tools_needed=plan.wants_tools,
            tool_calls=[{"tool": c.tool, "input": c.input} for c in plan.tool_calls],
            filters=[
                {"field": f.field, "operator": str(f.operator), "values": list(f.values)}
                for f in plan.filters
            ],
            reasoning=plan.reasoning,
        )
        return RAGState(
            plan=plan,
            revisions=revisions + 1 if is_retry else revisions,
            trace=[f"{NODE}:{'replan' if is_retry else 'plan'}:{plan.search_type}"],
        )

    def _tool_catalogue(self) -> str:
        """List the tools this deployment actually has.

        Generated from the registry rather than written into the template, so a
        tool that is not registered can never be planned, and a newly registered
        one needs no prompt edit.
        """
        tools = self._registered_tools()
        return "\n".join(tools) if tools else "No tools are available."

    def _registered_tools(self) -> list[str]:
        return [
            tool.describe() if hasattr(tool, "describe") else f"- {tool.name}: {tool.description}"
            for tool in self._tools.all()
        ]

    def _ensure_search_text(
        self, plan: RetrievalPlan, analysis: QueryAnalysis, query: str
    ) -> RetrievalPlan:
        """Guarantee the plan carries something searchable."""
        if plan.search_text.strip():
            return plan
        return plan.model_copy(update={"search_text": analysis.search_text(query)})


def _describe(analysis: QueryAnalysis) -> str:
    keywords = ", ".join(analysis.keywords) or "none"
    return (
        f"intent: {analysis.intent}\n"
        f"normalised query: {analysis.normalised_query}\n"
        f"keywords: {keywords}\n"
        f"ambiguous: {analysis.is_ambiguous}"
    )


def _describe_feedback(critique: Critique | None) -> str:
    if critique is None:
        return _NO_FEEDBACK
    missing = "; ".join(critique.unsupported_claims) or "none listed"
    return (
        f"{_FEEDBACK_HEADER}\n"
        f"sufficient context: {critique.sufficient_context}\n"
        f"grounded: {critique.grounded}\n"
        f"unsupported claims: {missing}\n"
        f"guidance: {critique.feedback or 'none given'}"
    )
