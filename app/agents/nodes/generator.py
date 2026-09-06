"""Node 4 — Answer Generator: write the answer from retrieved context only."""

from __future__ import annotations

from app.agents.context import format_context, format_tool_results
from app.agents.state import RAGState
from app.config.settings import AgentSettings, ObservabilitySettings
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced, truncate
from app.prompts import registry
from app.services.llm import ChatService

NODE = "generator"
_TEMPLATE = "answer_generator"
_NO_CONTEXT_ANSWER = (
    "I could not find anything in the indexed documents that answers this question."
)
_FAILED_ANSWER = "The answer could not be generated because the language model was unreachable."


class AnswerGeneratorNode(Component):
    """Composes a grounded, cited answer.

    Reads `query`, `memory`, `tool_results`, and `documents`; writes `answer`.
    Context blocks are numbered so the model can cite them and the critic can
    check those citations against the same numbering.

    Documents and successful tool calls are both grounds for an answer, so the
    node refuses only when it has neither. Memory is context for interpreting
    the question, never grounds for a factual claim, which is why it is passed
    separately rather than folded into the numbered blocks.
    """

    def __init__(
        self,
        chat: ChatService,
        settings: AgentSettings,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        self.logger = get_logger(__name__)
        self._chat = chat
        self._settings = settings
        self._observability = observability or ObservabilitySettings()

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        query = state.get("query", "")
        documents = state.get("documents", [])
        tool_results = state.get("tool_results", [])
        memory = state.get("memory")
        usable_tools = [result for result in tool_results if result.ok]
        record(
            node=NODE,
            query=query,
            context_documents=len(documents),
            tool_results=len(usable_tools),
        )
        if not documents and not usable_tools:
            # Answering with no grounds at all is how hallucinations start;
            # refuse here rather than asking the model to be disciplined.
            self.logger.info("agent.generation_skipped")
            return RAGState(answer=_NO_CONTEXT_ANSWER, trace=[f"{NODE}:no_context"])

        prompt = registry.render(
            _TEMPLATE,
            context=format_context(documents, limit=self._settings.max_context_documents),
            tools=format_tool_results(tool_results),
            memory=memory.describe() if memory else "No prior conversation.",
            query=query,
        )
        self._record_prompt(prompt)
        try:
            answer = await self._chat.complete(prompt, query)
        except DependencyError:
            self.logger.warning("agent.generation_failed")
            return RAGState(answer=_FAILED_ANSWER, trace=[f"{NODE}:failed"])

        self.logger.info(
            "agent.generated",
            characters=len(answer),
            documents=len(documents),
            tool_results=len(usable_tools),
        )
        if self._observability.capture_answers:
            record_outputs(answer=truncate(answer, self._observability.max_captured_characters))
        return RAGState(answer=answer.strip(), trace=[f"{NODE}:{len(documents)}"])

    def _record_prompt(self, prompt: str) -> None:
        """Record the prompt the model will actually receive.

        This is the fully rendered text — instructions plus numbered context —
        not the template, because a grounding failure is usually explained by
        what ended up in the context, not by the template it was poured into.
        """
        record(final_prompt_characters=len(prompt))
        if self._observability.capture_prompts:
            record(final_prompt=truncate(prompt, self._observability.max_captured_characters))
