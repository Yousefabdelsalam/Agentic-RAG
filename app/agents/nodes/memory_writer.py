"""Node 6 — Memory Writer: persist the turn and keep the summary current."""

from __future__ import annotations

from app.agents.state import RAGState
from app.config.settings import MemorySettings
from app.core.base import Component
from app.core.exceptions import DependencyError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.memory.base import MemoryRecord
from app.memory.models import ASSISTANT, USER, MemorySnapshot, UserContext
from app.memory.store import InMemoryStore
from app.prompts import registry
from app.services.llm import ChatService

NODE = "memory_writer"
_SUMMARY_TEMPLATE = "conversation_summary"
_CONTEXT_TEMPLATE = "user_context"


class MemoryWriterNode(Component):
    """Records the exchange, then compresses and extracts as needed.

    Reads `session_id`, `query`, `answer`, and `memory`; writes nothing back to
    state — it is the run's last step and its effects belong to the next run.

    Three things happen, in order and only when warranted: both turns are always
    appended; the summary is rewritten once enough turns have accumulated since
    the last one; user facts are extracted when the turn plausibly contained
    any. The two model calls are skipped by default, so an ordinary turn adds no
    latency beyond a dictionary write.
    """

    def __init__(self, store: InMemoryStore, chat: ChatService, settings: MemorySettings) -> None:
        self.logger = get_logger(__name__)
        self._store = store
        self._chat = chat
        self._settings = settings

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        session_id = state.get("session_id", "")
        query = state.get("query", "")
        answer = state.get("answer", "")
        record(node=NODE, session_id=session_id, enabled=self._settings.enabled)

        if not self._settings.enabled or not session_id:
            return RAGState(trace=[f"{NODE}:disabled"])

        try:
            await self._append(session_id, query, answer)
            summarised = await self._maybe_summarise(session_id, state.get("memory"))
            extracted = await self._maybe_extract(session_id, query, answer)
        except Exception:
            # Memory is written after the answer is already final. Losing it
            # costs the next turn some context; failing the run would cost the
            # user an answer they have effectively already been given.
            self.logger.exception("agent.memory_write_failed", session_id=session_id)
            return RAGState(trace=[f"{NODE}:failed"])

        record_outputs(summarised=summarised, facts_extracted=extracted)
        return RAGState(trace=[f"{NODE}:written"])

    async def _append(self, session_id: str, query: str, answer: str) -> None:
        for role, content in ((USER, query), (ASSISTANT, answer)):
            if content:
                await self._store.append(
                    MemoryRecord(session_id=session_id, role=role, content=content)
                )

    async def _maybe_summarise(self, session_id: str, memory: MemorySnapshot | None) -> bool:
        """Rewrite the rolling summary once the window has moved on far enough."""
        pending = await self._store.turns_since_summary(session_id)
        if pending < self._settings.summary_after_turns:
            return False

        snapshot = await self._store.snapshot(session_id)
        previous = (memory.summary if memory else "") or "None yet."
        try:
            summary = await self._chat.complete(
                registry.render(
                    _SUMMARY_TEMPLATE,
                    summary=previous,
                    transcript=snapshot.describe_transcript() or "No turns recorded.",
                ),
                "Summarise the conversation.",
            )
        except DependencyError:
            self.logger.warning("agent.summary_failed", session_id=session_id)
            return False

        await self._store.set_summary(session_id, summary.strip())
        self.logger.info("agent.summary_written", session_id=session_id, turns=pending)
        return True

    async def _maybe_extract(self, session_id: str, query: str, answer: str) -> int:
        """Pull durable user facts out of the exchange."""
        if not self._settings.extract_user_context or not query:
            return 0
        try:
            extracted = await self._chat.structured(
                UserContext,
                registry.render(_CONTEXT_TEMPLATE, question=query, answer=answer),
                query,
            )
        except DependencyError:
            self.logger.warning("agent.user_context_failed", session_id=session_id)
            return 0

        await self._store.merge_user_context(session_id, extracted.facts)
        if extracted.facts:
            self.logger.info(
                "agent.user_context_updated", session_id=session_id, facts=len(extracted.facts)
            )
        return len(extracted.facts)
