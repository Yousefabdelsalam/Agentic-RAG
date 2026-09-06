"""Node 0 — Memory Loader: bring the conversation's history into the run."""

from __future__ import annotations

from app.agents.state import RAGState
from app.config.settings import MemorySettings
from app.core.base import Component
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.memory.models import MemorySnapshot
from app.memory.store import InMemoryStore

NODE = "memory_loader"


class MemoryLoaderNode(Component):
    """Loads summary, recent turns, and user facts for the session.

    Reads `session_id`; writes `memory`. It runs first because every node after
    it may need context: the analyzer to resolve "it" and "that one", the planner
    to know what was already tried, the generator to avoid repeating itself.

    A memory failure yields an empty snapshot. A conversation the agent cannot
    remember is degraded; one it cannot answer is broken.
    """

    def __init__(self, store: InMemoryStore, settings: MemorySettings) -> None:
        self.logger = get_logger(__name__)
        self._store = store
        self._settings = settings

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        session_id = state.get("session_id", "")
        record(node=NODE, session_id=session_id, enabled=self._settings.enabled)

        if not self._settings.enabled:
            return RAGState(memory=MemorySnapshot(), trace=[f"{NODE}:disabled"])

        try:
            snapshot = await self._store.snapshot(session_id)
        except Exception:
            self.logger.exception("agent.memory_load_failed", session_id=session_id)
            return RAGState(memory=MemorySnapshot(), trace=[f"{NODE}:failed"])

        self.logger.info(
            "agent.memory_loaded",
            session_id=session_id,
            turns=len(snapshot.recent),
            has_summary=bool(snapshot.summary),
            facts=len(snapshot.user_context.facts),
        )
        record_outputs(
            recent_turns=len(snapshot.recent),
            has_summary=bool(snapshot.summary),
            user_facts=len(snapshot.user_context.facts),
        )
        return RAGState(memory=snapshot, trace=[f"{NODE}:{len(snapshot.recent)}"])
