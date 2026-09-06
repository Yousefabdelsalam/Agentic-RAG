"""In-process conversation memory.

Satisfies the `MemoryStore` contract and adds the two things a summarising
memory needs beyond a transcript: somewhere to keep the rolling summary, and
somewhere to keep durable user facts.

State lives in the process, so it does not survive a restart and does not span
replicas. That is the right trade for a single-node deployment and the wrong one
for anything else — the graph depends on this class only through the methods
below, so a Redis or Postgres implementation is a substitution, not a rewrite.

Access is guarded by a lock because a single session can be touched by
concurrent requests, and a read-modify-write of the summary would otherwise
interleave.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass, field

import anyio

from app.config.settings import MemorySettings
from app.core.base import Component
from app.core.logging import get_logger
from app.memory.base import MemoryRecord
from app.memory.models import MemorySnapshot, UserContext, UserFact


@dataclass
class _Session:
    """Everything retained for one conversation."""

    turns: deque[MemoryRecord]
    summary: str = ""
    user_context: UserContext = field(default_factory=UserContext)
    turns_recorded: int = 0
    turns_summarised: int = 0


class InMemoryStore(Component):
    """Bounded, per-session conversation memory."""

    def __init__(self, settings: MemorySettings) -> None:
        self.logger = get_logger(__name__)
        self._settings = settings
        self._sessions: OrderedDict[str, _Session] = OrderedDict()
        self._lock = anyio.Lock()

    # ------------------------------------------------------------ MemoryStore

    async def append(self, record: MemoryRecord) -> None:
        """Add one turn to a session's short-term window."""
        async with self._lock:
            session = self._session(record.session_id)
            session.turns.append(record)
            session.turns_recorded += 1

    async def history(self, session_id: str, *, limit: int = 20) -> list[MemoryRecord]:
        """Return the most recent turns, oldest first."""
        async with self._lock:
            session = self._sessions.get(session_id)
            return list(session.turns)[-limit:] if session else []

    async def clear(self, session_id: str) -> None:
        """Forget a session entirely, including its summary and user facts."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    # ------------------------------------------------------- summary & context

    async def snapshot(self, session_id: str) -> MemorySnapshot:
        """Assemble the full read-side view of a session in one pass.

        Built under a single lock acquisition so the summary and the turns it
        excludes can never be read from two different moments.
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return MemorySnapshot()
            return MemorySnapshot(
                summary=session.summary,
                recent=tuple(session.turns)[-self._settings.short_term_turns :],
                user_context=session.user_context,
                turns_recorded=session.turns_recorded,
            )

    async def set_summary(self, session_id: str, summary: str) -> None:
        """Replace the rolling summary and mark the transcript as covered."""
        async with self._lock:
            session = self._session(session_id)
            session.summary = summary
            session.turns_summarised = session.turns_recorded

    async def merge_user_context(self, session_id: str, facts: tuple[UserFact, ...]) -> None:
        """Fold newly observed facts into what is already known."""
        if not facts:
            return
        async with self._lock:
            session = self._session(session_id)
            session.user_context = session.user_context.merged_with(
                facts, limit=self._settings.max_user_facts
            )

    async def turns_since_summary(self, session_id: str) -> int:
        """How many turns have accumulated since the summary was last written."""
        async with self._lock:
            session = self._sessions.get(session_id)
            return session.turns_recorded - session.turns_summarised if session else 0

    async def sessions(self) -> int:
        """Number of sessions currently retained."""
        async with self._lock:
            return len(self._sessions)

    # ------------------------------------------------------------------ internals

    def _session(self, session_id: str) -> _Session:
        """Fetch or create a session, evicting the least recently used if full."""
        session = self._sessions.get(session_id)
        if session is None:
            session = _Session(turns=deque(maxlen=self._settings.short_term_turns))
            self._sessions[session_id] = session
            if len(self._sessions) > self._settings.max_sessions:
                evicted, _ = self._sessions.popitem(last=False)
                self.logger.info("memory.evicted", session_id=evicted)
        self._sessions.move_to_end(session_id)
        return session
