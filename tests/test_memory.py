from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Any

import pytest

from app.agents.nodes.memory_loader import MemoryLoaderNode
from app.agents.nodes.memory_writer import MemoryWriterNode
from app.agents.state import RAGState
from app.config.settings import MemorySettings
from app.core.exceptions import DependencyError
from app.memory.base import MemoryRecord
from app.memory.models import ASSISTANT, USER, MemorySnapshot, UserContext, UserFact
from app.memory.store import InMemoryStore
from app.models.base import Schema


class FakeChatService:
    """Scripted chat service, mirroring the one the agent tests use."""

    def __init__(self, **scripts: Sequence[Any]) -> None:
        self._scripts = {name: deque(values) for name, values in scripts.items()}
        self.prompts: list[str] = []
        self.fail_on: set[str] = set()

    def _next(self, key: str) -> Any:
        queue = self._scripts.get(key)
        if not queue:
            raise AssertionError(f"no scripted reply for {key!r}")
        return queue[0] if len(queue) == 1 else queue.popleft()

    async def complete(self, system: str, user: str) -> str:
        self.prompts.append(system)
        if "text" in self.fail_on:
            raise DependencyError("chat down")
        return str(self._next("text"))

    async def structured(self, schema: type[Schema], system: str, user: str) -> Any:
        self.prompts.append(system)
        if schema.__name__ in self.fail_on:
            raise DependencyError("chat down")
        return self._next(schema.__name__)


def _settings(**overrides: Any) -> MemorySettings:
    return MemorySettings(**({"short_term_turns": 4, "summary_after_turns": 4} | overrides))


def _record(session: str, role: str, content: str) -> MemoryRecord:
    return MemoryRecord(session_id=session, role=role, content=content)


# ------------------------------------------------------------------------------ store


async def test_history_returns_turns_oldest_first() -> None:
    store = InMemoryStore(_settings())
    for index in range(3):
        await store.append(_record("s1", USER, f"turn {index}"))

    history = await store.history("s1")

    assert [record.content for record in history] == ["turn 0", "turn 1", "turn 2"]


async def test_short_term_memory_is_a_window_not_a_log() -> None:
    store = InMemoryStore(_settings(short_term_turns=2, summary_after_turns=2))
    for index in range(5):
        await store.append(_record("s1", USER, f"turn {index}"))

    snapshot = await store.snapshot("s1")

    assert [record.content for record in snapshot.recent] == ["turn 3", "turn 4"]
    assert snapshot.turns_recorded == 5


async def test_sessions_are_isolated() -> None:
    store = InMemoryStore(_settings())
    await store.append(_record("s1", USER, "mine"))
    await store.append(_record("s2", USER, "yours"))

    assert [r.content for r in await store.history("s1")] == ["mine"]
    assert [r.content for r in await store.history("s2")] == ["yours"]


async def test_an_unknown_session_reads_as_empty() -> None:
    store = InMemoryStore(_settings())

    assert await store.history("never-seen") == []
    assert (await store.snapshot("never-seen")).is_empty


async def test_clear_forgets_everything_about_a_session() -> None:
    store = InMemoryStore(_settings())
    await store.append(_record("s1", USER, "hello"))
    await store.set_summary("s1", "a summary")
    await store.merge_user_context("s1", (UserFact(key="role", value="engineer"),))

    await store.clear("s1")

    assert (await store.snapshot("s1")).is_empty


async def test_summary_is_stored_and_resets_the_pending_count() -> None:
    store = InMemoryStore(_settings())
    for index in range(4):
        await store.append(_record("s1", USER, f"turn {index}"))
    assert await store.turns_since_summary("s1") == 4

    await store.set_summary("s1", "they discussed leave policy")

    assert await store.turns_since_summary("s1") == 0
    assert (await store.snapshot("s1")).summary == "they discussed leave policy"


async def test_least_recently_used_sessions_are_evicted() -> None:
    store = InMemoryStore(_settings(max_sessions=2))
    await store.append(_record("s1", USER, "one"))
    await store.append(_record("s2", USER, "two"))
    await store.append(_record("s1", USER, "one again"))  # s1 becomes most recent
    await store.append(_record("s3", USER, "three"))

    assert await store.sessions() == 2
    assert await store.history("s2") == []
    assert await store.history("s1")


# ----------------------------------------------------------------------- user context


def test_user_facts_are_merged_with_later_values_winning() -> None:
    existing = UserContext(facts=(UserFact(key="role", value="analyst"),))

    merged = existing.merged_with((UserFact(key="role", value="manager"),), limit=10)

    assert merged.facts == (UserFact(key="role", value="manager"),)


def test_user_facts_are_bounded() -> None:
    context = UserContext()
    incoming = tuple(UserFact(key=f"k{index}", value=str(index)) for index in range(10))

    merged = context.merged_with(incoming, limit=3)

    assert len(merged.facts) == 3
    assert merged.facts[-1].key == "k9"


def test_blank_facts_are_ignored() -> None:
    merged = UserContext().merged_with(
        (UserFact(key="", value="x"), UserFact(key="k", value="")), limit=10
    )

    assert merged.facts == ()


async def test_user_context_survives_across_turns() -> None:
    store = InMemoryStore(_settings())
    await store.merge_user_context("s1", (UserFact(key="team", value="finance"),))
    await store.merge_user_context("s1", (UserFact(key="units", value="metric"),))

    facts = (await store.snapshot("s1")).user_context.facts

    assert {fact.key for fact in facts} == {"team", "units"}


# --------------------------------------------------------------------------- snapshot


def test_an_empty_snapshot_says_so_rather_than_rendering_blank() -> None:
    assert MemorySnapshot().describe() == "No prior conversation."


def test_a_snapshot_renders_all_three_kinds_of_memory() -> None:
    snapshot = MemorySnapshot(
        summary="they asked about leave",
        recent=(_record("s1", USER, "and holidays?"),),
        user_context=UserContext(facts=(UserFact(key="team", value="finance"),)),
    )

    described = snapshot.describe()

    assert "they asked about leave" in described
    assert "team: finance" in described
    assert "user: and holidays?" in described


# ------------------------------------------------------------------------ loader node


async def test_loader_puts_the_snapshot_into_state() -> None:
    settings = _settings()
    store = InMemoryStore(settings)
    await store.append(_record("s1", USER, "earlier question"))

    update = await MemoryLoaderNode(store, settings)(RAGState(session_id="s1"))

    memory = update["memory"]
    assert memory is not None
    assert [record.content for record in memory.recent] == ["earlier question"]


async def test_loader_returns_empty_memory_when_disabled() -> None:
    settings = _settings(enabled=False)
    store = InMemoryStore(settings)
    await store.append(_record("s1", USER, "earlier question"))

    update = await MemoryLoaderNode(store, settings)(RAGState(session_id="s1"))

    assert update["memory"] == MemorySnapshot()
    assert update["trace"] == ["memory_loader:disabled"]


async def test_loader_degrades_when_the_store_fails() -> None:
    class BrokenStore(InMemoryStore):
        async def snapshot(self, session_id: str) -> MemorySnapshot:
            raise RuntimeError("store down")

    settings = _settings()
    update = await MemoryLoaderNode(BrokenStore(settings), settings)(RAGState(session_id="s1"))

    assert update["memory"] == MemorySnapshot()
    assert update["trace"] == ["memory_loader:failed"]


# ------------------------------------------------------------------------ writer node


def _writer_state(answer: str = "the answer") -> RAGState:
    return RAGState(session_id="s1", query="the question", answer=answer, memory=MemorySnapshot())


async def test_writer_records_both_sides_of_the_exchange() -> None:
    settings = _settings(extract_user_context=False)
    store = InMemoryStore(settings)
    chat = FakeChatService(text=["a summary"])

    await MemoryWriterNode(store, chat, settings)(_writer_state())  # type: ignore[arg-type]

    history = await store.history("s1")
    assert [(r.role, r.content) for r in history] == [
        (USER, "the question"),
        (ASSISTANT, "the answer"),
    ]


async def test_writer_summarises_only_once_the_threshold_is_reached() -> None:
    settings = _settings(short_term_turns=4, summary_after_turns=4, extract_user_context=False)
    store = InMemoryStore(settings)
    chat = FakeChatService(text=["they discussed leave"])
    writer = MemoryWriterNode(store, chat, settings)  # type: ignore[arg-type]

    await writer(_writer_state())
    assert (await store.snapshot("s1")).summary == ""  # two turns so far

    await writer(_writer_state())
    assert (await store.snapshot("s1")).summary == "they discussed leave"


async def test_writer_extracts_user_facts() -> None:
    settings = _settings(extract_user_context=True)
    store = InMemoryStore(settings)
    chat = FakeChatService(
        text=["a summary"],
        UserContext=[UserContext(facts=(UserFact(key="team", value="finance"),))],
    )

    await MemoryWriterNode(store, chat, settings)(_writer_state())  # type: ignore[arg-type]

    facts = (await store.snapshot("s1")).user_context.facts
    assert facts == (UserFact(key="team", value="finance"),)


async def test_writer_still_records_the_turn_when_summarising_fails() -> None:
    settings = _settings(short_term_turns=2, summary_after_turns=2, extract_user_context=False)
    store = InMemoryStore(settings)
    chat = FakeChatService(text=["unused"])
    chat.fail_on = {"text"}

    update = await MemoryWriterNode(store, chat, settings)(_writer_state())  # type: ignore[arg-type]

    assert len(await store.history("s1")) == 2
    assert (await store.snapshot("s1")).summary == ""
    assert update["trace"] == ["memory_writer:written"]


async def test_writer_does_nothing_when_memory_is_disabled() -> None:
    settings = _settings(enabled=False)
    store = InMemoryStore(settings)
    chat = FakeChatService()

    update = await MemoryWriterNode(store, chat, settings)(_writer_state())  # type: ignore[arg-type]

    assert await store.history("s1") == []
    assert update["trace"] == ["memory_writer:disabled"]


async def test_writer_reports_failure_without_raising() -> None:
    class BrokenStore(InMemoryStore):
        async def append(self, record: MemoryRecord) -> None:
            raise RuntimeError("store down")

    settings = _settings(extract_user_context=False)
    chat = FakeChatService(text=["a summary"])

    update = await MemoryWriterNode(BrokenStore(settings), chat, settings)(  # type: ignore[arg-type]
        _writer_state()
    )

    assert update["trace"] == ["memory_writer:failed"]


# --------------------------------------------------------------------------- settings


def test_summary_cadence_cannot_outpace_the_window() -> None:
    with pytest.raises(ValueError, match="summary_after_turns"):
        MemorySettings(short_term_turns=2, summary_after_turns=10)
