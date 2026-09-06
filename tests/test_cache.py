from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.base import AgentRequest, AgentResponse
from app.agents.rag import ANSWER, RESULT, STAGE, AgentEvent, AgenticRagAgent
from app.api.dependencies import get_agent
from app.cache.agent import CachedRagAgent
from app.cache.backend import ManagedCacheBackend
from app.cache.base import CachedAnswer, CacheVersions
from app.cache.gateway import CacheGateway
from app.cache.keys import answer_key, normalize_query
from app.cache.memory import InMemoryCacheBackend
from app.cache.metrics import LLM_CALLS_PER_RUN
from app.cache.redis import open_backend
from app.cache.versions import KnowledgeBaseVersion, VersionResolver, prompt_version
from app.config.settings import AgentSettings, CacheSettings, MemorySettings, Settings
from app.core.exceptions import DependencyError
from app.main import create_app
from app.memory.base import MemoryRecord
from app.memory.store import InMemoryStore

QUESTION = "How many days of annual leave do staff get?"
ANSWER_TEXT = "Staff receive 25 days [1]."

#: A run the critic accepted: grounded, sufficient context, confident.
GOOD: dict[str, Any] = {
    "revisions": 0,
    "trace": ["planner:plan:similarity", "critic:accept"],
    "critique": {
        "sufficient_context": True,
        "grounded": True,
        "confidence": 0.9,
        "unsupported_claims": [],
        "feedback": "",
    },
    "citations": [{"marker": 1, "id": "handbook-p3", "filename": "handbook.pdf", "page": "3"}],
    "tools": [],
}


class FakeClock:
    """A clock the test advances by hand."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeEmbeddings:
    """Returns the vector a test declared for a query, and counts the calls."""

    def __init__(self, vectors: dict[str, list[float]] | None = None) -> None:
        self.vectors = vectors or {}
        self.calls: list[str] = []

    async def embed_text(self, text: str) -> list[float]:
        self.calls.append(text)
        return self.vectors.get(text, [0.0, 0.0, 1.0])


class FakeAgent:
    """Stands in for the compiled graph, counting how often it actually runs."""

    def __init__(self, answer: str = ANSWER_TEXT, metadata: dict[str, Any] | None = None) -> None:
        self.answer = answer
        self.metadata = metadata if metadata is not None else GOOD
        self.invocations = 0

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        self.invocations += 1
        return AgentResponse(
            session_id=request.session_id, answer=self.answer, metadata=self.metadata
        )

    async def stream_events(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        self.invocations += 1
        yield AgentEvent(type=STAGE, stage="planner")
        yield AgentEvent(type=ANSWER, stage="generator", answer=self.answer)
        yield AgentEvent(type=RESULT, answer=self.answer, metadata=self.metadata)


class BrokenBackend(InMemoryCacheBackend):
    """A backend that fails the way an unreachable Redis does: by raising."""

    name = "broken"

    async def get(self, key: str) -> str | None:
        raise DependencyError("Redis get failed")

    async def set(self, key: str, value: str, *, ttl_seconds: float) -> None:
        raise DependencyError("Redis set failed")

    async def index_entries(self, namespace: str) -> list[tuple[str, tuple[float, ...]]]:
        raise DependencyError("Redis hgetall failed")


def build_gateway(
    *,
    backend: Any = None,
    embeddings: FakeEmbeddings | None = None,
    clock: FakeClock | None = None,
    model: str = "gpt-4o-mini@t0",
    prompts: str = "abc123",
    **settings: Any,
) -> tuple[CacheGateway, Any, FakeClock]:
    """Assemble a gateway over controllable versions, backend, and clock."""
    store = backend if backend is not None else InMemoryCacheBackend()
    ticker = clock or FakeClock()
    cache_settings = CacheSettings(**settings)

    versions = KnowledgeBaseVersion(store, cache_settings)
    resolver = VersionResolver(versions, model=model, prompts=prompts)

    gateway = CacheGateway(
        cache_settings,
        store,
        resolver,
        AgentSettings(),
        cast(Any, embeddings),
        clock=ticker,
    )
    return gateway, store, ticker


# ------------------------------------------------------------------- normalisation


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("How much leave?", "how much leave"),
        ("  How   much    leave  ", "How much leave."),
        ("How much leave???", "how much leave"),
        ("How much leave’s left", "How much leave's left"),
        ("HOW MUCH LEAVE!", "how much leave"),
    ],
)
def test_normalisation_treats_typographic_differences_as_the_same_question(
    left: str, right: str
) -> None:
    assert normalize_query(left) == normalize_query(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("What is 2+2", "What is 2 2"),
        ("leave for staff", "leave for contractors"),
        ("is it 5", "is it 6"),
    ],
)
def test_normalisation_keeps_questions_that_differ_in_meaning_apart(left: str, right: str) -> None:
    assert normalize_query(left) != normalize_query(right)


# --------------------------------------------------------------------- exact cache


async def test_an_identical_question_is_an_exact_hit() -> None:
    gateway, _, _ = build_gateway()
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    hit = await gateway.lookup(QUESTION)

    assert hit is not None
    assert hit.cache_type == "exact"
    assert hit.entry.answer == ANSWER_TEXT
    assert hit.entry.citations[0]["filename"] == "handbook.pdf"
    assert gateway.metrics.exact_cache_hits == 1


async def test_a_reworded_question_still_hits_the_exact_key() -> None:
    """Normalisation is what makes case and punctuation free."""
    gateway, _, _ = build_gateway()
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    hit = await gateway.lookup("   how many days of ANNUAL leave do staff get  ")

    assert hit is not None
    assert hit.cache_type == "exact"


async def test_a_different_question_is_a_miss() -> None:
    gateway, _, _ = build_gateway(semantic_enabled=False)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await gateway.lookup("What is the notice period?") is None
    assert gateway.metrics.cache_misses_total == 1
    assert gateway.metrics.cache_hits_total == 0


async def test_an_empty_cache_misses_without_raising() -> None:
    gateway, _, _ = build_gateway()

    assert await gateway.lookup(QUESTION) is None


async def test_the_cache_age_reports_how_old_the_answer_is() -> None:
    gateway, _, clock = build_gateway(ttl_seconds=3600)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)
    clock.advance(120)

    hit = await gateway.lookup(QUESTION)

    assert hit is not None
    assert hit.age_seconds == pytest.approx(120.0)


# ------------------------------------------------------------------ semantic cache


async def test_a_semantically_equivalent_question_is_a_semantic_hit() -> None:
    embeddings = FakeEmbeddings(
        {
            normalize_query(QUESTION): [1.0, 0.0, 0.0],
            "what is the annual leave allowance": [0.99, 0.14, 0.0],
        }
    )
    gateway, _, _ = build_gateway(embeddings=embeddings, similarity_threshold=0.92)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    hit = await gateway.lookup("What is the annual leave allowance?")

    assert hit is not None
    assert hit.cache_type == "semantic"
    assert hit.entry.answer == ANSWER_TEXT
    assert hit.similarity >= 0.92
    assert gateway.metrics.semantic_cache_hits == 1


async def test_a_merely_related_question_stays_below_the_threshold() -> None:
    embeddings = FakeEmbeddings(
        {
            normalize_query(QUESTION): [1.0, 0.0, 0.0],
            "what is the notice period": [0.0, 1.0, 0.0],
        }
    )
    gateway, _, _ = build_gateway(embeddings=embeddings, similarity_threshold=0.92)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await gateway.lookup("What is the notice period?") is None
    assert gateway.metrics.cache_misses_total == 1


async def test_the_similarity_threshold_decides_the_verdict() -> None:
    """The same pair of questions, cached or not depending only on the threshold."""
    vectors = {
        normalize_query(QUESTION): [1.0, 0.0, 0.0],
        "roughly how much leave": [0.8, 0.6, 0.0],  # cosine 0.8
    }
    strict, _, _ = build_gateway(
        embeddings=FakeEmbeddings(dict(vectors)), similarity_threshold=0.92
    )
    lenient, _, _ = build_gateway(
        embeddings=FakeEmbeddings(dict(vectors)), similarity_threshold=0.75
    )
    await strict.store(QUESTION, ANSWER_TEXT, GOOD)
    await lenient.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await strict.lookup("Roughly how much leave?") is None
    assert await lenient.lookup("Roughly how much leave?") is not None


async def test_semantic_lookup_is_skipped_when_disabled() -> None:
    embeddings = FakeEmbeddings(
        {
            normalize_query(QUESTION): [1.0, 0.0, 0.0],
            "what is the annual leave allowance": [1.0, 0.0, 0.0],
        }
    )
    gateway, _, _ = build_gateway(embeddings=embeddings, semantic_enabled=False)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await gateway.lookup("What is the annual leave allowance?") is None
    assert gateway.semantic_enabled is False


async def test_semantic_lookup_survives_an_index_entry_whose_answer_expired() -> None:
    """A vector can outlive the answer it points at; that is a miss, not a crash."""
    embeddings = FakeEmbeddings(
        {
            normalize_query(QUESTION): [1.0, 0.0, 0.0],
            "what is the annual leave allowance": [1.0, 0.0, 0.0],
        }
    )
    gateway, backend, _ = build_gateway(embeddings=embeddings)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)
    versions = await gateway.versions()
    await backend.delete(answer_key("agentic-rag", versions, normalize_query(QUESTION)))

    assert await gateway.lookup("What is the annual leave allowance?") is None


# ------------------------------------------------------------------------- expiry


async def test_an_expired_entry_is_not_served() -> None:
    # The backend keeps its own clock deliberately still, so what this asserts is
    # the gateway's TTL check rather than the backend's eviction.
    backend = InMemoryCacheBackend(clock=FakeClock())
    clock = FakeClock()
    gateway, _, _ = build_gateway(backend=backend, clock=clock, ttl_seconds=60)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    clock.advance(61)

    assert await gateway.lookup(QUESTION) is None
    assert gateway.metrics.cache_misses_total == 1


async def test_an_entry_within_its_ttl_is_served() -> None:
    gateway, _, clock = build_gateway(ttl_seconds=60)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    clock.advance(59)

    assert await gateway.lookup(QUESTION) is not None


async def test_an_expired_entry_is_deleted_on_the_way_past() -> None:
    backend = InMemoryCacheBackend(clock=FakeClock())
    clock = FakeClock()
    gateway, _, _ = build_gateway(backend=backend, clock=clock, ttl_seconds=60)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)
    versions = await gateway.versions()
    key = answer_key("agentic-rag", versions, normalize_query(QUESTION))

    clock.advance(61)
    await gateway.lookup(QUESTION)

    assert await backend.get(key) is None


# ---------------------------------------------------------------------- versioning


async def test_ingesting_a_document_makes_earlier_answers_unreachable() -> None:
    backend = InMemoryCacheBackend()
    settings = CacheSettings()
    knowledge_base = KnowledgeBaseVersion(backend, settings)
    resolver = VersionResolver(knowledge_base, model="m1", prompts="p1")
    gateway = CacheGateway(settings, backend, resolver, AgentSettings())
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)
    assert await gateway.lookup(QUESTION) is not None

    await knowledge_base.bump(reason="handbook.pdf")

    assert await gateway.lookup(QUESTION) is None


async def test_the_knowledge_base_version_increments() -> None:
    backend = InMemoryCacheBackend()
    knowledge_base = KnowledgeBaseVersion(backend, CacheSettings())

    assert await knowledge_base.current() == "0"
    assert await knowledge_base.bump() == "1"
    assert await knowledge_base.bump() == "2"
    assert await knowledge_base.current() == "2"


async def test_a_new_model_does_not_read_the_old_models_answers() -> None:
    backend = InMemoryCacheBackend()
    before, _, _ = build_gateway(backend=backend, model="gpt-4o-mini@t0")
    after, _, _ = build_gateway(backend=backend, model="gpt-4o@t0")
    await before.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await before.lookup(QUESTION) is not None
    assert await after.lookup(QUESTION) is None


async def test_an_edited_prompt_does_not_read_the_old_prompts_answers() -> None:
    backend = InMemoryCacheBackend()
    before, _, _ = build_gateway(backend=backend, prompts="prompts-v1")
    after, _, _ = build_gateway(backend=backend, prompts="prompts-v2")
    await before.store(QUESTION, ANSWER_TEXT, GOOD)

    assert await before.lookup(QUESTION) is not None
    assert await after.lookup(QUESTION) is None


@pytest.mark.parametrize("stale", ["knowledge_base_version", "model_version", "prompt_version"])
async def test_an_entry_stamped_with_the_wrong_version_is_refused(stale: str) -> None:
    """The key already namespaces by version; this is the check behind that one.

    An entry can reach the live key from a replica running different code, so
    the versions inside the payload are verified after it is read, not assumed
    from where it was found.
    """
    gateway, backend, clock = build_gateway()
    versions = await gateway.versions()
    normalized = normalize_query(QUESTION)
    key = answer_key("agentic-rag", versions, normalized)

    fields = versions.model_dump() | {stale: "superseded"}
    poisoned = CachedAnswer(
        query=QUESTION,
        normalized_query=normalized,
        answer=ANSWER_TEXT,
        created_at=clock.now,
        expires_at=clock.now + 3600,
        **fields,
    )
    await backend.set(key, poisoned.model_dump_json(), ttl_seconds=3600)

    assert await gateway.lookup(QUESTION) is None
    assert await backend.get(key) is None


def test_the_prompt_version_tracks_the_template_files(tmp_path: Any) -> None:
    (tmp_path / "critic.md").write_text("judge the answer", encoding="utf-8")
    first = prompt_version(tmp_path)

    (tmp_path / "critic.md").write_text("judge the answer harshly", encoding="utf-8")

    assert prompt_version(tmp_path) != first


def test_versions_with_the_same_parts_share_a_fingerprint() -> None:
    left = CacheVersions(knowledge_base_version="1", model_version="m", prompt_version="p")
    right = CacheVersions(knowledge_base_version="1", model_version="m", prompt_version="p")
    other = CacheVersions(knowledge_base_version="2", model_version="m", prompt_version="p")

    assert left.fingerprint == right.fingerprint
    assert left.fingerprint != other.fingerprint


# ------------------------------------------------------------ what is not cached


@pytest.mark.parametrize(
    ("reason", "answer", "metadata"),
    [
        (
            "insufficient context",
            ANSWER_TEXT,
            GOOD | {"critique": GOOD["critique"] | {"sufficient_context": False}},
        ),
        (
            "not grounded",
            ANSWER_TEXT,
            GOOD | {"critique": GOOD["critique"] | {"grounded": False}},
        ),
        (
            "low confidence",
            ANSWER_TEXT,
            GOOD | {"critique": GOOD["critique"] | {"confidence": 0.1}},
        ),
        (
            "failed tool call",
            ANSWER_TEXT,
            GOOD | {"tools": [{"tool": "web_search", "ok": False, "error": "no backend"}]},
        ),
        ("no critique at all", ANSWER_TEXT, {"citations": [], "tools": []}),
        ("an empty answer", "   ", GOOD),
    ],
)
async def test_an_answer_that_did_not_earn_it_is_not_cached(
    reason: str, answer: str, metadata: dict[str, Any]
) -> None:
    gateway, _, _ = build_gateway()

    assert await gateway.store(QUESTION, answer, metadata) is False, reason
    assert await gateway.lookup(QUESTION) is None
    assert gateway.metrics.entries_written == 0


async def test_an_accepted_answer_is_cached() -> None:
    gateway, _, _ = build_gateway()

    assert await gateway.store(QUESTION, ANSWER_TEXT, GOOD) is True
    assert gateway.metrics.entries_written == 1


async def test_a_successful_tool_call_does_not_block_caching() -> None:
    gateway, _, _ = build_gateway()
    metadata = GOOD | {"tools": [{"tool": "calculator", "input": "2+2", "ok": True}]}

    assert await gateway.store(QUESTION, ANSWER_TEXT, metadata) is True


# ------------------------------------------------------------------ disabled cache


async def test_a_disabled_cache_never_stores_or_serves() -> None:
    gateway, _, _ = build_gateway(enabled=False)

    assert await gateway.store(QUESTION, ANSWER_TEXT, GOOD) is False
    assert await gateway.lookup(QUESTION) is None
    assert gateway.enabled is False


async def test_a_disabled_cache_records_no_misses() -> None:
    """A lookup that never happened is not a miss; counting it would libel the cache."""
    gateway, _, _ = build_gateway(enabled=False)

    await gateway.lookup(QUESTION)

    assert gateway.metrics.cache_misses_total == 0
    assert gateway.metrics.cache_hit_rate == 0.0


async def test_a_disabled_cache_leaves_the_agent_running_every_question() -> None:
    agent, cached = _cached_agent(enabled=False)

    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))
    await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert agent.invocations == 2


# ------------------------------------------------------------- backend unavailable


async def test_an_unreachable_redis_falls_back_to_memory_at_startup() -> None:
    settings = CacheSettings(redis_url="redis://127.0.0.1:6399/0", redis_timeout_seconds=0.05)

    backend = await open_backend(settings)

    assert backend.name == "memory"


async def test_no_redis_url_uses_memory_without_trying() -> None:
    backend = await open_backend(CacheSettings())

    assert backend.name == "memory"


async def test_a_backend_that_fails_mid_flight_degrades_to_a_miss() -> None:
    gateway, _, _ = build_gateway(backend=BrokenBackend())

    assert await gateway.lookup(QUESTION) is None
    assert gateway.metrics.backend_errors_total == 1
    assert gateway.metrics.cache_misses_total == 1


async def test_a_failed_write_is_reported_without_raising() -> None:
    gateway, _, _ = build_gateway(backend=BrokenBackend())

    assert await gateway.store(QUESTION, ANSWER_TEXT, GOOD) is False
    assert gateway.metrics.backend_errors_total == 1


async def test_a_failing_backend_still_lets_the_agent_answer() -> None:
    agent, cached = _cached_agent(backend=BrokenBackend())

    response = await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    assert response.answer == ANSWER_TEXT
    assert agent.invocations == 1


async def test_a_failing_embedding_call_degrades_to_a_miss() -> None:
    class BrokenEmbeddings(FakeEmbeddings):
        async def embed_text(self, text: str) -> list[float]:
            raise DependencyError("Embedding request failed")

    working = FakeEmbeddings({normalize_query(QUESTION): [1.0, 0.0, 0.0]})
    gateway, backend, _ = build_gateway(embeddings=working)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    broken, _, _ = build_gateway(backend=backend, embeddings=cast(Any, BrokenEmbeddings()))

    assert await broken.lookup("something else entirely") is None
    assert broken.metrics.backend_errors_total == 1


# ----------------------------------------------------------- the agent in front


def _cached_agent(
    *,
    enabled: bool = True,
    backend: Any = None,
    embeddings: FakeEmbeddings | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[FakeAgent, CachedRagAgent]:
    gateway, _, _ = build_gateway(backend=backend, embeddings=embeddings, enabled=enabled)
    agent = FakeAgent(metadata=metadata)
    memory = InMemoryStore(MemorySettings())
    wrapped = CachedRagAgent(cast(AgenticRagAgent, agent), gateway, memory, MemorySettings())
    return agent, wrapped


async def test_a_cache_hit_never_reaches_the_graph() -> None:
    agent, cached = _cached_agent()

    first = await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))
    second = await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert agent.invocations == 1, "the second question must not run the graph"
    assert second.answer == first.answer
    assert second.metadata["cache"]["cache_hit"] is True
    assert second.metadata["cache"]["cache_type"] == "exact"


async def test_a_miss_runs_the_graph_and_reports_no_cache() -> None:
    agent, cached = _cached_agent()

    response = await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    assert agent.invocations == 1
    assert response.metadata["cache"] == {
        "cache_hit": False,
        "cache_type": "none",
        "cache_age": None,
    }


async def test_a_cached_answer_keeps_the_callers_own_session() -> None:
    _, cached = _cached_agent()
    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    response = await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert response.session_id == "s2"


async def test_a_cached_answer_carries_its_citations_and_verdict() -> None:
    _, cached = _cached_agent()
    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    response = await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert response.metadata["citations"][0]["filename"] == "handbook.pdf"
    assert response.metadata["critique"]["grounded"] is True
    assert response.metadata["critique"]["confidence"] == 0.9
    assert response.metadata["trace"][-1] == "cache:exact"


async def test_a_follow_up_question_bypasses_the_cache_in_both_directions() -> None:
    """A question asked mid-conversation may mean nothing on its own.

    The session is given a turn directly rather than by running the graph,
    because it is the presence of history that this guard keys on, and the fake
    graph does not write any.
    """
    gateway, _, _ = build_gateway()
    agent = FakeAgent()
    memory = InMemoryStore(MemorySettings())
    cached = CachedRagAgent(cast(AgenticRagAgent, agent), gateway, memory, MemorySettings())
    await cached.invoke(AgentRequest(session_id="fresh", query=QUESTION))
    assert gateway.metrics.entries_written == 1

    await memory.append(MemoryRecord(session_id="ongoing", role="user", content="earlier turn"))
    await cached.invoke(AgentRequest(session_id="ongoing", query=QUESTION))

    assert agent.invocations == 2, "a question with history behind it must not read the cache"


async def test_the_first_question_is_cached_even_though_the_graph_records_it() -> None:
    """The real graph writes the turn to memory before the store is offered.

    Asking "does this session have history?" after the run therefore answers yes
    for every question, first or not. Deciding once, before the run, is what
    keeps the first question of a conversation cacheable.
    """
    gateway, _, _ = build_gateway()
    memory = InMemoryStore(MemorySettings())

    class MemoryWritingAgent(FakeAgent):
        """A graph that appends the turn as it answers, as the real one does."""

        async def invoke(self, request: AgentRequest) -> AgentResponse:
            response = await super().invoke(request)
            await memory.append(
                MemoryRecord(session_id=request.session_id, role="user", content=request.query)
            )
            await memory.append(
                MemoryRecord(
                    session_id=request.session_id, role="assistant", content=response.answer
                )
            )
            return response

    agent = MemoryWritingAgent()
    cached = CachedRagAgent(cast(AgenticRagAgent, agent), gateway, memory, MemorySettings())

    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))
    second = await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert gateway.metrics.entries_written == 1
    assert agent.invocations == 1
    assert second.metadata["cache"]["cache_hit"] is True


async def test_a_follow_up_answer_is_not_written_to_the_cache() -> None:
    """A context-dependent answer must not become every session's answer."""
    gateway, _, _ = build_gateway()
    agent = FakeAgent()
    memory = InMemoryStore(MemorySettings())
    cached = CachedRagAgent(cast(AgenticRagAgent, agent), gateway, memory, MemorySettings())

    await memory.append(MemoryRecord(session_id="ongoing", role="user", content="earlier turn"))
    await cached.invoke(AgentRequest(session_id="ongoing", query=QUESTION))

    assert gateway.metrics.entries_written == 0
    assert await gateway.lookup(QUESTION) is None


async def test_a_hit_still_records_the_turn_in_memory() -> None:
    gateway, _, _ = build_gateway()
    agent = FakeAgent()
    memory = InMemoryStore(MemorySettings())
    cached = CachedRagAgent(cast(AgenticRagAgent, agent), gateway, memory, MemorySettings())
    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    history = await memory.history("s2")
    assert [record.role for record in history] == ["user", "assistant"]
    assert history[1].content == ANSWER_TEXT


async def test_a_rejected_answer_leaves_the_next_question_to_the_graph() -> None:
    rejected = GOOD | {"critique": GOOD["critique"] | {"grounded": False}}
    agent, cached = _cached_agent(metadata=rejected)

    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))
    await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert agent.invocations == 2


async def test_streaming_serves_a_hit_as_a_complete_run() -> None:
    agent, cached = _cached_agent()
    await cached.invoke(AgentRequest(session_id="s1", query=QUESTION))

    events = [
        event async for event in cached.stream_events(AgentRequest(session_id="s2", query=QUESTION))
    ]

    assert agent.invocations == 1
    assert [event.type for event in events] == ["stage", "answer", "result"]
    assert events[-1].metadata["cache"]["cache_hit"] is True
    assert events[-1].answer == ANSWER_TEXT


async def test_streaming_a_miss_stores_the_answer_for_next_time() -> None:
    agent, cached = _cached_agent()

    async for _ in cached.stream_events(AgentRequest(session_id="s1", query=QUESTION)):
        pass
    second = await cached.invoke(AgentRequest(session_id="s2", query=QUESTION))

    assert agent.invocations == 1
    assert second.metadata["cache"]["cache_hit"] is True


# ------------------------------------------------------------------------ metrics


async def test_metrics_count_hits_misses_and_the_calls_they_saved() -> None:
    gateway, _, _ = build_gateway(semantic_enabled=False)
    await gateway.store(QUESTION, ANSWER_TEXT, GOOD)

    await gateway.lookup(QUESTION)
    await gateway.lookup(QUESTION)
    await gateway.lookup("something else")

    metrics = gateway.metrics
    assert metrics.cache_hits_total == 2
    assert metrics.cache_misses_total == 1
    assert metrics.exact_cache_hits == 2
    assert metrics.semantic_cache_hits == 0
    assert metrics.cache_hit_rate == pytest.approx(2 / 3)
    assert metrics.estimated_llm_calls_saved == 2 * LLM_CALLS_PER_RUN


def test_the_hit_rate_of_an_idle_cache_is_zero_not_undefined() -> None:
    gateway, _, _ = build_gateway()

    assert gateway.metrics.cache_hit_rate == 0.0
    assert gateway.metrics.snapshot()["cache_hits_total"] == 0


# ------------------------------------------------------------------ the HTTP layer


@pytest.fixture
async def cached_api(settings: Settings) -> AsyncIterator[tuple[AsyncClient, FakeAgent]]:
    """The real app, with only the graph itself substituted."""
    agent, cached = _cached_agent()
    app = create_app(settings)
    app.dependency_overrides[get_agent] = lambda: cached
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            yield client, agent


async def test_the_response_reports_a_miss(
    cached_api: tuple[AsyncClient, FakeAgent],
) -> None:
    client, _ = cached_api

    body = (await client.post("/api/v1/chat", json={"query": QUESTION})).json()

    assert body["cache_hit"] is False
    assert body["cache_type"] == "none"
    assert body["cache_age"] is None


async def test_the_response_reports_a_hit_with_its_age(
    cached_api: tuple[AsyncClient, FakeAgent],
) -> None:
    client, agent = cached_api
    await client.post("/api/v1/chat", json={"query": QUESTION, "session_id": "s1"})

    body = (await client.post("/api/v1/chat", json={"query": QUESTION, "session_id": "s2"})).json()

    assert agent.invocations == 1
    assert body["cache_hit"] is True
    assert body["cache_type"] == "exact"
    assert body["cache_age"] is not None
    assert body["answer"] == ANSWER_TEXT
    assert body["citations"][0]["filename"] == "handbook.pdf"


async def test_the_stats_endpoint_reports_counters_and_versions(settings: Settings) -> None:
    app = create_app(Settings(openai={"api_key": "sk-test"}))  # type: ignore[arg-type]
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            body = (await client.get("/api/v1/cache/stats")).json()

    assert body["enabled"] is True
    assert body["backend"] == "memory"
    assert body["knowledge_base_version"] == "0"
    assert body["model_version"].startswith("gpt-")
    assert body["prompt_version"]
    assert body["cache_hits_total"] == 0
    assert body["cache_hit_rate"] == 0.0


async def test_the_stats_endpoint_explains_an_unconfigured_deployment(
    settings: Settings,
) -> None:
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            response = await client.get("/api/v1/cache/stats")

    assert response.status_code == 502
    assert "OPENAI__API_KEY" in response.json()["message"]


# ------------------------------------------------------------------------- wiring


async def test_ingesting_a_document_advances_the_version_the_cache_reads() -> None:
    """The pipeline and the gateway must agree on what the corpus version is."""
    from app.ingestion.pipeline import DocumentIngestionPipeline

    backend = ManagedCacheBackend(CacheSettings())
    knowledge_base = KnowledgeBaseVersion(backend, CacheSettings())

    class StubIndexer:
        async def index(self, chunks: Any) -> int:
            return 3

    class StubLoader:
        async def read(self, uri: str) -> list[Any]:
            return ["page"]

    class StubChunker:
        def chunk(self, documents: list[Any]) -> list[Any]:
            return documents

    pipeline = DocumentIngestionPipeline(
        cast(Any, StubLoader()), cast(Any, StubChunker()), cast(Any, StubIndexer()), knowledge_base
    )

    assert await knowledge_base.current() == "0"
    await pipeline.ingest("handbook.pdf")
    assert await knowledge_base.current() == "1"


async def test_a_run_that_indexed_nothing_does_not_advance_the_version() -> None:
    from app.ingestion.pipeline import DocumentIngestionPipeline

    backend = InMemoryCacheBackend()
    knowledge_base = KnowledgeBaseVersion(backend, CacheSettings())

    class EmptyIndexer:
        async def index(self, chunks: Any) -> int:
            return 0

    class StubLoader:
        async def read(self, uri: str) -> list[Any]:
            return []

    class StubChunker:
        def chunk(self, documents: list[Any]) -> list[Any]:
            return documents

    pipeline = DocumentIngestionPipeline(
        cast(Any, StubLoader()), cast(Any, StubChunker()), cast(Any, EmptyIndexer()), knowledge_base
    )
    await pipeline.ingest("empty.pdf")

    assert await knowledge_base.current() == "0"


def test_the_configured_container_wires_the_cache_in_front_of_the_agent() -> None:
    from app.core.bootstrap import build_container

    container = build_container(Settings(openai={"api_key": "sk-test"}))  # type: ignore[arg-type]

    cached = container.resolve(CachedRagAgent)
    assert cached.agent is container.resolve(AgenticRagAgent)
    assert cached.gateway is container.resolve(CacheGateway)


async def test_ingestion_and_the_cache_share_one_knowledge_base_version() -> None:
    """The seam between "a document was indexed" and "answers are now stale".

    Both halves are tested on their own; what this asserts is that the container
    hands them the same object, so a bump on the ingestion side is a bump the
    gateway reads.
    """
    from app.core.bootstrap import build_container

    container = build_container(Settings(openai={"api_key": "sk-test"}))  # type: ignore[arg-type]
    gateway = container.resolve(CacheGateway)
    knowledge_base = container.resolve(KnowledgeBaseVersion)

    before = (await gateway.versions()).knowledge_base_version
    await knowledge_base.bump(reason="handbook.pdf")

    assert (await gateway.versions()).knowledge_base_version != before


def test_the_cache_reads_its_own_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_CACHE_ENABLED", "false")
    monkeypatch.setenv("RAG_CACHE_TTL_SECONDS", "60")
    monkeypatch.setenv("RAG_CACHE_SEMANTIC_ENABLED", "false")
    monkeypatch.setenv("RAG_CACHE_SIMILARITY_THRESHOLD", "0.8")

    cache = Settings().cache

    assert cache.enabled is False
    assert cache.ttl_seconds == 60
    assert cache.semantic_enabled is False
    assert cache.similarity_threshold == 0.8
