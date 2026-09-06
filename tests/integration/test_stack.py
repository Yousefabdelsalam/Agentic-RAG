"""Integration tests: real components, wired as production wires them.

Only the two paid network calls are substituted — the OpenAI chat and embedding
endpoints. Everything else is real: the real PDF loader over real bytes, the real
splitter, a real Chroma collection on disk, the real retrievers, the real graph,
and the real HTTP app. These are the tests that catch wiring mistakes the unit
tests cannot, because every unit test has already replaced the seam where the
mistake lives.

Marked `integration`; run the fast suite with `-m "not integration"`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.models import Critique, QueryAnalysis, RetrievalPlan
from app.agents.rag import AgenticRagAgent
from app.api.dependencies import get_agent, get_pipeline
from app.config.settings import Settings
from app.core.bootstrap import build_container
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.main import create_app
from app.memory.store import InMemoryStore
from app.models.base import Schema
from app.retrieval.query import RetrievalQuery, SearchType
from app.services.embeddings import EmbeddingService
from app.services.llm import ChatService
from tests.test_ingestion import _pdf_bytes

pytestmark = pytest.mark.integration

_VOCABULARY = ("leave", "holiday", "sick", "pay", "notice", "remote")


@dataclass
class _Reply:
    """The shape the services read back: text plus reported token usage."""

    content: str
    usage_metadata: dict[str, int]


class StubEmbeddings:
    """Deterministic bag-of-words vectors, in place of the OpenAI endpoint."""

    def _vector(self, text: str) -> list[float]:
        tokens = text.lower().split()
        counts = [
            float(sum(token.strip(".,!?") == word for token in tokens)) for word in _VOCABULARY
        ]
        norm = math.sqrt(sum(value * value for value in counts))
        return [value / norm for value in counts] if norm else [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._vector(text)


class StubChatClient:
    """Scripted replies, in place of the OpenAI chat endpoint."""

    def __init__(self) -> None:
        self.answers: deque[str] = deque(["Staff receive 25 days of leave [1]."])
        self.prompts: list[str] = []

    async def ainvoke(self, messages: list[Any]) -> Any:
        self.prompts.append(str(messages[0].content))

        text = self.answers[0] if len(self.answers) == 1 else self.answers.popleft()
        return _Reply(content=text, usage_metadata={"input_tokens": 100, "output_tokens": 20})

    def with_structured_output(self, schema: type[Schema], include_raw: bool = False) -> Any:
        return _StructuredRunnable(schema, self)


class _StructuredRunnable:
    def __init__(self, schema: type[Schema], client: StubChatClient) -> None:
        self._schema = schema
        self._client = client

    async def ainvoke(self, messages: list[Any]) -> Any:
        self._client.prompts.append(str(messages[0].content))

        raw = _Reply(content="", usage_metadata={"input_tokens": 50, "output_tokens": 10})
        return {"parsed": _reply_for(self._schema), "parsing_error": None, "raw": raw}


def _reply_for(schema: type[Schema]) -> Any:
    if schema is QueryAnalysis:
        return QueryAnalysis(intent="factual", normalised_query="leave policy")
    if schema is RetrievalPlan:
        return RetrievalPlan(
            retrieval_needed=True,
            search_type=SearchType.SIMILARITY,
            top_k=3,
            search_text="leave",
        )
    if schema is Critique:
        return Critique(sufficient_context=True, grounded=True, confidence=0.9)
    return schema()


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        environment="local",
        openai={"api_key": "sk-integration-test"},  # type: ignore[arg-type]
        chroma={"persist_directory": str(tmp_path / "chroma"), "collection": "integration"},  # type: ignore[arg-type]
        ingestion={"chunk_size": 200, "chunk_overlap": 20},  # type: ignore[arg-type]
        retrieval={"top_k": 3, "fetch_k": 5, "compression_enabled": False},  # type: ignore[arg-type]
        memory={"extract_user_context": False},  # type: ignore[arg-type]
        agent={"max_revisions": 1},  # type: ignore[arg-type]
    )


@pytest.fixture
def handbook() -> bytes:
    return _pdf_bytes(
        [
            "Staff receive 25 days of leave each year.",
            "Sick leave requires a doctor note after three days.",
            "Remote work is allowed two days per week.",
        ]
    )


@pytest.fixture
async def stack(tmp_path: Path) -> AsyncIterator[tuple[AsyncClient, Any]]:
    """The production container, with only the two paid clients stubbed."""
    settings = _settings(tmp_path)
    container = build_container(settings)

    embeddings = container.resolve(EmbeddingService)
    chat = container.resolve(ChatService)
    chat_client = StubChatClient()

    app = create_app(settings)
    app.state.container = container

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await container.startup()
        # Substituted after startup so the real construction path still runs.
        embeddings._client = StubEmbeddings()  # type: ignore[assignment]
        chat._client = chat_client  # type: ignore[assignment]
        app.dependency_overrides[get_agent] = lambda: container.resolve(AgenticRagAgent)
        app.dependency_overrides[get_pipeline] = lambda: container.resolve(
            DocumentIngestionPipeline
        )
        try:
            yield client, container
        finally:
            await container.shutdown()


async def _upload(client: AsyncClient, pdf: bytes, name: str = "handbook.pdf") -> Any:
    return await client.post("/api/v1/upload", files={"file": (name, pdf, "application/pdf")})


# --------------------------------------------------------------------------- lifecycle


async def test_the_stack_starts_and_reports_ready(stack: tuple[AsyncClient, Any]) -> None:
    client, _ = stack

    health = await client.get("/api/v1/health")
    ready = await client.get("/api/v1/ready")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["ready"] is True
    assert set(ready.json()["dependencies"]) >= {"EmbeddingService", "ChromaVectorStore"}


# ------------------------------------------------------------------- ingest and answer


async def test_upload_then_ask_answers_from_the_uploaded_document(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    client, _ = stack

    uploaded = await _upload(client, handbook)
    assert uploaded.status_code == 201
    assert uploaded.json()["chunks_indexed"] == 3

    answered = await client.post(
        "/api/v1/chat", json={"query": "How much leave?", "session_id": "s1"}
    )

    assert answered.status_code == 200
    body = answered.json()
    assert body["answer"]
    assert body["citations"], "an answer drawn from documents must cite them"
    assert body["citations"][0]["filename"] == "handbook.pdf"
    assert body["grounded"] is True


async def test_ingestion_is_idempotent_across_uploads(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    from app.retrieval.chroma import ChromaVectorStore

    client, container = stack
    store = container.resolve(ChromaVectorStore)

    await _upload(client, handbook)
    after_first = await store.count()
    await _upload(client, handbook)

    assert await store.count() == after_first


async def test_retrieval_reaches_the_right_page(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    from app.retrieval.factory import RetrieverFactory

    client, container = stack
    await _upload(client, handbook)

    query = RetrievalQuery(text="remote", top_k=1)
    result = await container.resolve(RetrieverFactory).for_query(query).search(query)

    assert result.documents
    assert "Remote work" in result.documents[0].document.content
    assert result.documents[0].document.metadata["page"] == "3"


async def test_metadata_filters_apply_end_to_end(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    from app.retrieval.factory import RetrieverFactory
    from app.retrieval.filters import MetadataFilters

    client, container = stack
    await _upload(client, handbook)

    query = RetrievalQuery(text="leave", top_k=5, filters=MetadataFilters.equals(page="2"))
    result = await container.resolve(RetrieverFactory).for_query(query).search(query)

    assert result.documents
    assert {scored.document.metadata["page"] for scored in result.documents} == {"2"}


# ------------------------------------------------------------------- memory and reset


async def test_a_conversation_is_remembered_then_forgotten(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    client, container = stack
    await _upload(client, handbook)
    store = container.resolve(InMemoryStore)

    await client.post("/api/v1/chat", json={"query": "How much leave?", "session_id": "s1"})
    assert len(await store.history("s1")) == 2

    reset = await client.post("/api/v1/reset-memory", json={"session_id": "s1"})

    assert reset.status_code == 200
    assert await store.history("s1") == []


async def test_the_endpoint_resets_the_memory_the_agent_writes_to(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    """The wiring hazard, verified against the real container rather than a count."""
    client, container = stack
    await _upload(client, handbook)

    await client.post("/api/v1/chat", json={"query": "How much leave?", "session_id": "s2"})
    await client.post("/api/v1/reset-memory", json={"session_id": "s2"})

    assert await container.resolve(InMemoryStore).history("s2") == []


# ------------------------------------------------------------------------- streaming


async def test_streaming_delivers_the_same_answer_as_the_json_path(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    import json

    client, _ = stack
    await _upload(client, handbook)

    plain = await client.post("/api/v1/chat", json={"query": "How much leave?"})
    streamed = await client.post("/api/v1/chat", json={"query": "How much leave?", "stream": True})

    events = [
        json.loads(line.removeprefix("data: "))
        for line in streamed.text.splitlines()
        if line.startswith("data: ")
    ]
    final = events[-1]

    assert final["type"] == "result"
    assert final["result"]["answer"] == plain.json()["answer"]
    assert final["result"]["citations"] == plain.json()["citations"]


# --------------------------------------------------------------------------- caching


async def test_a_repeated_question_reuses_the_cached_query_embedding(
    stack: tuple[AsyncClient, Any], handbook: bytes
) -> None:
    client, container = stack
    await _upload(client, handbook)
    cache = container.resolve(EmbeddingService)._query_cache

    await client.post("/api/v1/chat", json={"query": "How much leave?"})
    hits_before = cache.hits
    await client.post("/api/v1/chat", json={"query": "How much leave?"})

    assert cache.hits > hits_before


# --------------------------------------------------------------------------- failure


async def test_a_corrupt_upload_is_rejected_before_it_reaches_the_index(
    stack: tuple[AsyncClient, Any],
) -> None:
    from app.retrieval.chroma import ChromaVectorStore

    client, container = stack
    store = container.resolve(ChromaVectorStore)

    response = await _upload(client, b"not a pdf at all", name="broken.pdf")

    assert response.status_code == 422
    assert await store.count() == 0


async def test_asking_before_anything_is_ingested_says_so(stack: tuple[AsyncClient, Any]) -> None:
    client, _ = stack

    response = await client.post("/api/v1/chat", json={"query": "How much leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    assert body["sufficient_context"] is False


async def test_an_unsupported_file_type_is_rejected(stack: tuple[AsyncClient, Any]) -> None:
    client, _ = stack

    response = await _upload(client, b"%PDF-1.4 fake", name="notes.txt")

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
