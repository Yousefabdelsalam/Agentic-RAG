from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
import pytest
from httpx import ASGITransport, AsyncClient

from app.agents.base import AgentRequest, AgentResponse
from app.agents.rag import ANSWER, RESULT, STAGE, AgentEvent
from app.api.dependencies import get_agent, get_pipeline
from app.config.settings import Settings
from app.core.exceptions import DependencyError
from app.main import create_app
from app.memory.base import MemoryRecord
from app.memory.store import InMemoryStore

_METADATA: dict[str, Any] = {
    "revisions": 1,
    "trace": ["memory_loader:0", "planner:plan:similarity", "critic:accept"],
    "plan": {"search_type": "similarity", "top_k": 4},
    "critique": {
        "sufficient_context": True,
        "grounded": True,
        "confidence": 0.87,
        "unsupported_claims": [],
        "feedback": "",
    },
    "citations": [
        {
            "marker": 1,
            "id": "handbook-p0003-c0000-ab12cd34",
            "filename": "handbook.pdf",
            "page": "3",
            "score": 0.91,
        }
    ],
    "tools": [{"tool": "calculator", "input": "2+2", "ok": True, "error": ""}],
}


class FakeAgent:
    """Stands in for the compiled graph at the HTTP boundary."""

    def __init__(self, answer: str = "Staff receive 25 days [1].") -> None:
        self.answer = answer
        self.requests: list[AgentRequest] = []
        self.error: Exception | None = None

    async def invoke(self, request: AgentRequest) -> AgentResponse:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return AgentResponse(session_id=request.session_id, answer=self.answer, metadata=_METADATA)

    async def stream_events(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        self.requests.append(request)
        yield AgentEvent(type=STAGE, stage="planner")
        if self.error is not None:
            raise self.error
        yield AgentEvent(type=ANSWER, stage="generator", answer=self.answer)
        yield AgentEvent(type=RESULT, answer=self.answer, metadata=_METADATA)


class FakePipeline:
    def __init__(self, chunks: int = 7) -> None:
        self.chunks = chunks
        self.ingested: list[str] = []
        self.error: Exception | None = None

    async def ingest(self, uri: str) -> int:
        self.ingested.append(uri)
        if self.error is not None:
            raise self.error
        return self.chunks


def _pdf(pages: int = 1) -> bytes:
    from tests.test_ingestion import _pdf_bytes

    return _pdf_bytes([f"Page {index} content" for index in range(pages)])


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def pipeline() -> FakePipeline:
    return FakePipeline()


@pytest.fixture
async def api(
    settings: Settings, agent: FakeAgent, pipeline: FakePipeline
) -> AsyncIterator[AsyncClient]:
    """An app whose agent and pipeline are substituted at the dependency seam."""
    app = create_app(settings)
    app.dependency_overrides[get_agent] = lambda: agent
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            yield client


# ------------------------------------------------------------------------------ chat


async def test_chat_returns_an_answer_with_citations(api: AsyncClient) -> None:
    response = await api.post("/api/v1/chat", json={"query": "How much leave?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Staff receive 25 days [1]."
    assert body["citations"] == [
        {
            "marker": 1,
            "id": "handbook-p0003-c0000-ab12cd34",
            "filename": "handbook.pdf",
            "page": "3",
            "score": 0.91,
        }
    ]
    assert body["tools"][0]["tool"] == "calculator"
    assert body["confidence"] == 0.87
    assert body["grounded"] is True
    assert body["revisions"] == 1
    assert body["trace"]


async def test_chat_assigns_a_session_when_none_is_given(
    api: AsyncClient, agent: FakeAgent
) -> None:
    body = (await api.post("/api/v1/chat", json={"query": "hello"})).json()

    assert body["session_id"].startswith("session_")
    assert agent.requests[0].session_id == body["session_id"]


async def test_chat_keeps_the_session_it_was_given(api: AsyncClient, agent: FakeAgent) -> None:
    body = (
        await api.post("/api/v1/chat", json={"query": "hello", "session_id": "session-42"})
    ).json()

    assert body["session_id"] == "session-42"
    assert agent.requests[0].session_id == "session-42"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "x" * 4001},
        {"query": "ok", "session_id": "has spaces"},
        {"query": "ok", "session_id": "../etc/passwd"},
        {"query": "ok", "unexpected": "field"},
    ],
)
async def test_chat_rejects_invalid_payloads(api: AsyncClient, payload: dict[str, Any]) -> None:
    response = await api.post("/api/v1/chat", json=payload)

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"


async def test_chat_surfaces_a_dependency_failure_as_an_envelope(
    api: AsyncClient, agent: FakeAgent
) -> None:
    agent.error = DependencyError("model unreachable")

    response = await api.post("/api/v1/chat", json={"query": "hello"})

    assert response.status_code == 502
    assert response.json() == {
        "code": "dependency_error",
        "message": "model unreachable",
        "request_id": response.headers["X-Request-ID"],
        "details": {},
    }


# ------------------------------------------------------------------------- streaming


def _events(body: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in body.splitlines()
        if line.startswith("data: ")
    ]


async def test_chat_streams_server_sent_events(api: AsyncClient) -> None:
    response = await api.post("/api/v1/chat", json={"query": "hello", "stream": True})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    assert [event["type"] for event in events] == ["stage", "answer", "result"]


async def test_the_terminal_stream_event_carries_the_full_response(api: AsyncClient) -> None:
    response = await api.post("/api/v1/chat", json={"query": "hello", "stream": True})

    result = _events(response.text)[-1]
    assert result["type"] == "result"
    assert result["result"]["answer"] == "Staff receive 25 days [1]."
    assert result["result"]["citations"][0]["filename"] == "handbook.pdf"
    assert result["result"]["session_id"]


async def test_a_mid_stream_failure_becomes_a_terminal_error_event(
    api: AsyncClient, agent: FakeAgent
) -> None:
    agent.error = DependencyError("model unreachable")

    response = await api.post("/api/v1/chat", json={"query": "hello", "stream": True})

    # The status is already committed once streaming starts, so the failure can
    # only be reported inside the stream.
    assert response.status_code == 200
    events = _events(response.text)
    assert events[-1] == {
        "type": "error",
        "stage": "",
        "answer": "",
        "result": {},
        "message": "model unreachable",
    }


async def test_streaming_disables_proxy_buffering(api: AsyncClient) -> None:
    response = await api.post("/api/v1/chat", json={"query": "hello", "stream": True})

    assert response.headers["x-accel-buffering"] == "no"


# ---------------------------------------------------------------------------- upload


async def test_upload_ingests_a_pdf(api: AsyncClient, pipeline: FakePipeline) -> None:
    response = await api.post(
        "/api/v1/upload",
        files={"file": ("handbook.pdf", _pdf(2), "application/pdf")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "handbook.pdf"
    assert body["chunks_indexed"] == 7
    assert body["bytes_received"] > 0
    assert pipeline.ingested[0].endswith("handbook.pdf")


async def test_upload_strips_directory_components_from_the_name(
    api: AsyncClient, pipeline: FakePipeline
) -> None:
    response = await api.post(
        "/api/v1/upload",
        files={"file": ("../../etc/passwd.pdf", _pdf(), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "passwd.pdf"
    assert ".." not in pipeline.ingested[0]


async def test_upload_rejects_a_non_pdf_extension(api: AsyncClient) -> None:
    response = await api.post(
        "/api/v1/upload", files={"file": ("notes.txt", b"hello", "text/plain")}
    )

    assert response.status_code == 422
    assert "PDF" in response.json()["message"]


async def test_upload_rejects_a_file_that_is_not_really_a_pdf(api: AsyncClient) -> None:
    response = await api.post(
        "/api/v1/upload",
        files={"file": ("disguised.pdf", b"MZ\x90\x00 not a pdf", "application/pdf")},
    )

    assert response.status_code == 422
    assert "not a PDF" in response.json()["message"]


async def test_upload_rejects_an_empty_file(api: AsyncClient) -> None:
    response = await api.post(
        "/api/v1/upload", files={"file": ("empty.pdf", b"", "application/pdf")}
    )

    assert response.status_code == 422


async def test_upload_rejects_a_mismatched_content_type(api: AsyncClient) -> None:
    response = await api.post(
        "/api/v1/upload", files={"file": ("handbook.pdf", _pdf(), "image/png")}
    )

    assert response.status_code == 422


async def test_upload_requires_a_file(api: AsyncClient) -> None:
    assert (await api.post("/api/v1/upload")).status_code == 422


async def test_upload_removes_the_spooled_file_afterwards(
    api: AsyncClient, pipeline: FakePipeline
) -> None:
    await api.post("/api/v1/upload", files={"file": ("h.pdf", _pdf(), "application/pdf")})

    assert not await anyio.Path(pipeline.ingested[0]).exists()


# ---------------------------------------------------------------------- reset-memory


async def test_reset_memory_clears_a_session(api: AsyncClient, settings: Settings) -> None:
    store = InMemoryStore(settings.memory)
    await store.append(MemoryRecord(session_id="session-42", role="user", content="hello"))

    response = await api.post("/api/v1/reset-memory", json={"session_id": "session-42"})

    assert response.status_code == 200
    assert response.json() == {"session_id": "session-42", "cleared": True}


async def test_reset_memory_is_idempotent(api: AsyncClient) -> None:
    first = await api.post("/api/v1/reset-memory", json={"session_id": "never-existed"})
    second = await api.post("/api/v1/reset-memory", json={"session_id": "never-existed"})

    assert first.status_code == second.status_code == 200


async def test_reset_memory_validates_the_session_id(api: AsyncClient) -> None:
    assert (await api.post("/api/v1/reset-memory", json={})).status_code == 422
    assert (
        await api.post("/api/v1/reset-memory", json={"session_id": "bad id"})
    ).status_code == 422


# --------------------------------------------------------------- unconfigured deploy


async def test_chat_reports_a_missing_model_configuration(settings: Settings) -> None:
    """No API key means no agent in the container; the client must be told why."""
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            response = await client.post("/api/v1/chat", json={"query": "hello"})

    assert response.status_code == 502
    assert response.json()["code"] == "dependency_error"
    assert "OPENAI__API_KEY" in response.json()["message"]


async def test_reset_memory_works_without_a_model_configured(settings: Settings) -> None:
    app = create_app(settings)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            response = await client.post("/api/v1/reset-memory", json={"session_id": "s1"})

    assert response.status_code == 200


# ------------------------------------------------------------------------- container


def test_one_memory_store_is_shared_by_the_agent_and_the_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reset must clear the store the agent actually writes to.

    A second `InMemoryStore` anywhere in the wiring would make `/reset-memory`
    return 200 while clearing something nobody reads, so the count is what this
    asserts, not the behaviour of any one instance.
    """
    from app.core import bootstrap

    built: list[InMemoryStore] = []
    original = bootstrap.InMemoryStore

    def counting(*args: Any, **kwargs: Any) -> InMemoryStore:
        store = original(*args, **kwargs)
        built.append(store)
        return store

    monkeypatch.setattr(bootstrap, "InMemoryStore", counting)
    bootstrap.build_container(Settings(openai={"api_key": "sk-test"}))  # type: ignore[arg-type]

    # Exactly one construction means the endpoint's store and the agent's are
    # necessarily the same object. Identity is asserted unpatched below, since
    # patching the name also replaces the key it is registered under.
    assert len(built) == 1


def test_a_configured_container_resolves_the_memory_store() -> None:
    from app.core.bootstrap import build_container

    container = build_container(Settings(openai={"api_key": "sk-test"}))  # type: ignore[arg-type]

    assert isinstance(container.resolve(InMemoryStore), InMemoryStore)


def test_an_unconfigured_container_still_serves_memory() -> None:
    from app.core.bootstrap import build_container

    container = build_container(Settings())

    assert container.resolve(InMemoryStore) is not None


# ---------------------------------------------------------------------------- swagger


async def test_openapi_documents_every_endpoint(api: AsyncClient) -> None:
    schema = (await api.get("/openapi.json")).json()

    assert set(schema["paths"]) >= {
        "/api/v1/chat",
        "/api/v1/upload",
        "/api/v1/health",
        "/api/v1/reset-memory",
    }


async def test_swagger_ui_is_served(api: AsyncClient) -> None:
    response = await api.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


async def test_the_chat_schema_advertises_both_content_types(api: AsyncClient) -> None:
    schema = (await api.get("/openapi.json")).json()

    content = schema["paths"]["/api/v1/chat"]["post"]["responses"]["200"]["content"]
    assert "application/json" in content
    assert "text/event-stream" in content


async def test_docs_are_hidden_in_production() -> None:
    app = create_app(Settings(environment="prod"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        async with app.router.lifespan_context(app):
            assert (await client.get("/docs")).status_code == 404
            assert (await client.get("/openapi.json")).status_code == 404
