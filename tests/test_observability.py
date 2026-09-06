from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any, ClassVar

import pytest
from langsmith import Client
from langsmith.run_helpers import tracing_context

from app.config.settings import LangSmithSettings, ObservabilitySettings
from app.core.costs import DEFAULT_PRICES, ModelPrice, estimate_cost, estimate_tokens
from app.core.observability import (
    record,
    record_error,
    record_outputs,
    record_usage,
    traced,
    truncate,
    usage_of,
)
from app.core.tracing import configure_tracing, tracing_enabled
from app.retrieval.base import Document, ScoredDocument
from app.retrieval.observability import describe_documents, record_query, record_result
from app.retrieval.query import RetrievalQuery, RetrievalResult, SearchType

_TRACING_VARS = (
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_ENDPOINT",
    "LANGSMITH_ENDPOINT",
)


@pytest.fixture(autouse=True)
def clean_environment() -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in _TRACING_VARS}
    for name in _TRACING_VARS:
        os.environ.pop(name, None)
    yield
    for name, value in saved.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


class CapturingClient(Client):
    """A LangSmith client that keeps runs in memory instead of posting them."""

    def __init__(self) -> None:
        super().__init__(
            api_key="test", api_url="https://example.invalid", auto_batch_tracing=False
        )
        self.captured_runs: list[dict[str, Any]] = []

    def create_run(self, *args: Any, **kwargs: Any) -> None:
        self.captured_runs.append(dict(kwargs))

    def update_run(self, run_id: Any, **kwargs: Any) -> None:
        self.captured_runs.append(dict(kwargs, _update=True))

    def merged(self) -> dict[str, Any]:
        """Collapse the create and update calls into one view of the run."""
        merged: dict[str, Any] = {}
        for entry in self.captured_runs:
            for key, value in entry.items():
                if isinstance(value, dict) and isinstance(merged.get(key), dict):
                    merged[key] = {**merged[key], **value}
                elif value is not None:
                    merged[key] = value
        return merged

    def metadata(self) -> dict[str, Any]:
        return dict(self.merged().get("extra", {}).get("metadata", {}))


@pytest.fixture
def captured() -> Iterator[CapturingClient]:
    client = CapturingClient()
    with tracing_context(enabled=True, client=client):
        yield client


# ------------------------------------------------------------------ environment setup


def test_tracing_sets_both_variable_generations() -> None:
    configure_tracing(
        LangSmithSettings(enabled=True, api_key="secret-key", project="my-project")  # type: ignore[arg-type]
    )

    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "secret-key"
    assert os.environ["LANGSMITH_API_KEY"] == "secret-key"
    assert os.environ["LANGCHAIN_PROJECT"] == "my-project"
    assert os.environ["LANGSMITH_PROJECT"] == "my-project"
    assert tracing_enabled()


def test_tracing_is_off_without_an_api_key() -> None:
    configure_tracing(LangSmithSettings(enabled=True, api_key=None))

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert "LANGCHAIN_API_KEY" not in os.environ
    assert not tracing_enabled()


def test_tracing_disabled_by_configuration() -> None:
    configure_tracing(LangSmithSettings(enabled=False, api_key="secret-key"))  # type: ignore[arg-type]

    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert not tracing_enabled()


# ------------------------------------------------------------------------- the decorator


async def test_traced_creates_a_run_and_records_latency(captured: CapturingClient) -> None:
    @traced("unit.work")
    async def work(value: int) -> int:
        return value * 2

    assert await work(21) == 42

    merged = captured.merged()
    assert merged["name"] == "unit.work"
    assert merged["run_type"] == "chain"
    assert captured.metadata()["latency_ms"] >= 0


async def test_traced_does_not_send_arguments(captured: CapturingClient) -> None:
    @traced("unit.secret")
    async def work(password: str) -> str:
        return "done"

    await work("hunter2")

    assert "hunter2" not in str(captured.merged())


async def test_traced_records_errors_and_re_raises(captured: CapturingClient) -> None:
    @traced("unit.broken")
    async def work() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await work()

    metadata = captured.metadata()
    assert metadata["error_type"] == "ValueError"
    assert metadata["error_message"] == "boom"
    assert captured.merged()["error"]


async def test_traced_passes_the_return_value_through_untouched() -> None:
    marker = object()

    @traced("unit.identity")
    async def work() -> object:
        return marker

    assert await work() is marker


async def test_run_type_reaches_langsmith(captured: CapturingClient) -> None:
    @traced("unit.retrieval", run_type="retriever")
    async def work() -> None:
        return None

    await work()

    assert captured.merged()["run_type"] == "retriever"


def test_annotation_helpers_are_inert_without_a_run() -> None:
    record(anything=1)
    record_outputs(anything=1)
    record_error(ValueError("boom"))
    assert record_usage("gpt-4o", input_tokens=1000) > 0


# ------------------------------------------------------------------------------- usage


def test_usage_is_extracted_from_a_message() -> None:
    class Message:
        usage_metadata: ClassVar[dict[str, int]] = {"input_tokens": 120, "output_tokens": 30}

    assert usage_of(Message()) == (120, 30)


def test_missing_usage_reads_as_zero() -> None:
    assert usage_of(object()) == (0, 0)
    assert usage_of(None) == (0, 0)


async def test_usage_and_cost_land_on_the_run(captured: CapturingClient) -> None:
    @traced("unit.llm", run_type="llm")
    async def work() -> None:
        record_usage("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)

    await work()

    metadata = captured.metadata()
    assert metadata["input_tokens"] == 1_000_000
    assert metadata["total_tokens"] == 2_000_000
    assert metadata["estimated_cost_usd"] == pytest.approx(12.50)
    assert metadata["model"] == "gpt-4o"


def test_cost_is_computed_from_the_price_table() -> None:
    assert estimate_cost("gpt-4o", input_tokens=1_000_000) == pytest.approx(2.50)
    assert estimate_cost("gpt-4o", output_tokens=1_000_000) == pytest.approx(10.00)
    assert estimate_cost("text-embedding-3-small", input_tokens=1_000_000) == pytest.approx(0.02)


def test_a_pinned_model_snapshot_prices_as_its_base_model() -> None:
    assert estimate_cost("gpt-4o-2024-11-20", input_tokens=1_000_000) == pytest.approx(2.50)


def test_an_unknown_model_costs_zero_rather_than_a_guess() -> None:
    assert estimate_cost("some-other-model", input_tokens=1_000_000) == 0.0


def test_prices_are_overridable() -> None:
    prices = {**DEFAULT_PRICES, "gpt-4o": ModelPrice(input_usd=99.0)}

    assert estimate_cost("gpt-4o", input_tokens=1_000_000, prices=prices) == pytest.approx(99.0)


def test_embedding_tokens_are_estimated_from_length() -> None:
    assert estimate_tokens(["a" * 400]) == 100
    assert estimate_tokens([]) == 0


async def test_estimated_token_counts_are_flagged(captured: CapturingClient) -> None:
    @traced("unit.embed", run_type="llm")
    async def work() -> None:
        record_usage("text-embedding-3-small", input_tokens=1000, estimated=True)

    await work()

    assert captured.metadata()["token_counts_estimated"] is True


# ------------------------------------------------------------------- retrieval capture


def _scored(identifier: str, content: str) -> ScoredDocument:
    return ScoredDocument(
        document=Document(
            id=identifier, content=content, metadata={"filename": "a.pdf", "page": "2"}
        ),
        score=0.75,
    )


async def test_retrieval_query_parameters_are_recorded(captured: CapturingClient) -> None:
    @traced("unit.retriever", run_type="retriever")
    async def work() -> None:
        record_query(
            RetrievalQuery(text="a question", top_k=7, search_type=SearchType.MMR),
            ObservabilitySettings(),
        )

    await work()

    metadata = captured.metadata()
    assert metadata["query"] == "a question"
    assert metadata["top_k"] == 7
    assert metadata["search_type"] == "mmr"


async def test_retrieved_documents_are_recorded(captured: CapturingClient) -> None:
    result = RetrievalResult(
        documents=(_scored("d1", "first"), _scored("d2", "second")),
        retriever="vector",
        candidates=9,
    )

    @traced("unit.retriever", run_type="retriever")
    async def work() -> None:
        record_result(result, ObservabilitySettings())

    await work()

    merged = captured.merged()
    documents = merged["outputs"]["documents"]
    assert [entry["id"] for entry in documents] == ["d1", "d2"]
    assert documents[0]["page_content"] == "first"
    assert documents[0]["metadata"]["page"] == "2"
    assert captured.metadata()["candidates_considered"] == 9


async def test_document_capture_can_be_switched_off(captured: CapturingClient) -> None:
    result = RetrievalResult(documents=(_scored("d1", "secret text"),), retriever="vector")

    @traced("unit.retriever", run_type="retriever")
    async def work() -> None:
        record_result(result, ObservabilitySettings(capture_documents=False))

    await work()

    assert "secret text" not in str(captured.merged())
    assert captured.metadata()["documents_returned"] == 1


def test_document_capture_respects_both_limits() -> None:
    settings = ObservabilitySettings(max_captured_documents=2, max_captured_characters=4)
    documents = [_scored(f"d{index}", "x" * 50) for index in range(5)]

    described = describe_documents(documents, settings)

    assert len(described) == 2
    assert described[0]["page_content"] == "xxxx... [truncated]"


def test_truncate_marks_what_it_removed() -> None:
    assert truncate("short", 100) == "short"
    assert truncate("x" * 10, 4) == "xxxx... [truncated]"
