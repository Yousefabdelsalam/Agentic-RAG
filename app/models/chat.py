"""Request and response payloads for the v1 API.

These are the API's contract, deliberately separate from the domain models the
graph passes around: a change to `RetrievalPlan` or `Critique` should not silently
alter what clients receive.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import ConfigDict, Field

from app.models.base import RequestSchema, ResponseSchema

MAX_QUERY_CHARACTERS = 4000
MAX_SESSION_CHARACTERS = 128


class ChatRequest(RequestSchema):
    """A question to answer, optionally within an ongoing conversation."""

    query: str = Field(
        min_length=1,
        max_length=MAX_QUERY_CHARACTERS,
        description="The question to answer.",
        examples=["How many days of annual leave do staff get?"],
    )
    session_id: str | None = Field(
        default=None,
        max_length=MAX_SESSION_CHARACTERS,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description=(
            "Conversation this question belongs to. Omit to start a new one; the "
            "id assigned is returned in the response."
        ),
        examples=["session-42"],
    )
    stream: bool = Field(
        default=False,
        description=(
            "Return Server-Sent Events instead of a single JSON body. Each event "
            "is one JSON object: `stage` as nodes complete, `answer` when a draft "
            "is written, and a final `result` carrying the answer and citations."
        ),
    )


class Citation(ResponseSchema):
    """A document the answer drew on, numbered as it was in the prompt."""

    marker: int = Field(description="The [n] marker used in the answer text.")
    id: str = Field(description="Chunk id, stable across re-ingestion.")
    filename: str = ""
    page: str = ""
    score: float = Field(default=0.0, description="Retrieval relevance, 0 to 1.")


class ToolInvocation(ResponseSchema):
    """A tool the agent called while answering."""

    tool: str
    input: str = ""
    ok: bool = True
    error: str = ""


CacheType = Literal["exact", "semantic", "none"]


class ChatResponse(ResponseSchema):
    """An answer, its sources, and an account of how it was produced."""

    session_id: str
    answer: str
    cache_hit: bool = Field(
        default=False, description="Whether this answer came from the cache instead of a fresh run."
    )
    cache_type: CacheType = Field(
        default="none",
        description=(
            "How it was found: `exact` for the same question, `semantic` for an "
            "equivalent one, `none` when the graph ran."
        ),
    )
    cache_age: float | None = Field(
        default=None,
        description="Seconds since the cached answer was produced; null on a miss.",
    )
    citations: list[Citation] = Field(default_factory=list)
    tools: list[ToolInvocation] = Field(default_factory=list)
    revisions: int = Field(
        default=0, description="How many times the critic sent the run back to the planner."
    )
    confidence: float | None = Field(
        default=None, description="The critic's confidence in its own verdict, 0 to 1."
    )
    grounded: bool | None = Field(
        default=None, description="Whether every claim was traceable to the sources."
    )
    sufficient_context: bool | None = Field(
        default=None, description="Whether the sources actually covered the question."
    )
    trace: list[str] = Field(
        default_factory=list, description="Nodes that ran, in order, with their outcomes."
    )


class UploadResponse(ResponseSchema):
    """The outcome of ingesting one document."""

    filename: str
    chunks_indexed: int = Field(description="Chunks written to the vector store.")
    bytes_received: int


class ResetMemoryRequest(RequestSchema):
    """The conversation to forget."""

    session_id: str = Field(
        min_length=1,
        max_length=MAX_SESSION_CHARACTERS,
        pattern=r"^[A-Za-z0-9._:-]+$",
        description="Conversation to clear.",
        examples=["session-42"],
    )


class ResetMemoryResponse(ResponseSchema):
    """Confirmation that a conversation was forgotten."""

    session_id: str
    cleared: bool = True


class CacheStatsResponse(ResponseSchema):
    """How much work the cache has avoided, and how it is configured."""

    # `model_version` is the name the cache contract fixes, so pydantic's
    # reserved `model_` namespace gives way to it here.
    model_config = ConfigDict(extra="ignore", protected_namespaces=())

    enabled: bool
    semantic_enabled: bool
    backend: str = Field(description="Which backend is serving: `redis` or `memory`.")
    ttl_seconds: float
    similarity_threshold: float

    knowledge_base_version: str
    model_version: str
    prompt_version: str

    cache_hits_total: int = 0
    cache_misses_total: int = 0
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    cache_hit_rate: float = Field(default=0.0, description="Hits as a fraction of lookups.")
    estimated_llm_calls_saved: int = Field(
        default=0, description="Model calls not made, at this graph's calls-per-run."
    )
    entries_written: int = 0
    backend_errors_total: int = 0


class StreamEvent(ResponseSchema):
    """One Server-Sent Event from a streaming chat run.

    Documented as a model so the shape appears in the schema; SSE bodies are not
    themselves described by OpenAPI.
    """

    type: Literal["stage", "answer", "result", "error"]
    stage: str = ""
    answer: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
