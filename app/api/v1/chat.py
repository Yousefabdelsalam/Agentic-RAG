"""The chat endpoint: ask a question, get a cited answer."""

from __future__ import annotations

from collections.abc import AsyncIterator
from http import HTTPStatus
from typing import Any

import orjson
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agents.base import AgentRequest, AgentResponse
from app.agents.rag import RESULT, AgentEvent
from app.api.dependencies import AgentDep
from app.cache.agent import CachedRagAgent
from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.models.chat import ChatRequest, ChatResponse, Citation, StreamEvent, ToolInvocation
from app.models.common import ErrorResponse
from app.utils.ids import new_id

router = APIRouter(tags=["chat"])
logger = get_logger(__name__)

SSE_MEDIA_TYPE = "text/event-stream"
_SESSION_PREFIX = "session"
_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Nginx buffers proxied responses by default, which would hold every event
    # until the run finished and defeat the point of streaming.
    "X-Accel-Buffering": "no",
}


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question",
    description=(
        "Runs the agentic RAG graph and returns a grounded answer with citations.\n\n"
        "Set `stream: true` to receive Server-Sent Events instead: one JSON object "
        "per event, `stage` as each node completes, `answer` when a draft is "
        "written, and a final `result` carrying the same fields as this response."
    ),
    responses={
        HTTPStatus.OK: {
            "content": {
                "application/json": {},
                SSE_MEDIA_TYPE: {"schema": StreamEvent.model_json_schema()},
            }
        },
        HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
        HTTPStatus.BAD_GATEWAY: {"model": ErrorResponse, "description": "Not configured"},
    },
)
async def chat(payload: ChatRequest, agent: AgentDep) -> ChatResponse | StreamingResponse:
    """Answer a question, streaming the run when asked to."""
    request = AgentRequest(
        session_id=payload.session_id or new_id(_SESSION_PREFIX), query=payload.query
    )
    logger.info("api.chat", session_id=request.session_id, stream=payload.stream)

    if payload.stream:
        return StreamingResponse(
            _stream(agent, request),
            media_type=SSE_MEDIA_TYPE,
            headers=_STREAM_HEADERS,
        )
    return to_response(await agent.invoke(request))


async def _stream(agent: CachedRagAgent, request: AgentRequest) -> AsyncIterator[bytes]:
    """Render the run as Server-Sent Events.

    The response status is already committed by the time the first event is
    produced, so a mid-run failure cannot become an HTTP error code. It is sent
    as a terminal `error` event instead, which is the only way a streaming
    client can be told what went wrong.
    """
    try:
        async for event in agent.stream_events(request):
            yield _sse(_render(event, request.session_id))
    except AppError as exc:
        logger.warning("api.chat_stream_failed", session_id=request.session_id, code=exc.code)
        yield _sse(StreamEvent(type="error", message=exc.message))
    except Exception:
        logger.exception("api.chat_stream_failed", session_id=request.session_id)
        yield _sse(StreamEvent(type="error", message="Internal server error."))


def _render(event: AgentEvent, session_id: str) -> StreamEvent:
    if event.type != RESULT:
        return StreamEvent(type="stage" if event.type == "stage" else "answer", **_fields(event))
    response = to_response(
        AgentResponse(session_id=session_id, answer=event.answer, metadata=event.metadata)
    )
    return StreamEvent(type="result", answer=event.answer, result=response.model_dump())


def _fields(event: AgentEvent) -> dict[str, str]:
    return {"stage": event.stage, "answer": event.answer}


def _sse(event: StreamEvent) -> bytes:
    """Frame one event. SSE requires a blank line to terminate the record."""
    return b"data: " + orjson.dumps(event.model_dump()) + b"\n\n"


def to_response(result: AgentResponse) -> ChatResponse:
    """Project the agent's run onto the API contract."""
    metadata: dict[str, Any] = result.metadata
    critique: dict[str, Any] = metadata.get("critique") or {}
    cache: dict[str, Any] = metadata.get("cache") or {}
    return ChatResponse(
        session_id=result.session_id,
        answer=result.answer,
        cache_hit=bool(cache.get("cache_hit", False)),
        cache_type=cache.get("cache_type", "none"),
        cache_age=cache.get("cache_age"),
        citations=[Citation(**citation) for citation in metadata.get("citations", [])],
        tools=[ToolInvocation(**tool) for tool in metadata.get("tools", [])],
        revisions=metadata.get("revisions", 0),
        confidence=critique.get("confidence"),
        grounded=critique.get("grounded"),
        sufficient_context=critique.get("sufficient_context"),
        trace=list(metadata.get("trace", [])),
    )
