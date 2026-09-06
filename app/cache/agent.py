"""The cache gateway in front of the graph.

Wraps `AgenticRagAgent` behind the same interface, so the API layer, the
streaming endpoint, and the graph itself are all unchanged: the graph does not
know it is being cached, and the transport does not know how.

On a hit, nothing downstream runs. No node executes, no model is called, and the
answer is returned with the citations and verdict it was written with. The one
thing a hit still does is append the turn to conversation memory — a dictionary
write, no model call — because a conversation whose second question depends on
its first must not be missing the first.

**Follow-up questions are not cached.** A question asked mid-conversation may
mean nothing on its own: "and for part-time staff?" is a different question in
every session it appears in, and a cache keyed on the text alone would answer
one session from another's context. Sessions with history therefore bypass the
cache in both directions — no lookup, no store — which leaves the cache holding
exactly the questions that are self-contained.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.agents.base import AgentRequest, AgentResponse
from app.agents.rag import ANSWER, RESULT, STAGE, AgentEvent, AgenticRagAgent
from app.cache.base import NONE, CacheHit
from app.cache.gateway import CacheGateway
from app.config.settings import MemorySettings
from app.core.base import Component
from app.core.logging import get_logger
from app.core.observability import record, traced
from app.memory.base import MemoryRecord
from app.memory.models import ASSISTANT, USER
from app.memory.store import InMemoryStore

#: Stage name reported for a hit, so a streaming client sees the run resolve
#: somewhere rather than receiving a result with no preceding stage.
CACHE_STAGE = "cache"


class CachedRagAgent(Component):
    """Serves known answers from the cache and delegates everything else.

    Holds no lifecycle of its own. The graph, the gateway, and the backend are
    each registered in the container in their own right, so starting this too
    would start them twice.
    """

    name = "cached_agentic_rag"

    def __init__(
        self,
        agent: AgenticRagAgent,
        gateway: CacheGateway,
        memory: InMemoryStore,
        memory_settings: MemorySettings,
    ) -> None:
        self.logger = get_logger(__name__)
        self._agent = agent
        self._gateway = gateway
        self._memory = memory
        self._memory_settings = memory_settings

    @property
    def agent(self) -> AgenticRagAgent:
        """The wrapped graph, for callers that need it unmediated."""
        return self._agent

    @property
    def gateway(self) -> CacheGateway:
        return self._gateway

    @property
    def graph(self) -> Any:
        return self._agent.graph

    @traced("cache.agent.invoke")
    async def invoke(self, request: AgentRequest) -> AgentResponse:
        """Answer from cache if possible, otherwise run the graph and record it."""
        standalone = await self._is_standalone(request)
        hit = await self._lookup(request) if standalone else None
        if hit is not None:
            await self._remember(request, hit.entry.answer)
            return _response_from_hit(request, hit)

        response = await self._agent.invoke(request)
        if standalone:
            await self._gateway.store(request.query, response.answer, response.metadata)
        return _annotated(response, _miss_metadata())

    async def stream_events(self, request: AgentRequest) -> AsyncIterator[AgentEvent]:
        """Stream the run, or emit a hit as a complete run that took one step."""
        standalone = await self._is_standalone(request)
        hit = await self._lookup(request) if standalone else None
        if hit is not None:
            await self._remember(request, hit.entry.answer)
            yield AgentEvent(type=STAGE, stage=CACHE_STAGE)
            yield AgentEvent(type=ANSWER, stage=CACHE_STAGE, answer=hit.entry.answer)
            yield AgentEvent(type=RESULT, answer=hit.entry.answer, metadata=_hit_metadata(hit))
            return

        final: AgentEvent | None = None
        async for event in self._agent.stream_events(request):
            if event.type != RESULT:
                yield event
                continue
            final = event

        if final is None:
            return

        if standalone:
            await self._gateway.store(request.query, final.answer, final.metadata)
        yield AgentEvent(
            type=RESULT,
            stage=final.stage,
            answer=final.answer,
            metadata={**final.metadata, **_miss_metadata()},
        )

    async def stream(self, request: AgentRequest) -> AsyncIterator[str]:
        """Yield each answer the run produces; a hit yields exactly one."""
        async for event in self.stream_events(request):
            if event.type == ANSWER and event.answer:
                yield event.answer

    # ----------------------------------------------------------------- helpers

    async def _lookup(self, request: AgentRequest) -> CacheHit | None:
        """Consult the cache when it is on."""
        if not self._gateway.enabled:
            return None
        return await self._gateway.lookup(request.query)

    async def _is_standalone(self, request: AgentRequest) -> bool:
        """Whether this question can be read without the conversation around it.

        Decided once, before the graph runs, and reused for the store afterwards.
        It cannot be asked again later: the memory writer records this very turn
        during the run, so a check made afterwards would find history for every
        question, including the first, and nothing would ever be cached.
        """
        if not self._gateway.enabled:
            return False
        if not request.session_id or not self._memory_settings.enabled:
            return True
        if await self._memory.history(request.session_id, limit=1):
            record(cache_hit=False, cache_type=NONE, cache_skip_reason="conversation_turn")
            return False
        return True

    async def _remember(self, request: AgentRequest, answer: str) -> None:
        """Append a cache-served turn to memory.

        The graph's memory writer never ran, so this stands in for the part of
        it that costs nothing: both turns are appended, and the summary and user
        facts — which need model calls — are left for the next uncached run.
        """
        if not self._memory_settings.enabled or not request.session_id:
            return
        try:
            for role, content in ((USER, request.query), (ASSISTANT, answer)):
                if content:
                    await self._memory.append(
                        MemoryRecord(session_id=request.session_id, role=role, content=content)
                    )
        except Exception:
            # The answer has already been produced; losing the turn costs the
            # next question some context, and failing here would cost this one
            # an answer that is sitting in hand.
            self.logger.warning("cache.memory_write_failed", session_id=request.session_id)


def _response_from_hit(request: AgentRequest, hit: CacheHit) -> AgentResponse:
    """Rebuild a full response from a cached entry.

    The session is the caller's, not the one that produced the answer: the
    entry is shared, the conversation it is served into is not.
    """
    return AgentResponse(
        session_id=request.session_id,
        answer=hit.entry.answer,
        metadata=_hit_metadata(hit),
    )


def _hit_metadata(hit: CacheHit) -> dict[str, Any]:
    """Present a cached entry in the same shape a live run reports."""
    entry = hit.entry
    return {
        "revisions": entry.revisions,
        "trace": [*entry.trace, f"cache:{hit.cache_type}"],
        "plan": None,
        "critique": {
            "sufficient_context": entry.sufficient_context,
            "grounded": entry.grounded,
            "confidence": entry.confidence,
            "unsupported_claims": [],
            "feedback": "",
        },
        "citations": [dict(citation) for citation in entry.citations],
        "tools": [dict(tool) for tool in entry.tools],
        "cache": {
            "cache_hit": True,
            "cache_type": hit.cache_type,
            "cache_age": round(hit.age_seconds, 3),
            "cache_key": hit.cache_key,
            "similarity": round(hit.similarity, 4),
        },
    }


def _miss_metadata() -> dict[str, Any]:
    return {"cache": {"cache_hit": False, "cache_type": NONE, "cache_age": None}}


def _annotated(response: AgentResponse, extra: dict[str, Any]) -> AgentResponse:
    """Return the response with extra metadata folded in; the model is frozen."""
    return AgentResponse(
        session_id=response.session_id,
        answer=response.answer,
        metadata={**response.metadata, **extra},
    )
