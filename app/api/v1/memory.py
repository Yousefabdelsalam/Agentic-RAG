"""The memory endpoint: forget a conversation."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter

from app.api.dependencies import MemoryDep
from app.core.logging import get_logger
from app.models.chat import ResetMemoryRequest, ResetMemoryResponse
from app.models.common import ErrorResponse

router = APIRouter(tags=["memory"])
logger = get_logger(__name__)


@router.post(
    "/reset-memory",
    response_model=ResetMemoryResponse,
    summary="Clear a conversation",
    description=(
        "Forgets a session's short-term turns, its rolling summary, and the user "
        "facts learned from it. Indexed documents are unaffected.\n\n"
        "Clearing a session that does not exist succeeds: the caller's intent is "
        "that nothing be remembered, and after this call nothing is."
    ),
    responses={HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse}},
)
async def reset_memory(payload: ResetMemoryRequest, memory: MemoryDep) -> ResetMemoryResponse:
    """Clear everything remembered about one conversation."""
    await memory.clear(payload.session_id)
    logger.info("api.memory_reset", session_id=payload.session_id)
    return ResetMemoryResponse(session_id=payload.session_id)
