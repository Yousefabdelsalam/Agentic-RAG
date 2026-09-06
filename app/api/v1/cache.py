"""The cache endpoint: what the cache is doing and what it is doing it under."""

from __future__ import annotations

from http import HTTPStatus

from fastapi import APIRouter

from app.api.dependencies import CacheDep
from app.core.logging import get_logger
from app.models.chat import CacheStatsResponse
from app.models.common import ErrorResponse

router = APIRouter(tags=["cache"])
logger = get_logger(__name__)


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache counters and versions",
    description=(
        "Hit and miss counts since this process started, alongside the three "
        "versions a cached answer must match to be served.\n\n"
        "Counters are per process and reset with it. `estimated_llm_calls_saved` "
        "is a lower bound: it prices every hit at one graph run's minimum number "
        "of model calls, so a run the critic sent back counts as costing no more "
        "than one that did not."
    ),
    responses={HTTPStatus.BAD_GATEWAY: {"model": ErrorResponse, "description": "Not configured"}},
)
async def cache_stats(cache: CacheDep) -> CacheStatsResponse:
    """Report the cache's counters, configuration, and current versions."""
    versions = await cache.versions()
    settings = cache.settings
    metrics = cache.metrics
    return CacheStatsResponse(
        enabled=cache.enabled,
        semantic_enabled=cache.semantic_enabled,
        backend=cache.backend_name,
        ttl_seconds=settings.ttl_seconds,
        similarity_threshold=settings.similarity_threshold,
        knowledge_base_version=versions.knowledge_base_version,
        model_version=versions.model_version,
        prompt_version=versions.prompt_version,
        cache_hits_total=metrics.cache_hits_total,
        cache_misses_total=metrics.cache_misses_total,
        exact_cache_hits=metrics.exact_cache_hits,
        semantic_cache_hits=metrics.semantic_cache_hits,
        cache_hit_rate=round(metrics.cache_hit_rate, 4),
        estimated_llm_calls_saved=metrics.estimated_llm_calls_saved,
        entries_written=metrics.entries_written,
        backend_errors_total=metrics.backend_errors_total,
    )
