"""Liveness and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.dependencies import ContainerDep, SettingsDep
from app.models.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        app=settings.app_name,
        environment=settings.environment,
        version=__version__,
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def ready(container: ContainerDep) -> ReadinessResponse:
    dependencies = await container.health()
    return ReadinessResponse(ready=all(dependencies.values()), dependencies=dependencies)
