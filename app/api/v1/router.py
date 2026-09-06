"""Aggregates every v1 route module into a single router."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import cache, chat, health, memory, upload

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(upload.router)
api_router.include_router(memory.router)
api_router.include_router(cache.router)
