"""Identifier generation."""

from __future__ import annotations

import uuid


def new_id(prefix: str | None = None) -> str:
    """Return a short, URL-safe unique id, optionally namespaced."""
    value = uuid.uuid4().hex
    return f"{prefix}_{value}" if prefix else value
