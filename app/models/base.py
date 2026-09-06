"""Base schema types shared by all API and domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Schema(BaseModel):
    """Immutable, strictly-validated model used for domain objects."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class RequestSchema(BaseModel):
    """Inbound payload model: rejects unknown fields to surface client errors."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ResponseSchema(BaseModel):
    """Outbound payload model."""

    model_config = ConfigDict(extra="ignore")
