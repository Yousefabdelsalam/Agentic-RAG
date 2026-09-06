"""Metadata filtering: a backend-neutral predicate language and its Chroma form.

Callers (and, later, a planner) express filters against chunk metadata without
knowing Chroma's `where` dialect; `to_chroma` is the only place that dialect is
spoken.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from app.models.base import Schema

FilterValue = str | int | float | bool
Combinator = Literal["and", "or"]

_MULTI_VALUE_OPERATORS = frozenset({"in", "nin"})


class FilterOperator(StrEnum):
    """Comparisons a metadata predicate can express."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NIN = "nin"

    @property
    def chroma_operator(self) -> str:
        return f"${self.value}"

    @property
    def takes_many(self) -> bool:
        return self.value in _MULTI_VALUE_OPERATORS


class MetadataFilter(Schema):
    """A single predicate over one metadata field."""

    field: str = Field(min_length=1)
    value: FilterValue | tuple[FilterValue, ...]
    operator: FilterOperator = FilterOperator.EQ

    @model_validator(mode="after")
    def _value_matches_operator(self) -> MetadataFilter:
        is_sequence = isinstance(self.value, tuple)
        if self.operator.takes_many and not is_sequence:
            raise ValueError(f"{self.operator} expects a tuple of values")
        if not self.operator.takes_many and is_sequence:
            raise ValueError(f"{self.operator} expects a single value")
        return self

    def to_chroma(self) -> dict[str, Any]:
        value: Any = list(self.value) if isinstance(self.value, tuple) else self.value
        return {self.field: {self.operator.chroma_operator: value}}


class MetadataFilters(Schema):
    """A set of predicates combined with a single boolean operator."""

    conditions: tuple[MetadataFilter, ...] = ()
    combinator: Combinator = "and"

    @classmethod
    def equals(cls, **fields: FilterValue) -> MetadataFilters:
        """Shorthand for the common case: every named field must match exactly."""
        return cls(
            conditions=tuple(
                MetadataFilter(field=name, value=value) for name, value in fields.items()
            )
        )

    def to_chroma(self) -> dict[str, Any] | None:
        """Return a Chroma `where` clause, or None when there is nothing to filter.

        Chroma rejects a boolean operator wrapping a single clause, so a lone
        predicate is emitted bare.
        """
        clauses = [condition.to_chroma() for condition in self.conditions]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {f"${self.combinator}": clauses}

    def __bool__(self) -> bool:
        return bool(self.conditions)
