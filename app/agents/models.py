"""The typed artefacts nodes write into the graph state.

These are the only vocabulary the nodes share. A node reads the artefacts it
needs and writes exactly one of its own, which is what keeps the nodes
independent: none of them imports another, and any of them can be replaced by
anything that produces the same artefact.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.models.base import Schema
from app.retrieval.filters import FilterOperator, MetadataFilter, MetadataFilters
from app.retrieval.query import SearchType

QueryIntent = Literal["factual", "comparative", "summarisation", "procedural", "conversational"]


class QueryAnalysis(Schema):
    """What the Query Analyzer determined about the incoming question."""

    intent: QueryIntent = "factual"
    normalised_query: str = ""
    keywords: tuple[str, ...] = ()
    is_ambiguous: bool = False
    reasoning: str = ""

    def search_text(self, fallback: str) -> str:
        """Return the text to search with, preferring the normalised form."""
        return self.normalised_query.strip() or fallback


class PlannedFilter(Schema):
    """A metadata predicate chosen by the planner.

    Values are carried as strings because ingestion writes every metadata value
    as a string; `values` is a list so one shape serves both single-value and
    set-membership operators.
    """

    field: str
    operator: FilterOperator = FilterOperator.EQ
    values: tuple[str, ...] = ()

    def to_filter(self) -> MetadataFilter | None:
        """Convert to a retrieval filter, or None if the planner left it unusable."""
        if not self.field or not self.values:
            return None
        if self.operator.takes_many:
            return MetadataFilter(field=self.field, operator=self.operator, value=self.values)
        return MetadataFilter(field=self.field, operator=self.operator, value=self.values[0])


class PlannedToolCall(Schema):
    """One tool the planner wants invoked before answering.

    A single text `input` rather than a nested argument object: this is produced
    as structured output by a model, and one flat field is the shape that comes
    back reliably.
    """

    tool: str
    input: str = ""
    reason: str = ""


class RetrievalPlan(Schema):
    """The Planner's decision about how — and whether — to retrieve and call tools.

    The two capabilities are independent: a question can need both, either, or
    neither, and the graph routes around whichever is not needed.
    """

    retrieval_needed: bool = True
    search_type: SearchType = SearchType.SIMILARITY
    top_k: int = Field(default=5, ge=1, le=50)
    filters: tuple[PlannedFilter, ...] = ()
    search_text: str = ""
    tools_needed: bool = False
    tool_calls: tuple[PlannedToolCall, ...] = ()
    reasoning: str = ""

    @property
    def wants_tools(self) -> bool:
        """Whether the tool node has anything to do.

        A model that sets the flag but lists no calls has not asked for
        anything, and a model that lists calls without setting the flag clearly
        has; the calls are the substance, the flag is the intent.
        """
        return bool(self.tool_calls) and self.tools_needed

    def to_metadata_filters(self) -> MetadataFilters | None:
        """Return the plan's filters in retrieval form, or None if it set none."""
        conditions = tuple(
            predicate for planned in self.filters if (predicate := planned.to_filter()) is not None
        )
        return MetadataFilters(conditions=conditions) if conditions else None


class Critique(Schema):
    """The Critic's verdict on the generated answer."""

    sufficient_context: bool = True
    grounded: bool = True
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    unsupported_claims: tuple[str, ...] = ()
    feedback: str = ""

    def accepts(self, *, min_confidence: float) -> bool:
        """Whether the answer may be returned as-is."""
        return self.sufficient_context and self.grounded and self.confidence >= min_confidence
