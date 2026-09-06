"""Rendering retrieved documents as prompt context.

Shared by the generator and the critic so both see byte-identical context: the
critic checks citations against the same numbering the generator was given, and
a mismatch here would make every grounding verdict meaningless.

It lives outside `nodes/` deliberately — a node importing another node would
couple two stages that are meant to be replaceable independently.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.retrieval.base import ScoredDocument
from app.tools.common import ToolResult

NO_CONTEXT = "No documents were retrieved."
NO_TOOLS = "No tools were used."


def format_context(documents: Sequence[ScoredDocument], *, limit: int) -> str:
    """Render documents as numbered, attributed blocks.

    The provenance line is what makes a citation checkable, so it carries the
    metadata ingestion recorded rather than an opaque chunk id.
    """
    if not documents:
        return NO_CONTEXT

    blocks: list[str] = []
    for position, scored in enumerate(documents[:limit], start=1):
        metadata = scored.document.metadata
        source = metadata.get("filename", scored.document.id)
        page = metadata.get("page")
        attribution = f"{source}, page {page}" if page else source
        blocks.append(f"[{position}] ({attribution})\n{scored.document.content}")
    return "\n\n".join(blocks)


def format_tool_results(results: Sequence[ToolResult]) -> str:
    """Render tool outcomes for a prompt, failures included.

    Failed calls are shown rather than filtered out: a model told only that a
    tool succeeded elsewhere will assume the missing one also did. Seeing the
    failure is what lets it say the information is unavailable.
    """
    if not results:
        return NO_TOOLS
    return "\n".join(f"- {result.describe()}" for result in results)
