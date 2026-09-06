"""Web search: the interface, and what happens when no backend is configured.

No provider ships here. Choosing one (Tavily, Brave, SerpAPI, an internal
index) is a deployment decision with its own credentials, rate limits, and data
handling rules, so this module defines the contract and leaves the choice open.

`UnavailableWebSearch` is registered by default. It fails every call with a
clear reason rather than being absent from the registry, which keeps the failure
legible: the planner can still choose web search, the tool node records why it
did not work, and the critic sees an explicit gap instead of a mystery.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.core.logging import get_logger
from app.models.base import Schema
from app.tools.common import TextTool, ToolError

_MAX_RESULTS = 5
_NOT_CONFIGURED = (
    "Web search is not configured in this deployment, so no external results are available."
)


class SearchHit(Schema):
    """One result from a web search backend."""

    title: str = ""
    url: str = ""
    snippet: str = ""

    def describe(self) -> str:
        return f"{self.title} ({self.url}): {self.snippet}".strip()


@runtime_checkable
class WebSearchBackend(Protocol):
    """A provider that answers a query with ranked web results."""

    async def search(self, query: str, *, limit: int = _MAX_RESULTS) -> list[SearchHit]: ...


class WebSearchTool(TextTool):
    """Searches the web through whichever backend was supplied."""

    name = "web_search"
    description = "Search the public web for current information not in the indexed documents"
    argument_hint = "a search query"

    def __init__(self, backend: WebSearchBackend, limit: int = _MAX_RESULTS) -> None:
        super().__init__()
        self._backend = backend
        self._limit = limit

    async def execute(self, text: str) -> str:
        query = text.strip()
        if not query:
            raise ToolError("No search query given")
        hits = await self._backend.search(query, limit=self._limit)
        if not hits:
            return f"No web results for: {query}"
        return "\n".join(f"{index}. {hit.describe()}" for index, hit in enumerate(hits, start=1))


class UnavailableWebSearch:
    """The default backend: refuses every call, and says why."""

    def __init__(self) -> None:
        self.logger = get_logger(__name__)

    async def search(self, query: str, *, limit: int = _MAX_RESULTS) -> list[SearchHit]:
        self.logger.info("tools.web_search_unconfigured", query=query)
        raise ToolError(_NOT_CONFIGURED)
