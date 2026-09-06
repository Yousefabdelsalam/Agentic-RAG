"""Entrypoint for the LangGraph Agent Server.

The compiled graph is not a module-level object anywhere in this application: it
is built in `AgenticRagAgent.__init__` from eight nodes that only
`build_container` knows how to construct. So `langgraph.json` cannot name a
variable, and this factory hands back the very graph the API serves —
`container.resolve(AgenticRagAgent).graph` — rather than compiling a second one.

Nothing here changes the graph, its nodes, or its routing. It is an adapter
between the CLI's "module:variable" contract and this application's container.

The container is built once and reused: it owns a Chroma client, an embeddings
client, and a chat client, and rebuilding that per run would open a new set on
every invocation.
"""

from __future__ import annotations

from typing import Any

import anyio

from app.agents.rag import AgenticRagAgent
from app.config.settings import get_settings
from app.core.container import Container
from app.core.logging import configure_logging, get_logger
from app.core.tracing import configure_tracing

logger = get_logger(__name__)

_container: Container | None = None
_lock = anyio.Lock()


async def _started_container() -> Container:
    """Build and start the application container once, then reuse it."""
    global _container
    async with _lock:
        if _container is None:
            settings = get_settings()
            configure_logging(settings.logging)
            # `create_app` normally does this, and the Agent Server never calls
            # it. Without the translation, `LANGSMITH__API_KEY` stays a setting
            # this application understands and the langsmith SDK does not, so
            # runs served here would go untraced.
            configure_tracing(settings.langsmith)

            # Imported here so that building the container is the first thing
            # that touches the wiring, keeping module import cheap.
            from app.core.bootstrap import build_container

            container = build_container(settings)
            await container.startup()
            _container = container
            logger.info("langgraph.container_started", environment=settings.environment)
    return _container


async def make_graph(config: Any = None) -> Any:
    """Return the compiled Agentic RAG graph the application already runs."""
    container = await _started_container()
    return container.resolve(AgenticRagAgent).graph
