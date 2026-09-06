"""Node 3a — Tool Executor: run the tools the planner asked for."""

from __future__ import annotations

import anyio

from app.agents.models import PlannedToolCall
from app.agents.state import RAGState
from app.config.settings import ToolSettings
from app.core.base import Component
from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.core.observability import record, record_outputs, traced
from app.tools.base import ToolRegistry
from app.tools.common import INPUT_ARGUMENT, ToolError, ToolResult

NODE = "tools"
_TIMED_OUT = "Tool did not finish in time"


class ToolNode(Component):
    """Executes planned tool calls and records their outcomes.

    Reads `plan`; writes `tool_results`. Calls run sequentially: there are at
    most a handful, and a stable order makes the results reproducible in a trace
    and in a prompt.

    No failure escapes this node. A tool that errors, times out, or does not
    exist becomes a failed `ToolResult`, which the generator can see and account
    for. Raising instead would lose an answer that retrieval alone could have
    supported.
    """

    def __init__(self, registry: ToolRegistry, settings: ToolSettings) -> None:
        self.logger = get_logger(__name__)
        self._registry = registry
        self._settings = settings

    @traced(f"node.{NODE}")
    async def __call__(self, state: RAGState) -> RAGState:
        plan = state.get("plan")
        calls = list(plan.tool_calls) if plan else []
        record(node=NODE, requested=len(calls), enabled=self._settings.enabled)

        if not self._settings.enabled or not calls:
            return RAGState(tool_results=[], trace=[f"{NODE}:skipped"])

        allowed = calls[: self._settings.max_calls_per_turn]
        if len(allowed) < len(calls):
            self.logger.info("agent.tool_calls_capped", requested=len(calls), allowed=len(allowed))

        results = [await self._invoke(call) for call in allowed]
        succeeded = sum(1 for result in results if result.ok)

        self.logger.info("agent.tools_executed", calls=len(results), succeeded=succeeded)
        record_outputs(
            calls=len(results),
            succeeded=succeeded,
            results=[result.describe() for result in results],
        )
        return RAGState(
            tool_results=results,
            trace=[f"{NODE}:{succeeded}/{len(results)}"],
        )

    async def _invoke(self, call: PlannedToolCall) -> ToolResult:
        """Run one planned call, converting every failure into a result."""
        try:
            tool = self._registry.get(call.tool)
        except NotFoundError:
            self.logger.warning("agent.tool_unknown", tool=call.tool)
            return ToolResult(
                tool=call.tool, input=call.input, ok=False, error=f"Unknown tool: {call.tool}"
            )

        try:
            with anyio.fail_after(self._settings.timeout_seconds):
                output = await tool.run(**{INPUT_ARGUMENT: call.input})
        except TimeoutError:
            self.logger.warning("agent.tool_timeout", tool=call.tool)
            return ToolResult(tool=call.tool, input=call.input, ok=False, error=_TIMED_OUT)
        except ToolError as exc:
            self.logger.info("agent.tool_rejected", tool=call.tool, reason=str(exc))
            return ToolResult(tool=call.tool, input=call.input, ok=False, error=str(exc))
        except Exception as exc:
            self.logger.exception("agent.tool_failed", tool=call.tool)
            return ToolResult(
                tool=call.tool, input=call.input, ok=False, error=f"{type(exc).__name__}: {exc}"
            )

        return ToolResult(tool=call.tool, input=call.input, output=str(output))
