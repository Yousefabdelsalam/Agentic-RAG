from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from app.agents.models import PlannedToolCall, RetrievalPlan
from app.agents.nodes.tools import ToolNode
from app.agents.state import RAGState
from app.config.settings import ToolSettings
from app.core.exceptions import NotFoundError
from app.tools.base import ToolRegistry
from app.tools.calculator import CalculatorTool
from app.tools.clock import CurrentTimeTool
from app.tools.common import INPUT_ARGUMENT, TextTool, ToolError, ToolResult
from app.tools.web_search import SearchHit, UnavailableWebSearch, WebSearchTool

# ------------------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("(1200 * 0.15) + 30", "210"),
        ("10 / 4", "2.5"),
        ("2 ** 10", "1024"),
        ("-5 + 3", "-2"),
        ("17 % 5", "2"),
        ("17 // 5", "3"),
    ],
)
async def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert await CalculatorTool().execute(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__",
        "[x for x in range(10)]",
        "lambda: 1",
        "print(1)",
        "x + 1",
    ],
)
async def test_calculator_rejects_everything_that_is_not_arithmetic(expression: str) -> None:
    with pytest.raises(ToolError):
        await CalculatorTool().execute(expression)


async def test_calculator_rejects_unbounded_exponents() -> None:
    with pytest.raises(ToolError, match="Exponent"):
        await CalculatorTool().execute("2 ** 10000000")


async def test_calculator_reports_division_by_zero() -> None:
    with pytest.raises(ToolError, match="Division by zero"):
        await CalculatorTool().execute("1 / 0")


async def test_calculator_rejects_empty_and_oversized_input() -> None:
    with pytest.raises(ToolError):
        await CalculatorTool().execute("   ")
    with pytest.raises(ToolError):
        await CalculatorTool().execute("1+" * 200 + "1")


async def test_calculator_rejects_booleans_as_numbers() -> None:
    with pytest.raises(ToolError):
        await CalculatorTool().execute("True + 1")


# ------------------------------------------------------------------------ current time


async def test_current_time_defaults_to_utc() -> None:
    fixed = datetime(2026, 8, 7, 12, 30, 0, tzinfo=UTC)

    result = await CurrentTimeTool(clock=lambda: fixed).execute("")

    assert result == "2026-08-07 12:30:00 UTC"


async def test_current_time_converts_to_a_named_zone() -> None:
    fixed = datetime(2026, 8, 7, 12, 30, 0, tzinfo=UTC)

    result = await CurrentTimeTool(clock=lambda: fixed).execute("Asia/Tokyo")

    assert result.startswith("2026-08-07 21:30:00")


async def test_current_time_rejects_an_unknown_zone() -> None:
    with pytest.raises(ToolError, match="Unknown timezone"):
        await CurrentTimeTool().execute("Mars/Olympus_Mons")


# -------------------------------------------------------------------------- web search


class FakeSearchBackend:
    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int = 5) -> list[SearchHit]:
        self.queries.append(query)
        return self._hits[:limit]


async def test_web_search_renders_ranked_hits() -> None:
    backend = FakeSearchBackend([SearchHit(title="A", url="https://a.example", snippet="first")])

    result = await WebSearchTool(backend).execute("what is x")

    assert "1. A (https://a.example): first" in result
    assert backend.queries == ["what is x"]


async def test_web_search_reports_no_results() -> None:
    result = await WebSearchTool(FakeSearchBackend([])).execute("obscure query")

    assert "No web results" in result


async def test_web_search_without_a_backend_fails_with_a_reason() -> None:
    with pytest.raises(ToolError, match="not configured"):
        await WebSearchTool(UnavailableWebSearch()).execute("anything")


async def test_web_search_rejects_an_empty_query() -> None:
    with pytest.raises(ToolError):
        await WebSearchTool(FakeSearchBackend([])).execute("  ")


# ---------------------------------------------------------------------------- registry


def test_registry_describes_its_tools_for_the_planner() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())

    described = [tool.describe() for tool in registry.all()]  # type: ignore[attr-defined]

    assert any("calculator" in line for line in described)
    assert any("current_time" in line for line in described)
    assert all(line.startswith("- ") for line in described)


def test_registry_raises_for_an_unknown_tool() -> None:
    with pytest.raises(NotFoundError):
        ToolRegistry().get("nonexistent")


async def test_tools_are_callable_through_the_protocol_entry_point() -> None:
    registry = ToolRegistry()
    registry.register(CalculatorTool())

    assert await registry.get("calculator").run(**{INPUT_ARGUMENT: "6 * 7"}) == "42"


# --------------------------------------------------------------------------- tool node


class SlowTool(TextTool):
    name = "slow"
    description = "never finishes"

    async def execute(self, text: str) -> str:
        import anyio

        await anyio.sleep(10)
        return "done"


class ExplodingTool(TextTool):
    name = "exploding"
    description = "raises an unexpected error"

    async def execute(self, text: str) -> str:
        raise RuntimeError("unexpected")


def _registry(*tools: Any) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


def _state(*calls: PlannedToolCall) -> RAGState:
    return RAGState(plan=RetrievalPlan(tools_needed=bool(calls), tool_calls=calls))


async def test_tool_node_runs_the_planned_calls() -> None:
    node = ToolNode(_registry(CalculatorTool()), ToolSettings())

    update = await node(_state(PlannedToolCall(tool="calculator", input="2+2")))

    results = update["tool_results"]
    assert [result.output for result in results] == ["4"]
    assert all(result.ok for result in results)
    assert update["trace"] == ["tools:1/1"]


async def test_tool_node_records_a_failure_instead_of_raising() -> None:
    node = ToolNode(_registry(CalculatorTool()), ToolSettings())

    update = await node(_state(PlannedToolCall(tool="calculator", input="1/0")))

    result = update["tool_results"][0]
    assert result.ok is False
    assert "Division by zero" in result.error


async def test_tool_node_handles_an_unknown_tool() -> None:
    node = ToolNode(_registry(), ToolSettings())

    update = await node(_state(PlannedToolCall(tool="teleporter", input="x")))

    assert update["tool_results"][0].error == "Unknown tool: teleporter"


async def test_tool_node_contains_an_unexpected_exception() -> None:
    node = ToolNode(_registry(ExplodingTool()), ToolSettings())

    update = await node(_state(PlannedToolCall(tool="exploding", input="x")))

    assert update["tool_results"][0].ok is False
    assert "RuntimeError" in update["tool_results"][0].error


async def test_tool_node_times_a_call_out() -> None:
    node = ToolNode(_registry(SlowTool()), ToolSettings(timeout_seconds=0.05))

    update = await node(_state(PlannedToolCall(tool="slow", input="x")))

    assert update["tool_results"][0].ok is False
    assert "did not finish in time" in update["tool_results"][0].error


async def test_tool_node_caps_the_number_of_calls() -> None:
    node = ToolNode(_registry(CalculatorTool()), ToolSettings(max_calls_per_turn=2))
    calls = tuple(PlannedToolCall(tool="calculator", input=f"{n}+1") for n in range(5))

    update = await node(_state(*calls))

    assert len(update["tool_results"]) == 2


async def test_tool_node_does_nothing_when_disabled() -> None:
    node = ToolNode(_registry(CalculatorTool()), ToolSettings(enabled=False))

    update = await node(_state(PlannedToolCall(tool="calculator", input="2+2")))

    assert update["tool_results"] == []
    assert update["trace"] == ["tools:skipped"]


async def test_tool_node_does_nothing_without_planned_calls() -> None:
    node = ToolNode(_registry(CalculatorTool()), ToolSettings())

    update = await node(RAGState(plan=RetrievalPlan()))

    assert update["tool_results"] == []


def test_tool_results_render_success_and_failure_distinctly() -> None:
    assert ToolResult(tool="calculator", input="2+2", output="4").describe() == (
        "calculator(2+2) = 4"
    )
    assert "failed: boom" in ToolResult(tool="x", input="y", ok=False, error="boom").describe()
