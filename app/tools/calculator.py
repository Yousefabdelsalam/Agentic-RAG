"""Calculator tool.

Arithmetic is evaluated by walking a parsed AST, not by `eval`. Tool input
originates from a language model acting on user text, so it is untrusted: `eval`
on that path would expose attribute access, imports, and calls. This walker
admits numbers, the six arithmetic operators, unary sign, and nothing else — an
unsupported node is rejected by construction rather than filtered by pattern.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from app.tools.common import TextTool, ToolError

_BINARY: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_MAX_EXPONENT = 64  # keeps 2**10**10 from hanging the event loop
_MAX_LENGTH = 200


class CalculatorTool(TextTool):
    """Evaluates a arithmetic expression and returns the result."""

    name = "calculator"
    description = "Evaluate an arithmetic expression exactly"
    argument_hint = "an arithmetic expression such as (1200 * 0.15) + 30"

    async def execute(self, text: str) -> str:
        expression = text.strip()
        if not expression:
            raise ToolError("No expression given")
        if len(expression) > _MAX_LENGTH:
            raise ToolError(f"Expression longer than {_MAX_LENGTH} characters")

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ToolError(f"Not a valid expression: {expression}") from exc

        value = _evaluate(tree.body)
        return _format(value)


def _evaluate(node: ast.expr) -> float:
    """Reduce an expression node to a number, rejecting anything else."""
    match node:
        case ast.Constant(value=bool()):
            raise ToolError("Booleans are not numbers")
        case ast.Constant(value=int() | float() as value):
            return float(value)
        case ast.UnaryOp(op=op, operand=operand) if type(op) in _UNARY:
            return float(_UNARY[type(op)](_evaluate(operand)))
        case ast.BinOp(op=op, left=left, right=right) if type(op) in _BINARY:
            return _apply(type(op), _evaluate(left), _evaluate(right))
        case _:
            raise ToolError(f"Unsupported syntax: {type(node).__name__}")


def _apply(op: type[ast.operator], left: float, right: float) -> float:
    if op is ast.Pow and abs(right) > _MAX_EXPONENT:
        raise ToolError(f"Exponent larger than {_MAX_EXPONENT}")
    try:
        result = float(_BINARY[op](left, right))
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except OverflowError as exc:
        raise ToolError("Result too large") from exc
    if math.isnan(result) or math.isinf(result):
        raise ToolError("Result is not a finite number")
    return result


def _format(value: float) -> str:
    """Render without a trailing `.0` for whole numbers."""
    return str(int(value)) if value.is_integer() else f"{value:g}"
