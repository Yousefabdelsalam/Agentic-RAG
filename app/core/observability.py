"""Tracing decorators and run annotation.

Every traced span in this application goes through `traced`. It wraps LangSmith's
`@traceable` and adds the two things that decorator does not do on its own:
structured latency logging that survives tracing being switched off, and errors
recorded on the run before they propagate.

Nothing here changes behaviour. A decorated function returns exactly what it
returned before, raises exactly what it raised before, and the annotation helpers
are no-ops when there is no active run — so tracing can be disabled, or the
langsmith client can be unreachable, without any caller noticing.

Run types follow LangSmith's vocabulary, because the UI renders them
differently: `chain` for graph nodes, `llm` for model calls, `retriever` for
anything returning documents.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine, Mapping, Sequence
from functools import wraps
from typing import Any, Literal, ParamSpec, TypeVar, cast

from langsmith import get_current_run_tree, traceable

from app.core.costs import ModelPrice, estimate_cost
from app.core.logging import get_logger

P = ParamSpec("P")
R = TypeVar("R")

RunType = Literal["chain", "llm", "retriever", "tool", "embedding", "prompt", "parser"]
AsyncCallable = Callable[P, Coroutine[Any, Any, R]]

CHAIN: RunType = "chain"
LLM: RunType = "llm"
RETRIEVER: RunType = "retriever"

logger = get_logger(__name__)

_TRUNCATION_SUFFIX = "... [truncated]"


def traced(
    name: str,
    *,
    run_type: RunType = CHAIN,
    tags: Sequence[str] | None = None,
    **metadata: Any,
) -> Callable[[AsyncCallable[P, R]], AsyncCallable[P, R]]:
    """Trace an async callable as a LangSmith run and log its latency.

    Inputs are not captured automatically. Bound methods would send `self` and
    the entire graph state to LangSmith on every call, which is both noisy and a
    way to leak more than intended; each call site annotates the few fields that
    are worth seeing instead.
    """

    def decorate(function: AsyncCallable[P, R]) -> AsyncCallable[P, R]:
        @wraps(function)
        async def instrumented(*args: P.args, **kwargs: P.kwargs) -> R:
            started = time.perf_counter()
            try:
                result = await function(*args, **kwargs)
            except Exception as exc:
                elapsed = _elapsed_ms(started)
                record(latency_ms=elapsed)
                record_error(exc)
                logger.warning(
                    "trace.failed", span=name, duration_ms=elapsed, error=type(exc).__name__
                )
                raise
            elapsed = _elapsed_ms(started)
            record(latency_ms=elapsed)
            logger.debug("trace.completed", span=name, duration_ms=elapsed)
            return result

        # `traceable` wraps the instrumented function, not the other way round:
        # the timing and error annotations have to execute *inside* the run's
        # context, or `record` finds no active run and silently drops them.
        #
        # The returned callable also accepts a `langsmith_extra` keyword; this
        # layer never passes one, so the narrower type is the honest description
        # of how it is actually called.
        return cast(
            AsyncCallable[P, R],
            traceable(
                run_type,
                name=name,
                tags=list(tags) if tags else None,
                metadata=dict(metadata) if metadata else None,
                process_inputs=_drop_inputs,
            )(instrumented),
        )

    return decorate


def record(**metadata: Any) -> None:
    """Attach metadata to the current run, if there is one."""
    tree = get_current_run_tree()
    if tree is None:
        return
    tree.add_metadata({key: value for key, value in metadata.items() if value is not None})


def record_outputs(**outputs: Any) -> None:
    """Attach output fields to the current run, if there is one."""
    tree = get_current_run_tree()
    if tree is None:
        return
    tree.add_outputs(outputs)


def record_error(error: BaseException) -> None:
    """Record an exception on the current run.

    LangSmith already marks a failed run from the raised exception; this adds the
    type and message as metadata so a failure is filterable, not just visible.
    """
    record(error_type=type(error).__name__, error_message=str(error)[:500])


def record_usage(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prices: dict[str, ModelPrice] | None = None,
    estimated: bool = False,
) -> float:
    """Record token counts and estimated cost on the current run.

    Returns the cost so callers can log it too. LangSmith computes its own cost
    server-side for known models; this is the in-process estimate and is tagged
    as such so the two are never mistaken for one measurement.
    """
    cost = estimate_cost(
        model, input_tokens=input_tokens, output_tokens=output_tokens, prices=prices
    )
    record(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        estimated_cost_usd=round(cost, 8),
        token_counts_estimated=estimated,
    )
    return cost


def usage_of(message: Any) -> tuple[int, int]:
    """Extract `(input_tokens, output_tokens)` from a LangChain message.

    Returns zeros when the provider reported nothing, which is normal for some
    endpoints and must not be mistaken for a free call.
    """
    usage = getattr(message, "usage_metadata", None)
    if not isinstance(usage, Mapping):
        return 0, 0
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def truncate(text: str, limit: int) -> str:
    """Shorten captured payloads, marking that something was removed."""
    if len(text) <= limit:
        return text
    return text[:limit] + _TRUNCATION_SUFFIX


def _drop_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Discard positional/keyword arguments before they reach LangSmith."""
    return {}


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
