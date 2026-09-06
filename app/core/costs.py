"""Local cost estimation for model calls.

LangSmith computes cost server-side from the model name and token counts. This
module exists so the same number is available *in process* — in logs, in run
metadata, and to callers who never open LangSmith — and so a self-hosted or
disabled-tracing deployment is not left without cost data.

Prices are per million tokens in USD and are a snapshot, not a live feed, so
every figure produced here is an estimate. Override them through settings rather
than editing this table when they move.
"""

from __future__ import annotations

from app.models.base import Schema

USD_PER_MILLION = 1_000_000
_CHARACTERS_PER_TOKEN = 4  # rough English average, used only when no count is reported


class ModelPrice(Schema):
    """Input and output price for one model, in USD per million tokens."""

    input_usd: float = 0.0
    output_usd: float = 0.0


DEFAULT_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(input_usd=2.50, output_usd=10.00),
    "gpt-4o-mini": ModelPrice(input_usd=0.15, output_usd=0.60),
    "text-embedding-3-small": ModelPrice(input_usd=0.02),
    "text-embedding-3-large": ModelPrice(input_usd=0.13),
}


def estimate_cost(
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    prices: dict[str, ModelPrice] | None = None,
) -> float:
    """Return the estimated USD cost of a call, or 0.0 for an unpriced model.

    An unknown model yields zero rather than a guess: a missing cost is obvious
    in a dashboard, whereas a fabricated one is not.
    """
    table = prices if prices is not None else DEFAULT_PRICES
    price = table.get(model) or _match_prefix(model, table)
    if price is None:
        return 0.0
    return (input_tokens * price.input_usd + output_tokens * price.output_usd) / USD_PER_MILLION


def estimate_tokens(texts: list[str]) -> int:
    """Approximate token count for text whose usage the provider does not report.

    The embeddings endpoint returns no usage, so this keeps embedding spend
    visible. It is a character heuristic and is labelled `estimated` everywhere
    it surfaces.
    """
    return sum(len(text) for text in texts) // _CHARACTERS_PER_TOKEN


def _match_prefix(model: str, table: dict[str, ModelPrice]) -> ModelPrice | None:
    """Fall back to the longest configured prefix, so dated snapshots still price.

    OpenAI publishes pinned names like `gpt-4o-2024-11-20` at the base model's
    price, and pricing them as unknown would understate spend.
    """
    candidates = [name for name in table if model.startswith(name)]
    return table[max(candidates, key=len)] if candidates else None
