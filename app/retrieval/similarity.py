"""Vector arithmetic used by the retrieval layer.

Pure functions over plain sequences: no store, no client, no I/O. Kept in stdlib
Python so the layer does not take on an array dependency for what amounts to a
few dot products over a candidate pool of tens of vectors.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector = Sequence[float]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Return cosine similarity in [-1, 1]; zero vectors score 0."""
    if len(left) != len(right):
        raise ValueError(f"Vector length mismatch: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def distance_to_score(distance: float) -> float:
    """Map a Chroma cosine distance onto a [0, 1] relevance score.

    Cosine distance runs 0..2 for the collection's configured space, so the
    complement is clamped rather than assumed to be positive.
    """
    return min(1.0, max(0.0, 1.0 - distance))


def maximal_marginal_relevance(
    query: Vector,
    candidates: Sequence[Vector],
    *,
    k: int,
    lambda_mult: float,
) -> list[int]:
    """Select up to `k` candidate indices balancing relevance and diversity.

    Returns positions into `candidates`, in selection order. `lambda_mult` is the
    weight on relevance: 1.0 reduces to plain similarity ranking, 0.0 selects
    purely for dissimilarity from what is already chosen.
    """
    if not candidates or k <= 0:
        return []

    relevance = [cosine_similarity(query, candidate) for candidate in candidates]
    selected = [max(range(len(candidates)), key=relevance.__getitem__)]
    remaining = {index for index in range(len(candidates))} - set(selected)

    while remaining and len(selected) < min(k, len(candidates)):
        best_index, best_value = None, -math.inf
        for index in remaining:
            redundancy = max(
                cosine_similarity(candidates[index], candidates[chosen]) for chosen in selected
            )
            value = lambda_mult * relevance[index] - (1.0 - lambda_mult) * redundancy
            if value > best_value:
                best_index, best_value = index, value
        if best_index is None:
            break
        selected.append(best_index)
        remaining.discard(best_index)

    return selected


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> dict[str, float]:
    """Fuse ranked id lists into a single score per id.

    RRF scores by rank rather than by raw score, which is what makes it safe to
    combine a dense retriever's cosine similarities with a sparse retriever's
    term-frequency scores: the two are never compared on the same scale.
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    factors = list(weights) if weights is not None else [1.0] * len(rankings)

    fused: dict[str, float] = {}
    for ranking, weight in zip(rankings, factors, strict=True):
        for rank, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return fused
