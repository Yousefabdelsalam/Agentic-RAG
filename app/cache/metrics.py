"""Cache counters.

Plain integers rather than a metrics client: this application has no metrics
exporter, and inventing one to hold six numbers would be the larger change. The
snapshot is shaped so a Prometheus or OpenTelemetry exporter reads it without
translation when one arrives.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.cache.base import EXACT, SEMANTIC, CacheType

#: Model calls one full graph run makes at minimum: query analysis, planning,
#: generation, criticism, and the memory write. A run the critic sends back
#: costs more, so treating every hit as saving this many is the conservative
#: reading — the estimate understates rather than flatters.
LLM_CALLS_PER_RUN = 5


@dataclass
class CacheMetrics:
    """Running totals for one process."""

    cache_hits_total: int = 0
    cache_misses_total: int = 0
    exact_cache_hits: int = 0
    semantic_cache_hits: int = 0
    entries_written: int = 0
    #: Lookups the backend could not answer because it was unreachable. Counted
    #: as misses too, but tracked apart: a rising number here is an
    #: infrastructure problem, while a rising miss count is a workload.
    backend_errors_total: int = 0

    def record_hit(self, cache_type: CacheType) -> None:
        self.cache_hits_total += 1
        if cache_type == EXACT:
            self.exact_cache_hits += 1
        elif cache_type == SEMANTIC:
            self.semantic_cache_hits += 1

    def record_miss(self) -> None:
        self.cache_misses_total += 1

    def record_write(self) -> None:
        self.entries_written += 1

    def record_backend_error(self) -> None:
        self.backend_errors_total += 1

    @property
    def lookups_total(self) -> int:
        return self.cache_hits_total + self.cache_misses_total

    @property
    def cache_hit_rate(self) -> float:
        """Hits as a fraction of lookups; zero lookups is zero, not undefined."""
        return self.cache_hits_total / self.lookups_total if self.lookups_total else 0.0

    @property
    def estimated_llm_calls_saved(self) -> int:
        """Model calls not made because an answer was already known."""
        return self.cache_hits_total * LLM_CALLS_PER_RUN

    def snapshot(self) -> dict[str, float]:
        """Return every counter as a flat mapping."""
        return {
            "cache_hits_total": self.cache_hits_total,
            "cache_misses_total": self.cache_misses_total,
            "exact_cache_hits": self.exact_cache_hits,
            "semantic_cache_hits": self.semantic_cache_hits,
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "estimated_llm_calls_saved": self.estimated_llm_calls_saved,
            "entries_written": self.entries_written,
            "backend_errors_total": self.backend_errors_total,
        }
