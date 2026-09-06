"""Typed application settings loaded from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.costs import DEFAULT_PRICES, ModelPrice

Environment = Literal["local", "dev", "staging", "prod"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class ServerSettings(BaseModel):
    """HTTP server binding and exposure."""

    host: str = "0.0.0.0"
    port: int = 8000
    root_path: str = ""
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class LoggingSettings(BaseModel):
    """Log emission behaviour."""

    level: LogLevel = "INFO"
    json_format: bool = True


class OpenAISettings(BaseModel):
    """Credentials and model selection for the OpenAI provider."""

    api_key: SecretStr | None = None
    base_url: str | None = None
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    timeout_seconds: float = 60.0
    max_retries: int = 3


class ChromaSettings(BaseModel):
    """Chroma vector store connection."""

    persist_directory: str = "./.chroma"
    collection: str = "documents"
    host: str | None = None
    port: int | None = None


class IngestionSettings(BaseModel):
    """Chunking and batching behaviour of the ingestion pipeline."""

    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    embed_batch_size: int = Field(default=128, gt=0)
    upsert_batch_size: int = Field(default=256, gt=0)
    max_concurrent_batches: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def _overlap_fits_in_chunk(self) -> IngestionSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


class RetrievalSettings(BaseModel):
    """Defaults for the retrieval layer; every value is overridable per query."""

    default_search_type: Literal["similarity", "mmr", "hybrid"] = "similarity"
    top_k: int = Field(default=5, gt=0)
    fetch_k: int = Field(default=20, gt=0)
    mmr_lambda: float = Field(default=0.5, ge=0.0, le=1.0)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    compression_enabled: bool = True
    compression_fetch_multiplier: int = Field(default=3, gt=0)
    compression_min_relevance: float = Field(default=0.15, ge=0.0, le=1.0)
    compression_redundancy_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    compression_max_characters: int = Field(default=8000, gt=0)
    compression_sentence_level: bool = True

    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, gt=0)

    @model_validator(mode="after")
    def _fetch_k_covers_top_k(self) -> RetrievalSettings:
        if self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be at least top_k")
        return self


class AgentSettings(BaseModel):
    """Behaviour of the agentic RAG graph."""

    max_revisions: int = Field(default=2, ge=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_context_documents: int = Field(default=8, gt=0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    request_timeout_seconds: float = Field(default=90.0, gt=0.0)


class LangSmithSettings(BaseModel):
    """LangSmith tracing configuration."""

    enabled: bool = False
    api_key: SecretStr | None = None
    project: str = "agentic-rag"
    endpoint: str = "https://api.smith.langchain.com"


class MemorySettings(BaseModel):
    """Conversation memory sizing.

    Short-term memory is a window, not a log: once a conversation outgrows
    `short_term_turns`, older turns survive only through the rolling summary.
    """

    enabled: bool = True
    short_term_turns: int = Field(default=6, gt=0)
    summary_after_turns: int = Field(default=6, gt=0)
    max_user_facts: int = Field(default=20, gt=0)
    max_sessions: int = Field(default=1000, gt=0)
    extract_user_context: bool = True

    @model_validator(mode="after")
    def _summary_keeps_up_with_the_window(self) -> MemorySettings:
        # Turns leave the short-term window once it is full. If the summary were
        # written less often than that, those turns would be dropped before
        # anything had summarised them, and the conversation would lose history
        # silently rather than compress it.
        if self.summary_after_turns > self.short_term_turns:
            raise ValueError("summary_after_turns must not exceed short_term_turns")
        return self


class ToolSettings(BaseModel):
    """Tool execution limits."""

    enabled: bool = True
    max_calls_per_turn: int = Field(default=3, gt=0)
    timeout_seconds: float = Field(default=15.0, gt=0.0)


class ResilienceSettings(BaseModel):
    """Retry, timeout, and cache behaviour for calls that leave the process."""

    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_initial_backoff_seconds: float = Field(default=0.2, gt=0.0)
    retry_max_backoff_seconds: float = Field(default=5.0, gt=0.0)
    retry_jitter: float = Field(default=0.2, ge=0.0, le=1.0)

    request_timeout_seconds: float = Field(default=120.0, gt=0.0)

    embedding_cache_size: int = Field(default=512, ge=0)
    embedding_cache_ttl_seconds: float = Field(default=900.0, gt=0.0)

    @model_validator(mode="after")
    def _backoff_range_is_ordered(self) -> ResilienceSettings:
        if self.retry_max_backoff_seconds < self.retry_initial_backoff_seconds:
            raise ValueError("retry_max_backoff_seconds must be at least the initial backoff")
        return self


class CacheSettings(BaseSettings):
    """Answer cache behaviour.

    Read from `RAG_CACHE_*` environment variables — `RAG_CACHE_ENABLED`,
    `RAG_CACHE_TTL_SECONDS`, and so on — rather than the nested `CACHE__` form
    the other groups use, because these names are part of the deployment
    contract. The nested form still works for anyone who prefers consistency.

    `model_version` and `prompt_version` are derived from the running
    configuration when left unset, which is what makes an edited prompt or a
    changed model invalidate the cache without anyone remembering to bump a
    number. Set them to pin a value explicitly.
    """

    model_config = SettingsConfigDict(
        env_prefix="RAG_CACHE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # `model_version` collides with pydantic's reserved `model_` namespace;
        # the name is part of the cache's stored contract, so the guard goes.
        protected_namespaces=(),
    )

    enabled: bool = True
    ttl_seconds: float = Field(default=3600.0, gt=0.0)

    semantic_enabled: bool = True
    similarity_threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    #: Entries held in the semantic index. Lookup scans it linearly, so this
    #: bounds the cost of a miss as much as it bounds memory.
    semantic_max_entries: int = Field(default=500, gt=0)

    #: Set to use Redis. Unset keeps everything in-process, which is the right
    #: default for a single node and wrong for more than one.
    redis_url: str | None = None
    redis_timeout_seconds: float = Field(default=2.0, gt=0.0)

    namespace: str = "agentic-rag"
    model_version: str | None = None
    prompt_version: str | None = None


class ObservabilitySettings(BaseModel):
    """What traced runs are allowed to record.

    Payload capture is bounded because a traced run carries document text and
    rendered prompts; these limits are the difference between a readable trace
    and one nobody opens twice.
    """

    capture_documents: bool = True
    capture_prompts: bool = True
    capture_answers: bool = True
    max_captured_characters: int = Field(default=2000, gt=0)
    max_captured_documents: int = Field(default=10, gt=0)
    model_prices: dict[str, ModelPrice] = Field(default_factory=lambda: dict(DEFAULT_PRICES))


class Settings(BaseSettings):
    """Root settings object; the single source of truth for configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "agentic-rag"
    environment: Environment = "local"
    debug: bool = False
    api_prefix: str = "/api/v1"

    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    tools: ToolSettings = Field(default_factory=ToolSettings)
    resilience: ResilienceSettings = Field(default_factory=ResilienceSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    langsmith: LangSmithSettings = Field(default_factory=LangSmithSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    @property
    def is_production(self) -> bool:
        return self.environment == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
