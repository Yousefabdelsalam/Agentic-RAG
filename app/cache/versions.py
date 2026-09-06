"""What a cached answer is valid under, and how each part is derived.

Three things invalidate an answer, and each is versioned separately so that
changing one does not needlessly discard entries that the others still cover:

- the **knowledge base**, incremented whenever ingestion writes to the index;
- the **model**, derived from the chat model and the temperature it runs at;
- the **prompts**, derived from the contents of the template files themselves.

Deriving the last two rather than declaring them is the point. A prompt edited
in a hurry, or a model swapped in an environment variable, invalidates the cache
without anyone remembering to bump a number — which is the failure this design
exists to prevent.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import anyio

from app.cache.base import CacheBackend, CacheVersions
from app.cache.keys import knowledge_base_key
from app.config.settings import AgentSettings, CacheSettings, OpenAISettings
from app.core.base import Component
from app.core.logging import get_logger
from app.prompts.registry import TEMPLATE_DIR, TEMPLATE_SUFFIX

INITIAL_KNOWLEDGE_BASE_VERSION = "0"

logger = get_logger(__name__)


def prompt_version(directory: Path | None = None) -> str:
    """Digest every prompt template into one version token.

    Names are hashed alongside contents, so adding or renaming a template moves
    the version even when the text of the others is untouched.
    """
    root = directory or TEMPLATE_DIR
    digest = hashlib.sha256()
    if root.is_dir():
        for path in sorted(root.glob(f"*{TEMPLATE_SUFFIX}")):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(path.read_bytes())
            digest.update(b"\x00")
    return digest.hexdigest()[:12]


def model_version(openai: OpenAISettings, agent: AgentSettings) -> str:
    """Identify the model configuration that writes answers.

    Temperature is part of it: the same model at a different temperature is a
    different answer generator, and serving one's output as the other's would be
    a silent configuration change.
    """
    return f"{openai.chat_model}@t{agent.temperature:g}"


class KnowledgeBaseVersion(Component):
    """The version of the indexed corpus, shared through the cache backend.

    Kept in the backend rather than in the process so that a replica which did
    not perform the ingestion still stops serving answers from the corpus that
    preceded it. With the in-memory backend the value is process-local, which is
    the same trade the in-memory backend makes everywhere else.

    The counter is stored without a TTL: it must outlive the entries it
    invalidates, or an expiry would silently make stale answers valid again.
    """

    def __init__(self, backend: CacheBackend, settings: CacheSettings) -> None:
        self.logger = get_logger(__name__)
        self._backend = backend
        self._key = knowledge_base_key(settings.namespace)
        self._lock = anyio.Lock()
        #: Last value seen, used when the backend cannot be read. Serving a
        #: remembered version beats serving none, which would make every entry
        #: unmatchable and disable the cache outright.
        self._last_known = INITIAL_KNOWLEDGE_BASE_VERSION

    async def current(self) -> str:
        """Return the version of the corpus currently indexed."""
        stored = await self._backend.get(self._key)
        if stored is None:
            return self._last_known
        self._last_known = stored
        return stored

    async def bump(self, *, reason: str = "") -> str:
        """Advance the version, invalidating every answer written under the old one.

        Read-modify-write under a lock. Two concurrent ingestions could still
        interleave across replicas and land on the same number; that is
        harmless, because what matters is that the value differs from the one
        the existing entries carry, not that it counts every write.
        """
        async with self._lock:
            current = await self.current()
            nxt = str(_as_int(current) + 1)
            await self._backend.set(self._key, nxt, ttl_seconds=0)
            self._last_known = nxt
        self.logger.info("cache.knowledge_base_bumped", version=nxt, reason=reason or "ingestion")
        return nxt


class VersionResolver:
    """Assembles the version triple in force for the current request."""

    def __init__(
        self,
        knowledge_base: KnowledgeBaseVersion,
        *,
        model: str,
        prompts: str,
    ) -> None:
        self._knowledge_base = knowledge_base
        self._model = model
        self._prompts = prompts

    @property
    def model(self) -> str:
        return self._model

    @property
    def prompts(self) -> str:
        return self._prompts

    async def resolve(self) -> CacheVersions:
        """Read the live knowledge base version and pair it with the static two."""
        return CacheVersions(
            knowledge_base_version=await self._knowledge_base.current(),
            model_version=self._model,
            prompt_version=self._prompts,
        )


def build_resolver(
    knowledge_base: KnowledgeBaseVersion,
    settings: CacheSettings,
    openai: OpenAISettings,
    agent: AgentSettings,
) -> VersionResolver:
    """Create the resolver, honouring any pinned versions from configuration."""
    return VersionResolver(
        knowledge_base,
        model=settings.model_version or model_version(openai, agent),
        prompts=settings.prompt_version or prompt_version(),
    )


def _as_int(value: str) -> int:
    """Read a stored counter, tolerating a value written by something else."""
    try:
        return int(value)
    except ValueError:
        return 0
