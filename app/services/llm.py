"""Chat service: the single way graph nodes talk to a language model.

Nodes never construct a client. They ask for either free text or a validated
schema instance, which keeps parsing and retry behaviour in one place and lets a
node be tested by substituting this service.
"""

from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAIError
from pydantic import ValidationError as PydanticValidationError

from app.config.settings import AgentSettings, ObservabilitySettings, OpenAISettings
from app.core.exceptions import ConfigurationError, DependencyError
from app.core.observability import (
    LLM,
    record,
    record_outputs,
    record_usage,
    traced,
    truncate,
    usage_of,
)
from app.models.base import Schema
from app.services.base import Service

SchemaT = TypeVar("SchemaT", bound=Schema)

_HEALTH_PROBE = "Reply with the single word: ok"


class ChatService(Service):
    """Async wrapper around the OpenAI chat endpoint.

    Both entry points are traced as `llm` runs carrying the prompt, the reply,
    token counts and estimated cost. The underlying `ChatOpenAI` call is also
    traced natively by LangChain and nests inside these spans, so the model's own
    parameters remain visible without this layer restating them.
    """

    def __init__(
        self,
        settings: OpenAISettings,
        agent: AgentSettings,
        observability: ObservabilitySettings | None = None,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._agent = agent
        self._observability = observability or ObservabilitySettings()
        self._client: ChatOpenAI | None = None

    @property
    def model(self) -> str:
        return self._settings.chat_model

    async def start(self) -> None:
        api_key = self._settings.api_key
        if api_key is None:
            raise ConfigurationError(
                "OPENAI__API_KEY is required to run the agent", details={"model": self.model}
            )
        self._client = ChatOpenAI(
            model=self.model,
            openai_api_key=api_key,
            openai_api_base=self._settings.base_url,
            request_timeout=self._agent.request_timeout_seconds,
            max_retries=self._settings.max_retries,
            temperature=self._agent.temperature,
        )
        self.logger.info("chat.ready", model=self.model)

    async def close(self) -> None:
        self._client = None

    @traced("chat.complete", run_type=LLM)
    async def complete(self, system: str, user: str) -> str:
        """Return the model's free-text reply."""
        client = self._require_client()
        self._record_prompt(system, user)
        try:
            message = await client.ainvoke(_messages(system, user))
        except OpenAIError as exc:
            raise DependencyError("Chat request failed", details={"model": self.model}) from exc

        answer = _text_of(message)
        self._record_usage(message)
        if self._observability.capture_answers:
            record_outputs(answer=truncate(answer, self._observability.max_captured_characters))
        return answer

    @traced("chat.structured", run_type=LLM)
    async def structured(self, schema: type[SchemaT], system: str, user: str) -> SchemaT:
        """Return a validated `schema` instance produced by the model.

        The raw message is requested alongside the parsed value purely so token
        usage is observable: `with_structured_output` otherwise returns only the
        parsed object, and every planning and critique call would be invisible in
        cost reporting.

        A model that answers with the wrong shape is an upstream failure, not a
        caller error, so a validation miss surfaces as a dependency error rather
        than a bare pydantic exception leaking into node code.
        """
        client = self._require_client()
        self._record_prompt(system, user, schema=schema.__name__)
        try:
            runnable = client.with_structured_output(schema, include_raw=True)
            envelope = await runnable.ainvoke(_messages(system, user))
        except OpenAIError as exc:
            raise DependencyError("Chat request failed", details={"model": self.model}) from exc
        except PydanticValidationError as exc:
            raise DependencyError(
                "Model returned a response the schema rejected",
                details={"model": self.model, "schema": schema.__name__},
            ) from exc

        parsed, error = _unwrap(envelope)
        self._record_usage(envelope.get("raw") if isinstance(envelope, dict) else None)
        if error is not None or parsed is None:
            raise DependencyError(
                "Model returned a response the schema rejected",
                details={"model": self.model, "schema": schema.__name__},
            )
        record_outputs(parsed=schema.__name__)
        return cast(SchemaT, parsed)

    def _record_prompt(self, system: str, user: str, **extra: str) -> None:
        if not self._observability.capture_prompts:
            record(model=self.model, **extra)
            return
        limit = self._observability.max_captured_characters
        record(
            model=self.model,
            prompt=truncate(system, limit),
            prompt_input=truncate(user, limit),
            prompt_characters=len(system) + len(user),
            **extra,
        )

    def _record_usage(self, message: Any) -> None:
        input_tokens, output_tokens = usage_of(message)
        cost = record_usage(
            self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prices=self._observability.model_prices,
        )
        self.logger.info(
            "chat.usage",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=round(cost, 8),
        )

    async def healthy(self) -> bool:
        """Report whether the service is configured and ready to be called.

        Deliberately cheap. A readiness probe is polled continuously, and a live
        model call per poll would bill real money to answer a question the local
        state already answers. `probe` exists for when an actual round trip is
        what is wanted.
        """
        return self._client is not None

    async def probe(self) -> bool:
        """Make a real round trip to the chat endpoint. Costs a token or two."""
        if self._client is None:
            return False
        try:
            await self.complete("You are a health probe.", _HEALTH_PROBE)
        except (DependencyError, ConfigurationError):
            return False
        return True

    def _require_client(self) -> ChatOpenAI:
        if self._client is None:
            raise ConfigurationError("Chat service used before start()")
        return self._client


def _unwrap(envelope: Any) -> tuple[Any, Any]:
    """Split `include_raw` output into `(parsed, parsing_error)`.

    The envelope shape is a LangChain detail; older paths return the parsed
    object directly, so a non-dict reply is treated as already-parsed.
    """
    if not isinstance(envelope, dict):
        return envelope, None
    return envelope.get("parsed"), envelope.get("parsing_error")


def _messages(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _text_of(message: BaseMessage) -> str:
    """Flatten a reply to text; content blocks arrive as a list for some models."""
    content: Any = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str | dict)
        ]
        return "".join(parts)
    return str(content)
