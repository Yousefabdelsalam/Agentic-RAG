"""What memory holds.

Three kinds, because they answer different questions and expire differently:

- short-term: the last few turns, verbatim, so pronouns and follow-ups resolve;
- summary: everything older, compressed, so a long conversation still fits;
- user context: durable facts about the person, which outlive the conversation.

A snapshot is the read-side view of all three, assembled once per run.
"""

from __future__ import annotations

from app.memory.base import MemoryRecord
from app.models.base import Schema

USER = "user"
ASSISTANT = "assistant"


class UserFact(Schema):
    """One durable thing known about the user."""

    key: str
    value: str


class UserContext(Schema):
    """Durable facts about the user, keyed so later turns can overwrite earlier ones."""

    facts: tuple[UserFact, ...] = ()

    def merged_with(self, incoming: tuple[UserFact, ...], *, limit: int) -> UserContext:
        """Fold new facts in, letting a later value replace an earlier one.

        Keeping the most recent `limit` entries bounds growth without a
        background eviction pass; the oldest untouched facts fall off first.
        """
        combined: dict[str, str] = {fact.key: fact.value for fact in self.facts}
        for fact in incoming:
            if fact.key and fact.value:
                combined.pop(fact.key, None)
                combined[fact.key] = fact.value
        kept = list(combined.items())[-limit:]
        return UserContext(facts=tuple(UserFact(key=key, value=value) for key, value in kept))

    def describe(self) -> str:
        """Render for a prompt; empty when nothing is known."""
        return "\n".join(f"- {fact.key}: {fact.value}" for fact in self.facts)


class MemorySnapshot(Schema):
    """Everything memory contributes to one run."""

    summary: str = ""
    recent: tuple[MemoryRecord, ...] = ()
    user_context: UserContext = UserContext()
    turns_recorded: int = 0

    @property
    def is_empty(self) -> bool:
        return not (self.summary or self.recent or self.user_context.facts)

    def describe_transcript(self) -> str:
        """Render recent turns as a readable exchange."""
        return "\n".join(f"{record.role}: {record.content}" for record in self.recent)

    def describe(self) -> str:
        """Render the whole snapshot for a prompt.

        Returns a stated absence rather than an empty string: a model handed a
        blank section tends to invent one, whereas "no prior conversation" is
        unambiguous.
        """
        if self.is_empty:
            return "No prior conversation."
        sections = [
            ("Conversation summary", self.summary),
            ("Known about the user", self.user_context.describe()),
            ("Recent turns", self.describe_transcript()),
        ]
        return "\n\n".join(f"{title}:\n{body}" for title, body in sections if body)
