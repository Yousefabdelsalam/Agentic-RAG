"""Current-time tool.

Exists because a language model cannot know the current time: without this, any
question about "today" is answered from training data or invented. The clock is
injected rather than read directly from `datetime.now`, which is what makes a
run reproducible in a test.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.tools.common import TextTool, ToolError

_FORMAT = "%Y-%m-%d %H:%M:%S %Z"


class CurrentTimeTool(TextTool):
    """Reports the current date and time, in UTC or a named timezone."""

    name = "current_time"
    description = "Get the current date and time"
    argument_hint = "an IANA timezone such as Europe/London, or empty for UTC"

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        super().__init__()
        self._clock = clock or (lambda: datetime.now(UTC))

    async def execute(self, text: str) -> str:
        now = self._clock()
        zone = text.strip()
        if not zone:
            return now.strftime(_FORMAT)
        try:
            return now.astimezone(ZoneInfo(zone)).strftime(_FORMAT)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"Unknown timezone: {zone}") from exc
