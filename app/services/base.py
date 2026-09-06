"""Base class for application services.

Services orchestrate domain components (retrieval, memory, agents) for a single
use case. They receive their collaborators through the constructor, never
importing the container or FastAPI.
"""

from __future__ import annotations

from app.core.base import Component
from app.core.logging import get_logger


class Service(Component):
    """Common lifecycle and logging for use-case orchestrators."""

    def __init__(self) -> None:
        self.logger = get_logger(type(self).__module__)
