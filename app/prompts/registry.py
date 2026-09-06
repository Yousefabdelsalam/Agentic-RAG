"""Prompt template registry.

Templates live as files under `app/prompts/templates/` so they can be reviewed
and versioned independently of the code that renders them.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from string import Template

from app.core.exceptions import NotFoundError

TEMPLATE_DIR = Path(__file__).parent / "templates"
TEMPLATE_SUFFIX = ".md"


@lru_cache(maxsize=None)
def load(name: str) -> Template:
    """Load a prompt template by name, caching the parsed result."""
    path = TEMPLATE_DIR / f"{name}{TEMPLATE_SUFFIX}"
    if not path.is_file():
        raise NotFoundError(f"Unknown prompt template: {name}")
    return Template(path.read_text(encoding="utf-8"))


def render(name: str, **variables: str) -> str:
    """Render a template, failing loudly on missing variables."""
    return load(name).substitute(**variables)
