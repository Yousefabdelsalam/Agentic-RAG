FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies resolve from the lockfile alone, so this layer is reused for
# every build that does not change pyproject.toml or uv.lock.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project --frozen

# README.md is not documentation here: pyproject declares it as the project
# readme, so the build backend reads it while installing the project.
COPY README.md ./
COPY app ./app
RUN uv sync --no-dev --frozen


FROM python:3.12-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1

WORKDIR /app

RUN useradd --create-home --uid 1000 appuser

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --chown=appuser:appuser app ./app

# Persisted Chroma lives here when running without a separate Chroma service.
# Created ahead of the USER switch so the volume mount is writable.
RUN mkdir -p /app/.chroma && chown appuser:appuser /app/.chroma

USER appuser
EXPOSE 8000

# Liveness only. Readiness is /api/v1/ready, which an orchestrator should poll
# separately: a container that is alive but not ready should stop receiving
# traffic, not be killed and restarted.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=4).status == 200 else 1)"

# One worker by design. Conversation memory is in-process, so a second worker
# would answer follow-up questions from a session it has never seen. Scale by
# replacing InMemoryStore with a shared implementation first.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
