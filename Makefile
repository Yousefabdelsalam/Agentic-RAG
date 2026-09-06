# Development shortcuts. Everything here is a plain command you can also run
# directly; nothing depends on make.

.DEFAULT_GOAL := help
.PHONY: help install dev test unit integration lint typecheck check format up down logs ingest

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Sync dependencies, including dev
	uv sync

dev: ## Run the API with reload
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

test: ## Run every test
	uv run pytest

unit: ## Run the fast suite only
	uv run pytest -m "not integration"

integration: ## Run the real-stack suite only
	uv run pytest -m integration

lint: ## Check formatting and lint rules
	uv run ruff check app tests
	uv run ruff format --check app tests

typecheck: ## Run mypy in strict mode
	uv run mypy app

check: lint typecheck test ## Everything CI runs

format: ## Apply formatting and safe fixes
	uv run ruff format app tests
	uv run ruff check --fix app tests

up: ## Start the stack in Docker
	docker compose up --build -d

down: ## Stop the stack, keeping volumes
	docker compose down

logs: ## Follow API logs
	docker compose logs -f api

ingest: ## Ingest a PDF offline: make ingest FILE=handbook.pdf
	uv run python -m app.ingestion $(FILE)
