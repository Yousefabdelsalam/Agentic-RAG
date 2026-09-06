"""Command-line entrypoint: `python -m app.ingestion <file.pdf> [<file.pdf> ...]`."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import anyio

from app.config.settings import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.pipeline import DocumentIngestionPipeline
from app.ingestion.wiring import build_ingestion_container

logger = get_logger(__name__)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.ingestion",
        description="Ingest PDF documents into the configured vector store.",
    )
    parser.add_argument("sources", nargs="+", help="Paths to the PDF files to ingest")
    return parser.parse_args(argv)


async def ingest_sources(sources: Sequence[str]) -> int:
    """Ingest every source in turn and return the total chunks indexed."""
    settings = get_settings()
    configure_logging(settings.logging)

    container = build_ingestion_container(settings)
    await container.startup()
    try:
        pipeline = container.resolve(DocumentIngestionPipeline)
        return sum([await pipeline.ingest(source) for source in sources])
    finally:
        await container.shutdown()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    total = anyio.run(ingest_sources, args.sources)
    logger.info("ingestion.finished", sources=len(args.sources), chunks=total)


if __name__ == "__main__":
    main()
