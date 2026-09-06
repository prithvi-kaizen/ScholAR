#!/usr/bin/env python3
"""Refresh extracted figures and tables for a given paper or all papers."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.services.pdf_service import extract_figures, write_json
from backend.services.storage_service import StorageService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def refresh_paper(paper_dir: Path) -> bool:
    pdf_path = paper_dir / "paper.pdf"
    if not pdf_path.exists():
        logger.warning("No paper.pdf in %s, skipping", paper_dir)
        return False

    figures_dir = paper_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures_json_path = paper_dir / "figures.json"

    logger.info("Extracting figures from %s...", pdf_path)
    records = extract_figures(pdf_path, figures_dir)
    write_json(figures_json_path, records)
    logger.info("Updated %s with %d figures/tables", figures_json_path, len(records))

    # Sync to document.db if present
    db_path = paper_dir / "document.db"
    if db_path.exists():
        try:
            from backend.services.pdf_service import read_json
            meta_path = paper_dir / "metadata.json"
            chunks_path = paper_dir / "chunks.json"
            meta = read_json(meta_path) if meta_path.exists() else {"paper_id": paper_dir.name}
            chunks = read_json(chunks_path) if chunks_path.exists() else []
            StorageService.sync_paper_to_db(
                paper_id=paper_dir.name,
                metadata=meta,
                chunks=chunks,
                figures=records,
                db_path=db_path,
            )
            logger.info("Synced updated figures to %s", db_path)
        except Exception as exc:
            logger.warning("Could not sync to document.db: %s", exc)

    return True


def main():
    parser = argparse.ArgumentParser(description="Refresh figures and tables for papers.")
    parser.add_argument("--paper-id", type=str, default="1706.03762", help="Specific paper ID or 'all'")
    parser.add_argument("--data-dir", type=str, default="backend/data/papers", help="Data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if args.paper_id == "all":
        count = 0
        for p in data_dir.iterdir():
            if p.is_dir() and not p.name.startswith("."):
                if refresh_paper(p):
                    count += 1
        logger.info("Refreshed %d papers", count)
    else:
        target = data_dir / args.paper_id
        if not target.exists():
            logger.error("Paper directory %s does not exist", target)
            sys.exit(1)
        refresh_paper(target)


if __name__ == "__main__":
    main()
