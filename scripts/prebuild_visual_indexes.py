#!/usr/bin/env python3
"""Prebuild full-page visual indexes from already prepared local papers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Transformers reads these flags during import. Set them before importing the
# retrieval services so cache-only prebuilds cannot start conversion checks.
os.environ.setdefault("SCHOLAR_NETWORK_MODE", "strict-local")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from backend.services.document_visual_retrieval_service import (  # noqa: E402
    DocumentVisualRetrievalService,
)
from backend.services.colqwen_visual_retrieval_service import (  # noqa: E402
    ColQwenVisualRetrievalService,
)
from evaluation.corpus.manifest import load_selection  # noqa: E402


PAPERS_DIR = ROOT / "backend" / "data" / "papers"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_complete_indexes(source_ids: list[str], backend: str) -> list[str]:
    """Reject missing, partial, stale, or checksum-invalid selected-paper indexes."""
    errors: list[str] = []
    manifest_name = (
        "colqwen_page_manifest.json"
        if backend == "colqwen2"
        else "visual_page_embeddings_manifest.json"
    )
    for source_id in source_ids:
        directory = PAPERS_DIR / source_id
        try:
            units = json.loads((directory / "visual_units.json").read_text())
            expected_rows = [
                {
                    "visual_id": row["visual_id"],
                    "page": row["page"],
                    "image_relpath": row["image_relpath"],
                    "image_sha256": row["image_sha256"],
                }
                for row in units
                if row.get("unit_type") == "page"
            ]
            manifest = json.loads((directory / manifest_name).read_text())
            actual_rows = [
                {
                    "visual_id": row["visual_id"],
                    "page": row["page"],
                    "image_relpath": row["image_relpath"],
                    "image_sha256": row["image_sha256"],
                }
                for row in manifest["rows"]
            ]
            if not expected_rows or actual_rows != expected_rows:
                raise ValueError("manifest rows differ from full-page visual units")
            if manifest.get("source_paper_id") != source_id:
                raise ValueError("manifest source identity differs")
            if backend == "clip":
                vector_path = directory / "visual_page_embeddings.npy"
                if manifest.get("vector_sha256") != _sha256_file(vector_path):
                    raise ValueError("vector checksum differs")
            else:
                for name, expected_sha in (manifest.get("checksums") or {}).items():
                    if expected_sha != _sha256_file(directory / name):
                        raise ValueError(f"checksum differs: {name}")
                if not manifest.get("checksums"):
                    raise ValueError("manifest lacks companion checksums")
        except Exception as exc:
            errors.append(f"{source_id}: {type(exc).__name__}: {exc}")
    return errors


def prepared_paper_ids() -> list[str]:
    if not PAPERS_DIR.is_dir():
        return []
    return sorted(
        path.name
        for path in PAPERS_DIR.iterdir()
        if path.is_dir() and (path / "visual_units.json").is_file()
    )


def _full_page_count(source_id: str) -> int:
    """Return page work size so resumable builds commit small sources first."""
    try:
        units = json.loads((PAPERS_DIR / source_id / "visual_units.json").read_text())
    except (OSError, ValueError):
        return sys.maxsize
    return sum(row.get("unit_type") == "page" for row in units)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("colqwen2", "clip"),
        default="colqwen2",
        help="page encoder to prebuild",
    )
    parser.add_argument(
        "--paper-id",
        action="append",
        default=[],
        help="prepared paper ID; repeat for multiple papers (default: all)",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="prebuild indexes for a frozen corpus selection",
    )
    parser.add_argument(
        "--split",
        choices=("development", "test", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.selection and args.paper_id:
        parser.error("choose --paper-id or --selection, not both")
    if args.split != "all" and not args.selection:
        parser.error("--split requires --selection")
    if args.selection:
        selection = load_selection(args.selection)
        if args.split == "development":
            source_ids = selection.development_paper_ids
        elif args.split == "test":
            source_ids = selection.test_paper_ids
        else:
            source_ids = selection.all_paper_ids
    else:
        source_ids = sorted(set(args.paper_id or prepared_paper_ids()))
    if not source_ids:
        raise SystemExit("No prepared paper visual_units.json files were found.")
    source_ids = sorted(source_ids, key=lambda source_id: (
        _full_page_count(source_id), source_id
    ))

    # Index construction is deliberately cache-only even when the wider app is
    # acquisition-enabled. Model acquisition must be an explicit setup action.
    indexed_pages = 0
    index_bytes = 0
    try:
        for position, source_id in enumerate(source_ids, start=1):
            if args.backend == "colqwen2":
                status = ColQwenVisualRetrievalService.prebuild_source(source_id)
            else:
                status = DocumentVisualRetrievalService.search(
                    "scientific document page",
                    [source_id],
                    top_k=1,
                    backend=args.backend,
                ).status
            progress = {
                "backend": status.backend,
                "completed": position,
                "index_bytes": status.index_bytes,
                "indexed_pages": status.indexed_pages,
                "paper_id": source_id,
                "succeeded": status.succeeded,
                "total": len(source_ids),
            }
            print(json.dumps(progress, sort_keys=True), flush=True)
            if not status.model_loaded or status.succeeded is False:
                print(json.dumps(status.as_dict(), indent=2, sort_keys=True))
                return 2
            if status.indexed_pages <= 0:
                return 3
            indexed_pages += status.indexed_pages
            index_bytes += status.index_bytes
    finally:
        DocumentVisualRetrievalService.release()
    coverage_errors = validate_complete_indexes(source_ids, args.backend)
    if coverage_errors:
        print("Index coverage validation failed:", file=sys.stderr)
        for error in coverage_errors:
            print(f"- {error}", file=sys.stderr)
        return 4
    print(json.dumps({
        "backend": args.backend,
        "complete": True,
        "index_bytes": index_bytes,
        "indexed_pages": indexed_pages,
        "paper_count": len(source_ids),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
