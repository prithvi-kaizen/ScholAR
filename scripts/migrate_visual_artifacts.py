#!/usr/bin/env python3
"""Atomically rebuild legacy local paper bundles with canonical visual pages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.paper_finalize_service import PaperFinalizeService
from backend.services.pdf_service import PAPERS_DIR, read_json, safe_paper_id
from evaluation.corpus.manifest import (
    DERIVED_INDEX_MANIFESTS,
    build_corpus_data_card,
    build_corpus_manifest,
    load_selection,
    validate_corpus_manifest,
    write_corpus_data_card,
    write_corpus_manifest,
)


def _candidates(
    requested: list[str],
    migrate_all: bool,
    *,
    selection_path: Path | None = None,
    split: str = "all",
) -> list[Path]:
    if selection_path is not None:
        selection = load_selection(selection_path)
        if split == "development":
            paper_ids = selection.development_paper_ids
        elif split == "test":
            paper_ids = selection.test_paper_ids
        else:
            paper_ids = selection.all_paper_ids
        return [PAPERS_DIR / paper_id for paper_id in paper_ids]
    if migrate_all:
        return sorted(path for path in PAPERS_DIR.iterdir() if path.is_dir())
    return [PAPERS_DIR / safe_paper_id(paper_id) for paper_id in requested]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Re-finalize local legacy papers so they contain checksummed full-page "
            "visual artifacts. The default is a read-only dry run."
        )
    )
    parser.add_argument("paper_ids", nargs="*", help="local paper IDs to migrate")
    parser.add_argument(
        "--all",
        action="store_true",
        help="inspect every local paper bundle",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        help="migrate every paper declared by a frozen corpus selection",
    )
    parser.add_argument(
        "--split",
        choices=("development", "test", "all"),
        default="all",
        help="selection split to migrate (default: all)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform transactional rebuilds instead of reporting them",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="after a successful full-selection apply, freeze the corpus manifest",
    )
    parser.add_argument(
        "--data-card-out",
        type=Path,
        help="write parser/degraded-mode disclosure beside the corpus manifest",
    )
    parser.add_argument(
        "--require-index-manifest",
        action="append",
        default=[],
        choices=DERIVED_INDEX_MANIFESTS,
    )
    args = parser.parse_args()
    source_count = (
        int(args.all) + int(bool(args.paper_ids)) + int(args.selection is not None)
    )
    if source_count != 1:
        parser.error("choose exactly one of paper IDs, --all, or --selection")
    if args.split != "all" and args.selection is None:
        parser.error("--split requires --selection")
    if args.manifest_out and (not args.apply or args.selection is None or args.split != "all"):
        parser.error("--manifest-out requires --apply with a complete --selection")
    if args.data_card_out and not args.manifest_out:
        parser.error("--data-card-out requires --manifest-out")

    migrated = 0
    skipped = 0
    failed = 0
    candidates = _candidates(
        args.paper_ids,
        args.all,
        selection_path=args.selection,
        split=args.split,
    )
    selected_mode = args.selection is not None
    for directory in candidates:
        paper_id = safe_paper_id(directory.name)
        pdf_path = directory / "paper.pdf"
        metadata_path = directory / "metadata.json"
        if not pdf_path.is_file() or not metadata_path.is_file():
            print(
                f"{'FAIL' if selected_mode else 'SKIP'} {paper_id}: "
                "missing paper.pdf or metadata.json"
            )
            if selected_mode:
                failed += 1
            else:
                skipped += 1
            continue
        if PaperFinalizeService.load_if_complete(paper_id, target_dir=directory):
            print(f"OK   {paper_id}: current visual artifact schema")
            skipped += 1
            continue
        if not args.apply:
            print(f"PLAN {paper_id}: transactional local rebuild required")
            continue
        try:
            metadata = read_json(metadata_path)
            if not isinstance(metadata, dict):
                raise ValueError("metadata.json is not an object")
            result = PaperFinalizeService.finalize(
                pdf_path,
                paper_id,
                metadata,
                target_dir=directory,
            )
            print(
                f"DONE {paper_id}: {result['pages']} pages, "
                f"{result['visual_units']} visual units"
            )
            migrated += 1
        except Exception as exc:
            print(f"FAIL {paper_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed += 1

    if args.manifest_out and failed == 0:
        selection = load_selection(args.selection)
        manifest = build_corpus_manifest(
            selection,
            required_index_manifests=tuple(args.require_index_manifest),
        )
        write_corpus_manifest(args.manifest_out, manifest)
        validation_errors = validate_corpus_manifest(manifest, selection)
        if validation_errors:
            for error in validation_errors:
                print(f"FAIL corpus manifest: {error}", file=sys.stderr)
            failed += len(validation_errors)
        else:
            print(
                f"FROZEN {args.manifest_out}: {manifest.paper_count} papers, "
                f"sha256={manifest.corpus_sha256}"
            )
            if args.data_card_out:
                write_corpus_data_card(
                    args.data_card_out,
                    build_corpus_data_card(manifest),
                )
                print(f"CARD {args.data_card_out}")

    mode = "applied" if args.apply else "dry-run"
    print(
        f"Summary ({mode}): selected={len(candidates)}, migrated={migrated}, "
        f"skipped={skipped}, failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
