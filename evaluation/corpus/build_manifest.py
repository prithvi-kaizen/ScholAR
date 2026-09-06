#!/usr/bin/env python3
"""Build or validate a deterministic experimental corpus manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.pdf_service import PAPERS_DIR  # noqa: E402
from evaluation.corpus.manifest import (  # noqa: E402
    CorpusDataCard,
    CorpusManifest,
    DERIVED_INDEX_MANIFESTS,
    build_corpus_data_card,
    build_corpus_manifest,
    load_selection,
    validate_corpus_manifest,
    write_corpus_data_card,
    write_corpus_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--papers-dir", type=Path, default=PAPERS_DIR)
    parser.add_argument(
        "--data-card",
        type=Path,
        help="also write a parser/degraded-mode corpus data card",
    )
    parser.add_argument(
        "--require-index-manifest",
        action="append",
        default=[],
        choices=DERIVED_INDEX_MANIFESTS,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate an existing output without rewriting it",
    )
    args = parser.parse_args()
    selection = load_selection(args.selection)

    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"Corpus manifest does not exist: {args.output}")
        manifest = CorpusManifest.model_validate(
            json.loads(args.output.read_text(encoding="utf-8"))
        )
        errors = validate_corpus_manifest(
            manifest,
            selection,
            papers_dir=args.papers_dir,
        )
        if args.data_card:
            if not args.data_card.is_file():
                errors.append(f"corpus data card does not exist: {args.data_card}")
            else:
                try:
                    actual_card = CorpusDataCard.model_validate(
                        json.loads(args.data_card.read_text(encoding="utf-8"))
                    )
                    expected_card = build_corpus_data_card(manifest)
                    if actual_card != expected_card:
                        errors.append("corpus data card differs from the frozen manifest")
                except Exception as exc:
                    errors.append(
                        f"corpus data card validation failed: {type(exc).__name__}: {exc}"
                    )
        if errors:
            print("Corpus manifest validation failed:", file=sys.stderr)
            for error in errors:
                print(f"- {error}", file=sys.stderr)
            return 1
        print(
            f"Corpus manifest OK: {manifest.paper_count} papers, "
            f"sha256={manifest.corpus_sha256}"
        )
        return 0

    manifest = build_corpus_manifest(
        selection,
        papers_dir=args.papers_dir,
        required_index_manifests=tuple(args.require_index_manifest),
    )
    write_corpus_manifest(args.output, manifest)
    if args.data_card:
        write_corpus_data_card(args.data_card, build_corpus_data_card(manifest))
    print(
        f"Wrote {args.output}: {manifest.paper_count} papers, "
        f"{manifest.total_pages} pages, sha256={manifest.corpus_sha256}"
    )
    if args.data_card:
        print(f"Wrote {args.data_card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
