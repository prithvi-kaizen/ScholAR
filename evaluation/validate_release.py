"""Validate release-v1 schemas, provenance, accounting, checksums, and hygiene."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.io import read_jsonl  # noqa: E402
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import RawReleaseRow  # noqa: E402
from evaluation.release.validate import validate_release_directory  # noqa: E402


def run(release_dir: Path, mark_validated: bool = True) -> list[str]:
    errors = validate_release_directory(release_dir)
    if errors or not mark_validated:
        return errors
    raw = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
    manifest = load_manifest(release_dir / "manifest.json")
    update_manifest(manifest, release_dir, raw_rows=raw, lifecycle_status="VALIDATED", completed=True)
    return validate_release_directory(release_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    errors = run(args.release_dir, mark_validated=not args.check_only)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"release validation OK: {args.release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
