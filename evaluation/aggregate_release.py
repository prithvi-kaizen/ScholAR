"""Create deterministic case-balanced aggregates and tables for release-v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.aggregate import aggregate_release, render_tables  # noqa: E402
from evaluation.release.io import load_config, read_jsonl, resolve_repo_path  # noqa: E402
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import RawReleaseRow  # noqa: E402


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    release_dir = resolve_repo_path(config.output_dir)
    aggregate_release(
        release_dir / "scored/rows.jsonl",
        release_dir / "expected_keys.json",
        release_dir / "aggregates/summary.json",
        [metric.name for metric in config.metrics],
    )
    render_tables(release_dir / "aggregates/summary.json", release_dir / "tables")
    raw = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
    manifest = load_manifest(release_dir / "manifest.json")
    update_manifest(manifest, release_dir, raw_rows=raw, lifecycle_status="AGGREGATED", completed=True)
    return release_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"aggregate release complete: {run(args.config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
