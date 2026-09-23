"""Create deterministic case-balanced aggregates and gated release tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.aggregate import aggregate_release, render_tables  # noqa: E402
from evaluation.release.gates import validate_gate_registry  # noqa: E402
from evaluation.release.identity import validate_row_against_condition  # noqa: E402
from evaluation.release.io import (  # noqa: E402
    key_index,
    load_cases,
    load_config,
    load_corpus_manifest,
    read_json,
    read_jsonl,
    resolve_repo_path,
)
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import ExpectedKeySet, RawReleaseRow, ScoredReleaseRow  # noqa: E402


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    release_dir = resolve_repo_path(config.output_dir)
    cases = {case.case_id: case for case in load_cases(config)}
    manifest = load_manifest(release_dir / "manifest.json")
    if config.schema_version == "2.0":
        if manifest.lifecycle_status != "SCORED":
            raise ValueError("schema-v2 table inputs require a SCORED release manifest")
        _, gate_errors = validate_gate_registry(
            release_dir / "gates.json", required_phase="PRE_PAPER"
        )
        if gate_errors:
            raise ValueError("schema-v2 tables blocked by PRE_PAPER gates:\n- " + "\n- ".join(gate_errors))
        expected = ExpectedKeySet.model_validate(read_json(release_dir / "expected_keys.json"))
        raw_rows = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
        scored_rows = read_jsonl(release_dir / "scored/rows.jsonl", ScoredReleaseRow)
        expected_keys = {item.as_tuple() for item in expected.keys}
        if set(key_index(raw_rows)) != expected_keys or set(key_index(scored_rows)) != expected_keys:
            raise ValueError("schema-v2 table inputs do not cover every frozen expected key")
        corpus_manifest = load_corpus_manifest(config)
        identity_errors = [
            f"{row.key.as_string()}: {error}"
            for row in raw_rows
            for error in validate_row_against_condition(
                row, config, cases[row.key.case_id], manifest, corpus_manifest
            )
        ]
        if identity_errors:
            raise ValueError("schema-v2 tables rejected mismatched raw rows:\n- " + "\n- ".join(identity_errors))
    statistics = None
    if config.protocol_path:
        protocol = read_json(resolve_repo_path(config.protocol_path))
        statistics = protocol.get("statistics") if isinstance(protocol, dict) else None
    aggregate_release(
        release_dir / "scored/rows.jsonl",
        release_dir / "expected_keys.json",
        release_dir / "aggregates/summary.json",
        [metric.name for metric in config.metrics],
        cases=cases,
        statistics=statistics,
    )
    render_tables(
        release_dir / "aggregates/summary.json",
        release_dir / "tables",
        release_dir=release_dir if config.schema_version == "2.0" else None,
    )
    raw = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
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
