"""Recompute release-v1 scored rows from raw rows without generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.gates import validate_gate_registry  # noqa: E402
from evaluation.release.identity import validate_row_against_condition  # noqa: E402
from evaluation.release.io import load_cases, load_config, read_json, read_jsonl, resolve_repo_path  # noqa: E402
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import ExpectedKeySet, RawReleaseRow  # noqa: E402
from evaluation.release.scoring import score_release  # noqa: E402


def run(config_path: Path) -> Path:
    config = load_config(config_path)
    release_dir = resolve_repo_path(config.output_dir)
    expected = ExpectedKeySet.model_validate(read_json(release_dir / "expected_keys.json"))
    raw = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
    if len(raw) != expected.n_expected:
        raise ValueError("scoring requires one immutable raw row per expected key")
    manifest = load_manifest(release_dir / "manifest.json")
    cases = {case.case_id: case for case in load_cases(config)}
    for row in raw:
        identity_errors = validate_row_against_condition(row, config, cases[row.key.case_id], manifest)
        if identity_errors:
            raise ValueError(
                f"scoring rejected row identity for {row.key.as_string()}: "
                + "; ".join(identity_errors)
            )
    if config.dataset.evidence_class != "toy":
        _, gate_errors = validate_gate_registry(
            release_dir / "gates.json",
            required_phase="PRE_SCORING",
        )
        if gate_errors:
            raise ValueError("scoring blocked by evidence gates:\n- " + "\n- ".join(gate_errors))
    score_release(release_dir / "raw/rows.jsonl", release_dir / "scored/rows.jsonl", config.metrics)
    update_manifest(manifest, release_dir, raw_rows=raw, lifecycle_status="SCORED")
    return release_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"scored release complete: {run(args.config)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
