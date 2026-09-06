#!/usr/bin/env python3
"""Score the predeclared independent-human support/coverage release gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.human_scoring import score_primary_gate  # noqa: E402
from evaluation.release.io import read_json, read_jsonl, write_json  # noqa: E402
from evaluation.release.schemas import (  # noqa: E402
    HumanEvaluationBundle,
    PairedComparisonSpec,
    RawReleaseRow,
)


def run(release_dir: Path, annotations: Path, spec_path: Path) -> Path:
    bundle = HumanEvaluationBundle.model_validate(read_json(annotations))
    spec = PairedComparisonSpec.model_validate(read_json(spec_path))
    rows = read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow)
    result = score_primary_gate(bundle, spec, rows)
    output = release_dir / "human/primary_gate.json"
    write_json(output, result)
    if result.decision == "BLOCKED":
        raise ValueError("primary human gate is BLOCKED: " + "; ".join(result.reasons))
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    try:
        output = run(args.release_dir, args.annotations, args.spec)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print(f"human support/coverage gate written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
