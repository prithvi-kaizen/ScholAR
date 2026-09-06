"""Artifact-only score -> aggregate -> table -> validate reproduction of the toy release."""

from __future__ import annotations

import shutil
import socket
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.aggregate import aggregate_release, render_tables  # noqa: E402
from evaluation.release.io import load_config, read_jsonl  # noqa: E402
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import RawReleaseRow, RowStatus  # noqa: E402
from evaluation.release.scoring import score_release  # noqa: E402
from evaluation.release.validate import validate_release_directory  # noqa: E402

SOURCE = ROOT / "evaluation/fixtures/releases/release_v1_minimal"
GOLDEN = (
    "scored/rows.jsonl",
    "aggregates/summary.json",
    "tables/summary.csv",
    "tables/summary.tex",
    "tables/provenance.json",
)


def _deny_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("artifact-only release reproduction attempted network access")


def main() -> int:
    committed_errors = validate_release_directory(SOURCE)
    if committed_errors:
        raise RuntimeError("committed fixture is invalid:\n- " + "\n- ".join(committed_errors))
    raw = read_jsonl(SOURCE / "raw/rows.jsonl", RawReleaseRow)
    keys = [row.key.as_tuple() for row in raw]
    if keys == sorted(keys):
        raise AssertionError("fixture raw rows must remain deliberately out of order")
    if {row.key.seed for row in raw} != {11, 29}:
        raise AssertionError("fixture no longer covers multiple seeds")
    if {row.status for row in raw} != {RowStatus.SUCCESS, RowStatus.ABSTAINED, RowStatus.ERROR}:
        raise AssertionError("fixture must cover SUCCESS, ABSTAINED, and ERROR")

    with tempfile.TemporaryDirectory(prefix="scholar-release-fixture-") as temporary:
        candidate = Path(temporary) / SOURCE.name
        shutil.copytree(SOURCE, candidate)
        config = load_config(candidate / "configs/release_config.json")
        with patch.object(socket.socket, "connect", _deny_network):
            score_release(candidate / "raw/rows.jsonl", candidate / "scored/rows.jsonl", config.metrics)
            aggregate_release(
                candidate / "scored/rows.jsonl",
                candidate / "expected_keys.json",
                candidate / "aggregates/summary.json",
                [metric.name for metric in config.metrics],
            )
            render_tables(candidate / "aggregates/summary.json", candidate / "tables")
        for name in GOLDEN:
            if (candidate / name).read_bytes() != (SOURCE / name).read_bytes():
                raise AssertionError(f"artifact-only reproduction is not byte-identical: {name}")
        manifest = load_manifest(candidate / "manifest.json")
        update_manifest(manifest, candidate, raw_rows=raw, lifecycle_status="VALIDATED", completed=True)
        errors = validate_release_directory(candidate)
        if errors:
            raise RuntimeError("reproduced fixture is invalid:\n- " + "\n- ".join(errors))
    print("release-v1 artifact-only reproduction OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
