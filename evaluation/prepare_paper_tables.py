#!/usr/bin/env python3
"""Create the only TeX switch allowed to include claim-bearing release tables."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.gates import validate_gate_registry  # noqa: E402
from evaluation.release.io import read_json, sha256_file, write_json  # noqa: E402
from evaluation.release.schemas import (  # noqa: E402
    PrimaryGateResult,
    ReleaseManifest,
    ReleaseTableSeal,
)
from evaluation.release.validate import validate_release_directory  # noqa: E402


def build_seal(paper_dir: Path, release_dir: Path) -> ReleaseTableSeal:
    errors = validate_release_directory(release_dir)
    if errors:
        raise ValueError("release validation failed:\n- " + "\n- ".join(errors))
    manifest = ReleaseManifest.model_validate(read_json(release_dir / "manifest.json"))
    if manifest.evidence_class != "measured" or manifest.claim_status != "current":
        raise ValueError("paper tables require a current measured release")
    registry, gate_errors = validate_gate_registry(release_dir / "gates.json")
    if registry is None:
        raise ValueError("paper tables require a valid gate registry")
    gate_errors.extend(
        f"gate {gate.id} is {gate.status}"
        for gate in registry.gates
        if gate.phase != "POST_BUILD" and gate.status != "CLEARED"
    )
    if gate_errors:
        raise ValueError("paper tables blocked by evidence gates:\n- " + "\n- ".join(sorted(set(gate_errors))))
    primary_path = release_dir / "human/primary_gate.json"
    if not primary_path.is_file():
        raise ValueError("paper tables require the independent-human primary gate result")
    primary = PrimaryGateResult.model_validate(read_json(primary_path))
    if (
        primary.decision != "PASS"
        or primary.evidence_class != "measured"
        or primary.claim_status != "current"
    ):
        raise ValueError("paper tables require a current measured PASS primary gate")
    return ReleaseTableSeal(
        release_id=manifest.release_id,
        run_id=manifest.run_id,
        evidence_class="measured",
        claim_status="current",
        release_manifest_sha256=sha256_file(release_dir / "manifest.json"),
        release_checksums_sha256=sha256_file(release_dir / "checksums.sha256"),
        summary_table_sha256=sha256_file(release_dir / "tables/summary.tex"),
        gate_registry_sha256=sha256_file(release_dir / "gates.json"),
        primary_gate_sha256=sha256_file(primary_path),
        claim_map_sha256=sha256_file(paper_dir / "claim_map.json"),
    )


def validate_seal(paper_dir: Path, release_dir: Path) -> list[str]:
    build_dir = paper_dir / "build"
    seal_path = build_dir / "release_table_gate.json"
    tex_path = build_dir / "release_table_gate.tex"
    if not seal_path.exists() and not tex_path.exists():
        return []
    pending_tex = "% No validated current measured release seal exists.\n\\releasetablesfalse\n"
    if not seal_path.exists() and tex_path.is_file():
        return [] if tex_path.read_text(encoding="utf-8") == pending_tex else [
            "release table gate is incomplete or forged"
        ]
    if not seal_path.is_file() or not tex_path.is_file():
        return ["release table gate is incomplete or forged"]
    try:
        declared = ReleaseTableSeal.model_validate(read_json(seal_path))
        actual = build_seal(paper_dir, release_dir)
    except Exception as exc:
        return [f"release table gate is invalid: {exc}"]
    if declared != actual:
        return ["release table gate hashes are stale or forged"]
    expected_tex = (
        "% Generated only by evaluation/prepare_paper_tables.py.\n"
        f"% seal-sha256: {sha256_file(seal_path)}\n"
        "\\releasetablestrue\n"
    )
    if tex_path.read_text(encoding="utf-8") != expected_tex:
        return ["release table TeX gate differs from its checksum-bound seal"]
    return []


def prepare(paper_dir: Path, release_dir: Path, *, pending_ok: bool) -> bool:
    build_dir = paper_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    seal_path = build_dir / "release_table_gate.json"
    tex_path = build_dir / "release_table_gate.tex"
    try:
        seal = build_seal(paper_dir, release_dir)
    except Exception:
        if not pending_ok:
            raise
        seal_path.unlink(missing_ok=True)
        tex_path.write_text(
            "% No validated current measured release seal exists.\n\\releasetablesfalse\n",
            encoding="utf-8",
        )
        return False
    write_json(seal_path, seal)
    tex_path.write_text(
        "% Generated only by evaluation/prepare_paper_tables.py.\n"
        f"% seal-sha256: {sha256_file(seal_path)}\n"
        "\\releasetablestrue\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper/eacl_industry")
    parser.add_argument("--release-dir", type=Path, default=ROOT / "evaluation/releases/eacl_industry_v1")
    parser.add_argument("--pending-ok", action="store_true")
    args = parser.parse_args()
    try:
        included = prepare(args.paper_dir.resolve(), args.release_dir.resolve(), pending_ok=args.pending_ok)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    print("release tables enabled" if included else "release tables remain blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
