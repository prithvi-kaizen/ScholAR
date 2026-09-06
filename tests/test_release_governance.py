"""Regressions for evidence-backed gates, table seals, and style provenance."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from evaluation.prepare_paper_tables import validate_seal
from evaluation.release.gates import validate_gate_registry
from evaluation.release.io import write_json
from evaluation.release.schemas import GateRegistry
from evaluation.validate_submission_pdf import validate_style_provenance

ROOT = Path(__file__).resolve().parents[1]


class TestReleaseGovernance(unittest.TestCase):
    def test_cleared_gate_without_typed_hashed_evidence_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "typed hashed evidence"):
            GateRegistry.model_validate({
                "schema_version": "1.0",
                "release_id": "release",
                "status": "READY",
                "gates": [{
                    "id": "heldout",
                    "phase": "PRE_GENERATION",
                    "status": "CLEARED",
                    "evidence": [],
                }],
            })

    def test_fake_gate_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="gate-evidence-", dir=ROOT / "evaluation") as temporary:
            directory = Path(temporary)
            evidence = directory / "evidence.json"
            evidence.write_text("{}\n", encoding="utf-8")
            relative = evidence.relative_to(ROOT).as_posix()
            registry_path = directory / "gates.json"
            write_json(registry_path, {
                "schema_version": "1.0",
                "release_id": "release",
                "status": "READY",
                "gates": [{
                    "id": "heldout",
                    "phase": "PRE_GENERATION",
                    "status": "CLEARED",
                    "evidence": [{
                        "path": relative,
                        "sha256": "0" * 64,
                        "bytes": evidence.stat().st_size,
                        "schema_name": "heldout_data_card",
                        "schema_version": "1.0",
                        "evidence_class": "measured",
                        "claim_status": "current",
                        "allowed_use": "release_generation",
                    }],
                }],
            })
            _registry, errors = validate_gate_registry(registry_path, require_all_cleared=True)
        self.assertTrue(any("hash differs" in error for error in errors), errors)

    def test_forged_table_marker_and_style_bytes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paper = Path(temporary) / "paper"
            release = Path(temporary) / "release"
            (paper / "build").mkdir(parents=True)
            release.mkdir()
            (paper / "build/release_table_gate.tex").write_text(
                "\\releasetablestrue\n", encoding="utf-8"
            )
            seal_errors = validate_seal(paper, release)
            self.assertTrue(any("forged" in error for error in seal_errors), seal_errors)

            (paper / "style").mkdir()
            (paper / "style/acl.sty").write_text("modified", encoding="utf-8")
            (paper / "style/acl_natbib.bst").write_text("modified", encoding="utf-8")
            (paper / "style/UPSTREAM_COMMIT.txt").write_text("a" * 40 + "\n", encoding="utf-8")
            (paper / "style/official_style_manifest.json").write_text(json.dumps({
                "schema_version": "1.0",
                "artifact_type": "official_acl_style_manifest",
                "status": "INSTALLED",
                "repository": "https://github.com/acl-org/acl-style-files",
                "upstream_commit": "a" * 40,
                "modified": False,
                "files": [
                    {"path": "style/acl.sty", "sha256": "0" * 64, "bytes": 8},
                    {"path": "style/acl_natbib.bst", "sha256": "0" * 64, "bytes": 8},
                ],
            }), encoding="utf-8")
            style_errors = validate_style_provenance(paper)
            self.assertTrue(any("differs from pinned" in error for error in style_errors), style_errors)


if __name__ == "__main__":
    unittest.main()
