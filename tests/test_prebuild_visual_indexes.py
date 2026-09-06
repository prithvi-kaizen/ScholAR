"""Fail-closed coverage checks for selected-corpus visual index prebuilds."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.prebuild_visual_indexes import _sha256_file, validate_complete_indexes


class TestVisualIndexCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for source_id in ("paper_a", "paper_b"):
            directory = self.root / source_id
            directory.mkdir()
            row = {
                "visual_id": "page_0001",
                "document_id": source_id,
                "source_paper_id": source_id,
                "page": 1,
                "unit_type": "page",
                "image_relpath": "page_images/page_0001.png",
                "image_sha256": "a" * 64,
            }
            (directory / "visual_units.json").write_text(json.dumps([row]))
            vector_path = directory / "visual_page_embeddings.npy"
            np.save(vector_path, np.ones((1, 2, 3), dtype=np.float16))
            manifest_row = {
                key: row[key]
                for key in ("visual_id", "page", "image_relpath", "image_sha256")
            }
            (directory / "visual_page_embeddings_manifest.json").write_text(
                json.dumps(
                    {
                        "source_paper_id": source_id,
                        "rows": [manifest_row],
                        "vector_sha256": _sha256_file(vector_path),
                    }
                )
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_clip_coverage_passes(self) -> None:
        with patch("scripts.prebuild_visual_indexes.PAPERS_DIR", self.root):
            errors = validate_complete_indexes(["paper_a", "paper_b"], "clip")
        self.assertEqual(errors, [])

    def test_missing_or_tampered_index_fails(self) -> None:
        (self.root / "paper_b" / "visual_page_embeddings_manifest.json").unlink()
        (self.root / "paper_a" / "visual_page_embeddings.npy").write_bytes(b"tampered")
        with patch("scripts.prebuild_visual_indexes.PAPERS_DIR", self.root):
            errors = validate_complete_indexes(["paper_a", "paper_b"], "clip")
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("vector checksum differs" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
