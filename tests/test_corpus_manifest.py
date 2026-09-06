"""Regressions for deterministic, paper-disjoint experimental corpus identity."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from pydantic import ValidationError

from backend.services.paper_finalize_service import PaperFinalizeService
from evaluation.corpus.manifest import (
    CorpusSelection,
    build_corpus_data_card,
    build_corpus_manifest,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_corpus_manifest,
)


def _write_pdf(path: Path, title: str) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    page.insert_text((50, 80), title, fontsize=14)
    page.insert_text((50, 120), "1 Results", fontsize=12)
    page.insert_text((50, 150), "The measured result is 92 percent.", fontsize=12)
    document.save(str(path))
    document.close()


class TestCorpusManifest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.papers_dir = self.root / "papers"
        self.selection = CorpusSelection(
            selection_id="test_corpus_v1",
            development_paper_ids=["paper_a"],
            test_paper_ids=["paper_b"],
            source_evaluation_files=[],
            selection_rule="Unit-test split declared before evaluation.",
        )
        for paper_id in self.selection.all_paper_ids:
            source = self.root / f"{paper_id}.pdf"
            _write_pdf(source, paper_id)
            with patch(
                "backend.services.ingestion_service.is_docling_available",
                return_value=False,
            ):
                PaperFinalizeService.finalize(
                    source,
                    paper_id,
                    {"title": paper_id, "authors": ["Test Author"], "year": 2026},
                    target_dir=self.papers_dir / paper_id,
                )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_manifest_is_deterministic_and_validates_against_sources(self) -> None:
        first = build_corpus_manifest(self.selection, papers_dir=self.papers_dir)
        second = build_corpus_manifest(self.selection, papers_dir=self.papers_dir)

        self.assertEqual(first, second)
        self.assertEqual(first.paper_count, 2)
        self.assertEqual(first.development_paper_count, 1)
        self.assertEqual(first.test_paper_count, 1)
        self.assertEqual(
            validate_corpus_manifest(
                first,
                self.selection,
                papers_dir=self.papers_dir,
            ),
            [],
        )
        self.assertTrue(all(paper.visual_artifacts for paper in first.papers))

    def test_source_tampering_invalidates_frozen_manifest(self) -> None:
        manifest = build_corpus_manifest(self.selection, papers_dir=self.papers_dir)
        (self.papers_dir / "paper_b" / "paper.pdf").write_bytes(b"tampered")

        errors = validate_corpus_manifest(
            manifest,
            self.selection,
            papers_dir=self.papers_dir,
        )

        self.assertTrue(errors)
        self.assertIn("source corpus validation failed", errors[-1])

    def test_required_derived_index_is_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks required index manifests"):
            build_corpus_manifest(
                self.selection,
                papers_dir=self.papers_dir,
                required_index_manifests=("colqwen_page_manifest.json",),
            )

    def test_required_page_index_payload_tampering_is_fail_closed(self) -> None:
        for paper_id in self.selection.all_paper_ids:
            directory = self.papers_dir / paper_id
            units = json.loads((directory / "visual_units.json").read_text())
            page_units = [unit for unit in units if unit["unit_type"] == "page"]
            clip_rows = [
                {
                    key: unit[key]
                    for key in ("visual_id", "page", "image_relpath", "image_sha256")
                }
                for unit in page_units
            ]
            clip_vectors = directory / "visual_page_embeddings.npy"
            clip_vectors.write_bytes(b"clip-index")
            (directory / "visual_page_embeddings_manifest.json").write_text(json.dumps({
                "source_paper_id": paper_id,
                "rows": clip_rows,
                "rows_sha256": sha256_bytes(canonical_json_bytes(clip_rows)),
                "vector_sha256": sha256_file(clip_vectors),
            }))

            colqwen_rows = [
                {
                    key: unit[key]
                    for key in (
                        "visual_id",
                        "page",
                        "image_relpath",
                        "image_sha256",
                        "width_px",
                        "height_px",
                    )
                }
                for unit in page_units
            ]
            checksums = {}
            for name in (
                "colqwen_page_metadata.json",
                "colqwen_page_offsets.npy",
                "colqwen_page_vectors.npy",
            ):
                path = directory / name
                path.write_bytes(name.encode("utf-8"))
                checksums[name] = sha256_file(path)
            (directory / "colqwen_page_manifest.json").write_text(json.dumps({
                "source_paper_id": paper_id,
                "rows": colqwen_rows,
                "rows_sha256": sha256_bytes(canonical_json_bytes(colqwen_rows)),
                "checksums": checksums,
            }))

        manifest = build_corpus_manifest(
            self.selection,
            papers_dir=self.papers_dir,
            required_index_manifests=(
                "visual_page_embeddings_manifest.json",
                "colqwen_page_manifest.json",
            ),
        )
        (self.papers_dir / "paper_b" / "colqwen_page_vectors.npy").write_bytes(
            b"tampered"
        )

        errors = validate_corpus_manifest(
            manifest,
            self.selection,
            papers_dir=self.papers_dir,
        )
        self.assertTrue(any("ColQwen checksum differs" in error for error in errors))

    def test_selection_source_rejects_any_arxiv_id_from_test_split(self) -> None:
        (self.root / "cases.json").write_text(
            '[{"anchor_paper_id":"paper_a",'
            '"expected_secondary_arxiv_id":"2401.12345"}]',
            encoding="utf-8",
        )
        selection = CorpusSelection(
            selection_id="leak_test",
            development_paper_ids=["paper_a"],
            test_paper_ids=["2401.12345"],
            source_evaluation_files=["cases.json"],
            selection_rule="Reject indirect held-out references.",
        )
        with (
            patch("evaluation.corpus.manifest.ROOT", self.root),
            self.assertRaisesRegex(ValueError, "non-development papers"),
        ):
            build_corpus_manifest(selection, papers_dir=self.papers_dir)

    def test_data_card_discloses_parser_and_degraded_papers(self) -> None:
        manifest = build_corpus_manifest(self.selection, papers_dir=self.papers_dir)
        card = build_corpus_data_card(manifest)

        self.assertEqual(card.corpus_sha256, manifest.corpus_sha256)
        self.assertEqual(card.degraded_mode_counts, {"development": 1, "test": 1})
        self.assertEqual(card.clean_mode_counts, {"development": 0, "test": 0})
        self.assertEqual(card.degraded_paper_ids, ["paper_a", "paper_b"])
        self.assertEqual(card.clean_paper_ids, [])
        self.assertEqual(sum(card.parser_engine_counts.values()), 2)

    def test_selection_rejects_overlap_and_duplicate_ids(self) -> None:
        with self.assertRaises(ValidationError):
            CorpusSelection(
                selection_id="invalid",
                development_paper_ids=["paper_a", "paper_a"],
                test_paper_ids=["paper_a"],
                selection_rule="Invalid overlap.",
            )


if __name__ == "__main__":
    unittest.main()
