"""Consistency and rollback regressions for canonical paper publication."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from backend.schemas.evidence import EvidenceAST
from backend.services.paper_finalize_service import PaperFinalizeService
from backend.services.pdf_service import read_json
from backend.services.storage_service import StorageService


def _write_pdf(path: Path, lines: list[str]) -> None:
    document = fitz.open()
    page = document.new_page(width=600, height=800)
    for index, line in enumerate(lines):
        page.insert_text((50, 80 + index * 35), line, fontsize=12)
    document.save(str(path))
    document.close()


class TestPaperFinalizeService(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.source_pdf = self.root / "source.pdf"
        self.destination = self.root / "papers" / "paper_alpha"
        _write_pdf(
            self.source_pdf,
            [
                "ScholAR Transaction Test",
                "1 Introduction",
                "This paper reports a deterministic ingestion consistency result.",
            ],
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _finalize(self, *, title: str = "Transaction Test") -> dict:
        with patch("backend.services.ingestion_service.is_docling_available", return_value=False):
            return PaperFinalizeService.finalize(
                self.source_pdf,
                "paper_alpha",
                {
                    "id": "external-alpha",
                    "title": title,
                    "authors": ["Test Author"],
                    "year": "2026",
                    "summary": "Preserved route summary.",
                    "categories": ["cs.CL"],
                    "pdf_url": "https://example.invalid/paper.pdf",
                    "source": "upload",
                    "filename": "uploaded.pdf",
                },
                target_dir=self.destination,
            )

    def test_finalize_publishes_consistent_json_ast_sqlite_generation(self) -> None:
        response = self._finalize()
        expected_files = {
            "paper.pdf",
            "metadata.json",
            "pages.json",
            "chunks.json",
            "figures.json",
            "visual_units.json",
            "evidence_ast.json",
            "document.db",
            "ingestion_manifest.json",
        }
        self.assertTrue(expected_files.issubset({path.name for path in self.destination.iterdir()}))

        metadata = read_json(self.destination / "metadata.json")
        pages = read_json(self.destination / "pages.json")
        chunks = read_json(self.destination / "chunks.json")
        figures = read_json(self.destination / "figures.json")
        visual_units = read_json(self.destination / "visual_units.json")
        manifest = read_json(self.destination / "ingestion_manifest.json")
        ast = EvidenceAST.model_validate(read_json(self.destination / "evidence_ast.json"))

        self.assertEqual(response["paper_id"], "paper_alpha")
        self.assertEqual((response["pages"], response["chunks"], response["figures"]), (len(pages), len(chunks), len(figures)))
        self.assertEqual(response["visual_units"], len(visual_units))
        self.assertEqual(metadata["id"], "external-alpha")
        self.assertEqual(metadata["local_id"], "paper_alpha")
        self.assertEqual(metadata["source"], "upload")
        self.assertEqual(metadata["summary"], "Preserved route summary.")
        self.assertEqual(metadata["page_count"], len(pages))
        self.assertEqual(metadata["chunk_count"], len(chunks))
        self.assertEqual(metadata["visual_unit_count"], len(visual_units))
        self.assertEqual(ast.page_count, len(pages))
        self.assertEqual(set(manifest["chunk_hashes"]), {chunk["chunk_id"] for chunk in chunks})
        self.assertEqual({chunk["evidence_id"] for chunk in chunks}, {block.evidence_id for block in ast.blocks})
        self.assertTrue(all(chunk["source_paper_id"] == "paper_alpha" for chunk in chunks))
        self.assertTrue(all("section_title" in chunk and "retrieval_text" in chunk for chunk in chunks))
        page_units = [unit for unit in visual_units if unit["unit_type"] == "page"]
        self.assertEqual(len(page_units), len(pages))
        self.assertEqual([unit["page"] for unit in page_units], list(range(1, len(pages) + 1)))
        for unit in page_units:
            image_path = self.destination / unit["image_relpath"]
            self.assertTrue(image_path.is_file())
            self.assertEqual(
                hashlib.sha256(image_path.read_bytes()).hexdigest(),
                unit["image_sha256"],
            )
        self.assertEqual(manifest["schema_version"], "2.0")
        self.assertEqual(manifest["visual_unit_count"], len(visual_units))

        with sqlite3.connect(self.destination / "document.db") as connection:
            paper_row = connection.execute(
                "SELECT paper_id, pages, chunks, figures FROM papers WHERE paper_id = ?",
                ("paper_alpha",),
            ).fetchone()
            db_chunk_ids = {
                row[0]
                for row in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE paper_id = ?", ("paper_alpha",)
                )
            }
        self.assertEqual(paper_row, ("paper_alpha", len(pages), len(chunks), len(figures)))
        self.assertEqual(db_chunk_ids, {chunk["chunk_id"] for chunk in chunks})

    def test_refinalize_replaces_generation_and_stale_database_rows(self) -> None:
        self._finalize(title="First Generation")
        first_manifest = read_json(self.destination / "ingestion_manifest.json")

        _write_pdf(
            self.source_pdf,
            [
                "Replacement Generation",
                "1 Results",
                "The replacement contains one complete evidence generation.",
            ],
        )
        self._finalize(title="Second Generation")
        second_manifest = read_json(self.destination / "ingestion_manifest.json")
        metadata = read_json(self.destination / "metadata.json")
        chunks = read_json(self.destination / "chunks.json")

        self.assertNotEqual(first_manifest["generation_id"], second_manifest["generation_id"])
        self.assertNotEqual(first_manifest["pdf_sha256"], second_manifest["pdf_sha256"])
        self.assertEqual(metadata["title"], "Second Generation")
        with sqlite3.connect(self.destination / "document.db") as connection:
            db_ids = {
                row[0]
                for row in connection.execute(
                    "SELECT chunk_id FROM chunks WHERE paper_id = ?", ("paper_alpha",)
                )
            }
        self.assertEqual(db_ids, {chunk["chunk_id"] for chunk in chunks})

    def test_storage_sync_removes_stale_children(self) -> None:
        db_path = self.root / "standalone.db"
        metadata = {"title": "A", "page_count": 2}
        initial = [
            {"chunk_id": "chunk_001", "page": 1, "text": "one"},
            {"chunk_id": "chunk_002", "page": 2, "text": "two"},
        ]
        replacement = [{"chunk_id": "chunk_003", "page": 1, "text": "three"}]
        StorageService.sync_paper_to_db("paper", metadata, initial, db_path=db_path)
        StorageService.sync_paper_to_db("paper", metadata, replacement, db_path=db_path)
        with sqlite3.connect(db_path) as connection:
            ids = {row[0] for row in connection.execute("SELECT chunk_id FROM chunks")}
        self.assertEqual(ids, {"chunk_003"})

    def test_visual_image_tamper_invalidates_published_bundle(self) -> None:
        self._finalize()
        visual_units = read_json(self.destination / "visual_units.json")
        page_unit = next(
            unit for unit in visual_units if unit["unit_type"] == "page"
        )
        (self.destination / page_unit["image_relpath"]).write_bytes(b"tampered")

        self.assertIsNone(
            PaperFinalizeService.load_if_complete(
                "paper_alpha", target_dir=self.destination
            )
        )

    def test_failed_replacement_preserves_previous_complete_generation(self) -> None:
        self._finalize(title="Stable Generation")
        before = {
            path.relative_to(self.destination).as_posix(): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }

        _write_pdf(self.source_pdf, ["Broken Replacement", "1 Failure", "This must not publish."])
        with (
            patch("backend.services.ingestion_service.is_docling_available", return_value=False),
            patch.object(PaperFinalizeService, "_validate_staged_bundle", side_effect=RuntimeError("injected validation failure")),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected validation failure"):
                PaperFinalizeService.finalize(
                    self.source_pdf,
                    "paper_alpha",
                    {"title": "Broken Generation"},
                    target_dir=self.destination,
                )

        after = {
            path.relative_to(self.destination).as_posix(): path.read_bytes()
            for path in self.destination.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertFalse(any(self.destination.parent.glob(".paper_alpha.staging-*")))
        self.assertFalse(any(self.destination.parent.glob(".paper_alpha.backup-*")))


if __name__ == "__main__":
    unittest.main()
