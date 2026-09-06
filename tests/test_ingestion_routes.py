"""Route-level regressions ensuring every acquisition path uses one finalized schema."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.pdf_service import atomic_write_bytes, read_json, safe_paper_id


def _pdf_bytes() -> bytes:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=600, height=800)
        page.insert_text((50, 80), "Canonical Route Paper", fontsize=16)
        page.insert_text((50, 130), "1 Introduction", fontsize=14)
        page.insert_text((50, 165), "This paper provides consistent route ingestion evidence.", fontsize=11)
        document.save(str(path))
        document.close()
        return path.read_bytes()


class TestIngestionRoutes(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name) / "papers"
        self.root.mkdir(parents=True)
        self.content = _pdf_bytes()
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _paper_dir(self, paper_id: str) -> Path:
        return self.root / safe_paper_id(paper_id)

    async def _download(self, _url: str, destination: Path) -> None:
        atomic_write_bytes(destination, self.content)

    def _assert_consistent(self, paper_id: str, payload: dict) -> None:
        directory = self._paper_dir(paper_id)
        for name in (
            "paper.pdf",
            "metadata.json",
            "pages.json",
            "chunks.json",
            "figures.json",
            "evidence_ast.json",
            "document.db",
            "ingestion_manifest.json",
        ):
            self.assertTrue((directory / name).is_file(), name)
        metadata = read_json(directory / "metadata.json")
        pages = read_json(directory / "pages.json")
        chunks = read_json(directory / "chunks.json")
        figures = read_json(directory / "figures.json")
        ast = read_json(directory / "evidence_ast.json")
        self.assertEqual(payload["pages"], len(pages))
        self.assertEqual(payload["chunks"], len(chunks))
        self.assertEqual(payload["figures"], len(figures))
        self.assertEqual(metadata["page_count"], len(pages))
        self.assertEqual(metadata["chunk_count"], len(chunks))
        self.assertEqual(ast["page_count"], len(pages))
        self.assertEqual({chunk["evidence_id"] for chunk in chunks}, {block["evidence_id"] for block in ast["blocks"]})
        self.assertTrue(all(chunk["source_paper_id"] == paper_id for chunk in chunks))
        with sqlite3.connect(directory / "document.db") as connection:
            row = connection.execute(
                "SELECT pages, chunks, figures FROM papers WHERE paper_id = ?", (paper_id,)
            ).fetchone()
        self.assertEqual(row, (len(pages), len(chunks), len(figures)))

    def test_prepare_and_upload_publish_same_canonical_bundle(self) -> None:
        with (
            patch("backend.main.paper_dir", side_effect=self._paper_dir),
            patch("backend.main.download_pdf", side_effect=self._download),
            patch("backend.services.ingestion_service.is_docling_available", return_value=False),
        ):
            prepare = self.client.post(
                "/api/papers/prepare",
                json={
                    "id": "1234.5678",
                    "title": "Prepared Paper",
                    "authors": ["A. Author"],
                    "year": "2026",
                    "summary": "Prepared summary",
                    "abstract": "Prepared abstract",
                    "categories": ["cs.CL"],
                    "pdf_url": "https://example.invalid/paper.pdf",
                    "abs_url": "https://example.invalid/abs",
                    "published": "2026-01-01",
                },
            )
            upload = self.client.post(
                "/api/papers/upload",
                files={"file": ("local.pdf", self.content, "application/pdf")},
                data={"title": "Uploaded Paper"},
            )

        self.assertEqual(prepare.status_code, 200, prepare.text)
        self.assertEqual(upload.status_code, 200, upload.text)
        prepared_payload = prepare.json()
        uploaded_payload = upload.json()
        self._assert_consistent(prepared_payload["paper_id"], prepared_payload)
        self._assert_consistent(uploaded_payload["paper_id"], uploaded_payload)
        self.assertEqual(
            read_json(self._paper_dir(uploaded_payload["paper_id"]) / "metadata.json")["source"],
            "upload",
        )

    def test_reference_is_marked_ingested_only_after_finalized_publish(self) -> None:
        reference = {
            "arxiv_id": "9999.0001",
            "pdf_url": "https://example.invalid/reference.pdf",
            "title": "Reference Paper",
            "authors": ["R. Author"],
            "year": "2025",
            "abstract": "Reference abstract",
            "abs_url": "https://example.invalid/reference",
            "ingested": False,
        }
        dummy = self.root / "anchor"
        dummy.mkdir()
        metadata_path = dummy / "metadata.json"
        pages_path = dummy / "pages.json"
        chunks_path = dummy / "chunks.json"
        pdf_path = dummy / "paper.pdf"
        for path, value in (
            (metadata_path, {}),
            (pages_path, []),
            (chunks_path, []),
        ):
            path.write_text(json.dumps(value), encoding="utf-8")
        pdf_path.write_bytes(self.content)

        with (
            patch("backend.main.paper_dir", side_effect=self._paper_dir),
            patch("backend.main._paths_or_404", return_value=(metadata_path, pages_path, chunks_path, pdf_path)),
            patch("backend.main.load_references", return_value=[reference]),
            patch("backend.main.mark_reference_ingested") as mark_ingested,
            patch("backend.main.download_pdf", side_effect=self._download),
            patch("backend.services.ingestion_service.is_docling_available", return_value=False),
        ):
            response = self.client.post("/api/papers/anchor/references/0/ingest")

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self._assert_consistent("9999.0001", {**payload, "pages": 1, "figures": 0})
        mark_ingested.assert_called_once_with("anchor", 0, "9999.0001")


if __name__ == "__main__":
    unittest.main()
