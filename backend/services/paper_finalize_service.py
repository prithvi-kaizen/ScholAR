"""Transactional publication of a complete, internally consistent paper bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any, Iterator

try:  # Unix process lock; the in-process lock remains the portable fallback.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-Unix platforms
    fcntl = None  # type: ignore[assignment]

from backend.schemas.evidence import EvidenceAST, ParserAblationConfig
from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType
from backend.services.ingestion_service import DualEngineIngestionService
from backend.services.pdf_service import atomic_write_bytes, paper_dir, read_json, write_json
from backend.services.storage_service import StorageService


_LOCKS_GUARD = threading.Lock()
_PAPER_LOCKS: dict[str, threading.Lock] = {}


class PaperFinalizeService:
    """Build, validate, and atomically publish every primary paper artifact."""

    SCHEMA_VERSION = "2.0"
    MANIFEST_NAME = "ingestion_manifest.json"
    REQUIRED_FILES = (
        "paper.pdf",
        "evidence_ast.json",
        "pages.json",
        "chunks.json",
        "figures.json",
        "visual_units.json",
        "metadata.json",
        "document.db",
        MANIFEST_NAME,
    )

    @classmethod
    def finalize(
        cls,
        pdf_path: Path,
        paper_id: str,
        metadata: dict[str, Any] | None = None,
        *,
        target_dir: Path | None = None,
        config: ParserAblationConfig | None = None,
    ) -> dict[str, Any]:
        """Finalize an already-acquired PDF and return the current route bundle.

        All primary artifacts are built in a sibling staging directory. The prior
        generation remains visible until the staged generation passes validation,
        after which a directory rename publishes the new bundle as one unit.
        """
        source_pdf = Path(pdf_path)
        if not source_pdf.is_file():
            raise RuntimeError("Acquired PDF does not exist")

        destination = Path(target_dir) if target_dir is not None else paper_dir(paper_id)
        destination.parent.mkdir(parents=True, exist_ok=True)
        generation_id = uuid.uuid4().hex
        stage: Path | None = None
        backup = destination.parent / f".{destination.name}.backup-{generation_id}"

        with cls._paper_lock(destination):
            try:
                stage = Path(tempfile.mkdtemp(
                    prefix=f".{destination.name}.staging-{generation_id}-",
                    dir=str(destination.parent),
                ))
                staged_pdf = stage / "paper.pdf"
                atomic_write_bytes(staged_pdf, source_pdf.read_bytes())

                normalized_metadata = dict(metadata or {})
                normalized_metadata["local_id"] = paper_id
                normalized_metadata["document_id"] = paper_id
                normalized_metadata.setdefault("id", paper_id)
                normalized_metadata.setdefault("filename", source_pdf.name)

                raw_authors = normalized_metadata.get("authors", [])
                authors = list(raw_authors) if isinstance(raw_authors, (list, tuple)) else [str(raw_authors)]
                DualEngineIngestionService.ingest_paper(
                    pdf_path=staged_pdf,
                    document_id=paper_id,
                    title=str(normalized_metadata.get("title", "")),
                    authors=authors,
                    year=normalized_metadata.get("year", 0),
                    abstract=str(
                        normalized_metadata.get("abstract")
                        or normalized_metadata.get("summary")
                        or ""
                    ),
                    config=config,
                    target_dir=stage,
                    metadata=normalized_metadata,
                )

                staged_metadata = read_json(stage / "metadata.json")
                pages = read_json(stage / "pages.json")
                chunks = read_json(stage / "chunks.json")
                figures = read_json(stage / "figures.json")
                visual_units = read_json(stage / "visual_units.json")
                StorageService.sync_paper_to_db(
                    paper_id,
                    staged_metadata,
                    chunks,
                    figures,
                    db_path=stage / "document.db",
                )

                manifest = cls._build_manifest(
                    stage,
                    paper_id,
                    generation_id,
                    pages,
                    chunks,
                    figures,
                    visual_units,
                )
                write_json(stage / cls.MANIFEST_NAME, manifest)
                cls._validate_staged_bundle(stage, paper_id)
                cls._publish_staged_directory(stage, destination, backup)
                stage = None

                return {
                    "paper_id": paper_id,
                    "metadata": staged_metadata,
                    "pages": len(pages),
                    "chunks": len(chunks),
                    "figures": len(figures),
                    "visual_units": len(visual_units),
                }
            finally:
                if stage is not None and stage.exists():
                    shutil.rmtree(stage, ignore_errors=True)
                # A backup should survive cleanup only when restoration itself
                # failed and no destination exists; never delete the sole copy.
                if backup.exists() and destination.exists():
                    shutil.rmtree(backup, ignore_errors=True)

    @classmethod
    def load_if_complete(
        cls,
        paper_id: str,
        *,
        target_dir: Path | None = None,
    ) -> dict[str, Any] | None:
        """Return a validated published bundle, or ``None`` for legacy/partial data."""
        destination = Path(target_dir) if target_dir is not None else paper_dir(paper_id)
        if not destination.is_dir():
            return None
        try:
            cls._validate_staged_bundle(destination, paper_id)
            metadata = read_json(destination / "metadata.json")
            pages = read_json(destination / "pages.json")
            chunks = read_json(destination / "chunks.json")
            figures = read_json(destination / "figures.json")
            visual_units = read_json(destination / "visual_units.json")
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError, sqlite3.Error):
            return None
        return {
            "paper_id": paper_id,
            "metadata": metadata,
            "pages": len(pages),
            "chunks": len(chunks),
            "figures": len(figures),
            "visual_units": len(visual_units),
        }

    @classmethod
    def _build_manifest(
        cls,
        stage: Path,
        paper_id: str,
        generation_id: str,
        pages: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        visual_units: list[dict[str, Any]],
    ) -> dict[str, Any]:
        chunk_hashes = {
            str(chunk["chunk_id"]): hashlib.sha256(
                json.dumps(chunk, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            for chunk in chunks
        }
        pdf_sha256 = cls._sha256_file(stage / "paper.pdf")
        chunks_sha256 = cls._sha256_file(stage / "chunks.json")
        counts = {
            "pages": len(pages),
            "chunks": len(chunks),
            "figures": len(figures),
            "visual_units": len(visual_units),
        }
        return {
            "schema_version": cls.SCHEMA_VERSION,
            "generation_id": generation_id,
            "paper_id": paper_id,
            "created_at": time.time(),
            "pdf_sha256": pdf_sha256,
            "chunks_sha256": chunks_sha256,
            "chunk_sha256": chunks_sha256,
            "chunk_hashes": chunk_hashes,
            "page_count": len(pages),
            "chunk_count": len(chunks),
            "figure_count": len(figures),
            "visual_unit_count": len(visual_units),
            "counts": counts,
            "derived_artifacts_excluded": [
                "embeddings",
                "visual_embeddings",
                "visual_page_embeddings",
            ],
        }

    @classmethod
    def _validate_staged_bundle(cls, stage: Path, paper_id: str) -> None:
        missing = [name for name in cls.REQUIRED_FILES if not (stage / name).is_file()]
        if missing:
            raise RuntimeError(f"Incomplete staged paper bundle: {', '.join(missing)}")

        metadata = read_json(stage / "metadata.json")
        pages = read_json(stage / "pages.json")
        chunks = read_json(stage / "chunks.json")
        figures = read_json(stage / "figures.json")
        raw_visual_units = read_json(stage / "visual_units.json")
        visual_units = [VisualDocumentUnit.model_validate(item) for item in raw_visual_units]
        manifest = read_json(stage / cls.MANIFEST_NAME)
        ast = EvidenceAST.model_validate(read_json(stage / "evidence_ast.json"))

        if not isinstance(metadata, dict) or not all(isinstance(value, list) for value in (pages, chunks, figures, raw_visual_units)):
            raise RuntimeError("Staged JSON artifacts have invalid top-level types")
        if ast.document_id != paper_id or metadata.get("document_id") != paper_id:
            raise RuntimeError("Paper identity differs between metadata and EvidenceAST")
        if metadata.get("local_id") != paper_id or manifest.get("paper_id") != paper_id:
            raise RuntimeError("Paper identity differs between manifest and metadata")

        expected_counts = (len(pages), len(chunks), len(figures))
        metadata_counts = (
            int(metadata.get("page_count", -1)),
            int(metadata.get("chunk_count", -1)),
            int(metadata.get("figure_count", -1)),
        )
        metadata_alias_counts = (
            int(metadata.get("pages", -1)),
            int(metadata.get("chunks", -1)),
            int(metadata.get("figures", -1)),
        )
        manifest_counts = (
            int(manifest.get("page_count", -1)),
            int(manifest.get("chunk_count", -1)),
            int(manifest.get("figure_count", -1)),
        )
        if metadata_counts != expected_counts or metadata_alias_counts != expected_counts:
            raise RuntimeError("Metadata artifact counts are inconsistent")
        if manifest_counts != expected_counts or ast.page_count != len(pages):
            raise RuntimeError("Manifest or EvidenceAST counts are inconsistent")
        if int(metadata.get("visual_unit_count", -1)) != len(visual_units):
            raise RuntimeError("Metadata visual-unit count is inconsistent")
        if int(manifest.get("visual_unit_count", -1)) != len(visual_units):
            raise RuntimeError("Manifest visual-unit count is inconsistent")

        page_numbers = [int(page.get("page", 0)) for page in pages]
        if page_numbers != list(range(1, len(pages) + 1)):
            raise RuntimeError("Page identifiers are not contiguous and 1-based")

        block_ids = [block.evidence_id for block in ast.blocks]
        chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]
        chunk_evidence_ids = [str(chunk.get("evidence_id", "")) for chunk in chunks]
        figure_ids = [str(figure.get("figure_id", "")) for figure in figures]
        for label, identifiers in (
            ("EvidenceAST block", block_ids),
            ("chunk", chunk_ids),
            ("figure", figure_ids),
        ):
            if not all(identifiers) or len(identifiers) != len(set(identifiers)):
                raise RuntimeError(f"{label} identifiers are empty or duplicated")
        if set(chunk_evidence_ids) != set(block_ids):
            raise RuntimeError("Chunk-to-EvidenceAST identifiers are inconsistent")

        for block in ast.blocks:
            if block.document_id != paper_id or not 1 <= block.page <= len(pages):
                raise RuntimeError("EvidenceAST block has invalid identity or page")
        for chunk in chunks:
            if chunk.get("document_id") != paper_id or chunk.get("source_paper_id") != paper_id:
                raise RuntimeError("Chunk has invalid document provenance")
            if not 1 <= int(chunk.get("page", 0)) <= len(pages):
                raise RuntimeError("Chunk has invalid page")
        for figure in figures:
            if not 1 <= int(figure.get("page", 0)) <= len(pages):
                raise RuntimeError("Figure has invalid page")
            image_file = str(figure.get("image_file") or "")
            if not image_file or Path(image_file).name != image_file:
                raise RuntimeError("Figure has an unsafe or missing image filename")
            if not (stage / "figures" / image_file).is_file():
                raise RuntimeError("Figure image is missing")

        visual_ids = [unit.visual_id for unit in visual_units]
        if not visual_ids or len(visual_ids) != len(set(visual_ids)):
            raise RuntimeError("Visual-unit identifiers are empty or duplicated")
        page_units = [unit for unit in visual_units if unit.unit_type == VisualUnitType.PAGE]
        if [unit.page for unit in page_units] != list(range(1, len(pages) + 1)):
            raise RuntimeError("Visual units must contain one ordered full-page image per page")
        for unit in visual_units:
            if unit.document_id != paper_id or unit.source_paper_id != paper_id:
                raise RuntimeError("Visual unit has invalid source provenance")
            if not 1 <= unit.page <= len(pages):
                raise RuntimeError("Visual unit has invalid page")
            image_path = stage.joinpath(*Path(unit.image_relpath).parts)
            try:
                image_path.resolve().relative_to(stage.resolve())
            except ValueError as exc:
                raise RuntimeError("Visual unit image escapes the paper bundle") from exc
            if (
                image_path.is_symlink()
                or not image_path.is_file()
                or cls._sha256_file(image_path) != unit.image_sha256
            ):
                raise RuntimeError("Visual unit image is missing or has an invalid checksum")

        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise RuntimeError("Manifest schema version is invalid")
        if manifest.get("pdf_sha256") != cls._sha256_file(stage / "paper.pdf"):
            raise RuntimeError("Manifest PDF hash is inconsistent")
        if manifest.get("chunks_sha256") != cls._sha256_file(stage / "chunks.json"):
            raise RuntimeError("Manifest chunk artifact hash is inconsistent")
        expected_chunk_hashes = cls._build_manifest(
            stage,
            paper_id,
            str(manifest.get("generation_id", "")),
            pages,
            chunks,
            figures,
            raw_visual_units,
        )["chunk_hashes"]
        if manifest.get("chunk_hashes") != expected_chunk_hashes:
            raise RuntimeError("Manifest per-chunk hashes are inconsistent")

        with sqlite3.connect(str(stage / "document.db")) as conn:
            paper_row = conn.execute(
                "SELECT paper_id, pages, chunks, figures FROM papers WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            if paper_row != (paper_id, *expected_counts):
                raise RuntimeError("SQLite paper counts or identity are inconsistent")
            db_chunk_ids = {
                row[0] for row in conn.execute("SELECT chunk_id FROM chunks WHERE paper_id = ?", (paper_id,))
            }
            db_figure_ids = {
                row[0] for row in conn.execute("SELECT figure_id FROM figures WHERE paper_id = ?", (paper_id,))
            }
        if db_chunk_ids != set(chunk_ids) or db_figure_ids != set(figure_ids):
            raise RuntimeError("SQLite child identifiers are inconsistent with JSON")

    @classmethod
    def _publish_staged_directory(cls, stage: Path, destination: Path, backup: Path) -> None:
        had_previous_generation = destination.exists()
        if had_previous_generation:
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        finally:
            cls._fsync_directory(destination.parent)

        if backup.exists():
            shutil.rmtree(backup, ignore_errors=True)
            cls._fsync_directory(destination.parent)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @classmethod
    @contextmanager
    def _paper_lock(cls, destination: Path) -> Iterator[None]:
        key = str(destination.resolve())
        with _LOCKS_GUARD:
            process_lock = _PAPER_LOCKS.setdefault(key, threading.Lock())
        with process_lock:
            lock_path = destination.parent / f".{destination.name}.lock"
            with lock_path.open("a+b") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        with suppress(OSError):
                            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
