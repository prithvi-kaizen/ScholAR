from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.schemas.document import (
    BoundingBox,
    CoordinateTransform,
    DocumentMetadata,
    EvidenceBlock,
    PageRender,
    ScientificDocument,
    SectionNode,
    TableBlock,
    VisualEvidence,
    VisualRegion,
)
from backend.services.pdf_service import paper_dir, read_json, write_json


def _get_db_path(paper_id: str) -> Path:
    directory = paper_dir(paper_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "document.db"


def _init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT,
                authors_json TEXT,
                year TEXT,
                summary TEXT,
                categories_json TEXT,
                pdf_url TEXT,
                pages INTEGER DEFAULT 0,
                chunks INTEGER DEFAULT 0,
                figures INTEGER DEFAULT 0,
                source TEXT DEFAULT 'arxiv',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS sections (
                section_id TEXT PRIMARY KEY,
                paper_id TEXT,
                title TEXT,
                level INTEGER DEFAULT 1,
                section_path_json TEXT,
                parent_section_id TEXT,
                page INTEGER,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
            );

            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                paper_id TEXT,
                page INTEGER,
                section_title TEXT,
                section_path_json TEXT,
                chunk_type TEXT,
                text TEXT,
                retrieval_text TEXT,
                paragraph_text TEXT,
                is_figure_chunk INTEGER DEFAULT 0,
                char_start INTEGER DEFAULT 0,
                char_end INTEGER DEFAULT 0,
                source_paper_id TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
            );

            CREATE TABLE IF NOT EXISTS figures (
                figure_id TEXT PRIMARY KEY,
                paper_id TEXT,
                page INTEGER,
                label TEXT,
                figure_type TEXT,
                caption TEXT,
                image_file TEXT,
                bbox_json TEXT,
                ocr_text TEXT,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
            );

            CREATE TABLE IF NOT EXISTS visual_regions (
                region_id TEXT PRIMARY KEY,
                parent_evidence_id TEXT,
                paper_id TEXT,
                page INTEGER,
                role TEXT,
                bbox_page_json TEXT,
                bbox_crop_json TEXT,
                proposal_source TEXT DEFAULT 'vlm',
                proposer_model_id TEXT,
                verification TEXT DEFAULT 'UNCERTAIN',
                confidence REAL DEFAULT 1.0,
                FOREIGN KEY (paper_id) REFERENCES papers (paper_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_paper_page ON chunks (paper_id, page);
            CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks (chunk_type);
            CREATE INDEX IF NOT EXISTS idx_figures_paper ON figures (paper_id);
            CREATE INDEX IF NOT EXISTS idx_regions_parent ON visual_regions (parent_evidence_id);
        """)
    return conn


class StorageService:
    """Unified storage service providing relational SQLite querying and AST reconstruction."""

    @classmethod
    def sync_paper_to_db(
        cls,
        paper_id: str,
        metadata: dict[str, Any],
        chunks: list[dict[str, Any]],
        figures: list[dict[str, Any]] | None = None,
    ) -> None:
        """Sync in-memory or JSON paper artifacts to the relational database."""
        db_path = _get_db_path(paper_id)
        conn = _init_db(db_path)
        figures = figures or []

        with conn:
            # 1. Upsert paper record
            conn.execute(
                """
                INSERT OR REPLACE INTO papers (
                    paper_id, title, authors_json, year, summary, categories_json,
                    pdf_url, pages, chunks, figures, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    metadata.get("title", ""),
                    json.dumps(metadata.get("authors", [])),
                    str(metadata.get("year", "")),
                    metadata.get("summary", ""),
                    json.dumps(metadata.get("categories", [])),
                    metadata.get("pdf_url", ""),
                    int(metadata.get("pages", 0)),
                    len(chunks),
                    len(figures),
                    metadata.get("source", "arxiv"),
                ),
            )

            # 2. Sync sections from chunk hierarchy
            seen_sections: set[str] = set()
            for chunk in chunks:
                sec_path = chunk.get("section_path") or [chunk.get("section_title", "Body")]
                sec_title = chunk.get("section_title", "Body")
                sec_id = f"sec_{paper_id}_{sec_title.replace(' ', '_')}"
                if sec_id not in seen_sections:
                    seen_sections.add(sec_id)
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO sections (
                            section_id, paper_id, title, level, section_path_json, page
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            sec_id,
                            paper_id,
                            sec_title,
                            len(sec_path),
                            json.dumps(sec_path),
                            int(chunk.get("page", 1)),
                        ),
                    )

            # 3. Sync chunks
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, paper_id, page, section_title, section_path_json,
                        chunk_type, text, retrieval_text, paragraph_text,
                        is_figure_chunk, char_start, char_end, source_paper_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.get("chunk_id", ""),
                        paper_id,
                        int(chunk.get("page", 1)),
                        chunk.get("section_title", ""),
                        json.dumps(chunk.get("section_path", [])),
                        chunk.get("chunk_type", "body"),
                        chunk.get("text", ""),
                        chunk.get("retrieval_text", chunk.get("text", "")),
                        chunk.get("paragraph_text", ""),
                        1 if chunk.get("is_figure_chunk") else 0,
                        int(chunk.get("char_start", 0)),
                        int(chunk.get("char_end", 0)),
                        chunk.get("source_paper_id", paper_id),
                    ),
                )

            # 4. Sync figures
            for fig in figures:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO figures (
                        figure_id, paper_id, page, label, figure_type,
                        caption, image_file, bbox_json, ocr_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fig.get("figure_id", ""),
                        paper_id,
                        int(fig.get("page", 1)),
                        fig.get("label", ""),
                        fig.get("figure_type", "figure"),
                        fig.get("caption", ""),
                        fig.get("image_file", ""),
                        json.dumps(fig.get("bbox", {})),
                        fig.get("ocr_text", ""),
                    ),
                )
        conn.close()

    @classmethod
    def query_sql(
        cls,
        paper_id: str,
        sql: str,
        params: tuple = (),
    ) -> list[dict[str, Any]]:
        """Run SQL query against paper's SQLite database."""
        db_path = _get_db_path(paper_id)
        if not db_path.exists():
            return []
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
            return rows
        finally:
            conn.close()

    @classmethod
    def get_document_ast(cls, paper_id: str) -> ScientificDocument | None:
        """Construct full ScientificDocument AST model from relational/JSON storage."""
        directory = paper_dir(paper_id)
        meta_file = directory / "metadata.json"
        chunks_file = directory / "chunks.json"
        figures_file = directory / "figures.json"

        if not meta_file.exists():
            return None

        metadata_raw = read_json(meta_file)
        chunks_raw = read_json(chunks_file) if chunks_file.exists() else []
        figures_raw = read_json(figures_file) if figures_file.exists() else []

        # 1. Document Metadata
        metadata = DocumentMetadata(
            document_id=paper_id,
            source_sha256="",
            filename=metadata_raw.get("filename", "paper.pdf"),
            page_count=int(metadata_raw.get("pages", 1)),
            title=metadata_raw.get("title"),
            authors=metadata_raw.get("authors", []),
            parser_name="pymupdf",
        )

        # 2. Evidence Blocks
        evidence_blocks: list[EvidenceBlock] = []
        for idx, chunk in enumerate(chunks_raw, start=1):
            evidence_blocks.append(
                EvidenceBlock(
                    evidence_id=chunk.get("chunk_id", f"chunk_{idx:03d}"),
                    document_id=paper_id,
                    block_type="caption" if chunk.get("is_figure_chunk") else "paragraph",
                    original_text=chunk.get("original_text") or chunk.get("text", ""),
                    retrieval_text=chunk.get("retrieval_text") or chunk.get("text", ""),
                    page=int(chunk.get("page", 1)),
                    section_path=chunk.get("section_path", []),
                    ordinal=idx,
                    source_paper_id=chunk.get("source_paper_id", paper_id),
                )
            )

        # 3. Visual Evidence
        visual_evidence: list[VisualEvidence] = []
        for fig in figures_raw:
            raw_bbox = fig.get("bbox") or {}
            bbox_norm = BoundingBox(
                x0=raw_bbox.get("x0", 0.0),
                y0=raw_bbox.get("y0", 0.0),
                x1=raw_bbox.get("x1", 1.0),
                y1=raw_bbox.get("y1", 1.0),
                coordinate_space="normalized_page",
            )
            visual_evidence.append(
                VisualEvidence(
                    evidence_id=f"fig_{fig.get('figure_id')}",
                    document_id=paper_id,
                    visual_type="table_image" if fig.get("figure_type") == "table" else "figure",
                    page=int(fig.get("page", 1)),
                    bbox_normalized=bbox_norm,
                    image_path=str(fig.get("image_file", "")),
                    image_sha256="",
                    pixel_width=int(fig.get("pixel_width", 800)),
                    pixel_height=int(fig.get("pixel_height", 600)),
                    figure_label=fig.get("label"),
                    caption=fig.get("caption"),
                    ocr_text=fig.get("ocr_text"),
                )
            )

        return ScientificDocument(
            metadata=metadata,
            evidence_blocks=evidence_blocks,
            visual_evidence=visual_evidence,
        )
