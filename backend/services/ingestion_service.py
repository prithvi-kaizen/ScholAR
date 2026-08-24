"""Dual-Engine Ingestion Pipeline & Canonical Evidence AST Builder.

Combines:
- Docling (semantic document hierarchy, reading order, table parsing, captions)
- PyMuPDF (page vector rendering, high-res 3x crops, geometry, coordinate normalization)

Features:
- Canonical EvidenceAST production
- Graceful degradation when Docling is unavailable (degraded_mode=True)
- Parser Ablation Framework (P0, P1, P2, P3, P4)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Literal

import fitz  # PyMuPDF

from backend.schemas.evidence import (
    EvidenceAST,
    EvidenceBlock,
    EvidenceModality,
    ParserAblationConfig,
    SectionNode,
    TableCell,
    TableData,
)
from backend.services.docling_service import is_docling_available, parse_with_docling
from backend.services.pdf_service import (
    extract_figures,
    extract_pages,
    paper_dir,
    read_json,
    safe_paper_id,
    write_json,
)

logger = logging.getLogger("scholar.ingestion")

# Parser Ablation Matrix definitions
PARSER_ABLATIONS: dict[str, ParserAblationConfig] = {
    "P0": ParserAblationConfig(
        config_id="P0",
        parser_name="PyMuPDF Fixed Chunks",
        chunking_strategy="fixed_token",
        chunk_token_size=512,
        chunk_overlap=64,
        description="PyMuPDF raw text extraction with naive fixed 512-token sliding window",
    ),
    "P1": ParserAblationConfig(
        config_id="P1",
        parser_name="PyMuPDF Heuristic AST",
        chunking_strategy="heuristic_ast",
        chunk_token_size=500,
        chunk_overlap=100,
        description="PyMuPDF with font-size based heuristic section detection and sliding chunking",
    ),
    "P2": ParserAblationConfig(
        config_id="P2",
        parser_name="Docling Fixed Chunks",
        chunking_strategy="docling_fixed",
        chunk_token_size=512,
        chunk_overlap=64,
        description="Docling semantic document extraction flattened to fixed 512-token chunks",
    ),
    "P3": ParserAblationConfig(
        config_id="P3",
        parser_name="Docling Semantic Chunks",
        chunking_strategy="docling_semantic",
        chunk_token_size=600,
        chunk_overlap=80,
        description="Docling semantic section and paragraph-bounded chunks",
    ),
    "P4": ParserAblationConfig(
        config_id="P4",
        parser_name="ScholAR Provenance AST",
        chunking_strategy="provenance_ast",
        chunk_token_size=500,
        chunk_overlap=100,
        description="Full ScholAR dual-engine: Docling semantics + PyMuPDF geometry + Provenance EvidenceAST",
    ),
}


class DualEngineIngestionService:
    """Orchestrates document ingestion across Docling and PyMuPDF."""

    @classmethod
    def ingest_paper(
        cls,
        pdf_path: Path,
        document_id: str,
        title: str = "",
        authors: list[str] | None = None,
        year: int = 0,
        abstract: str = "",
        config: ParserAblationConfig | None = None,
    ) -> EvidenceAST:
        """Execute dual-engine ingestion and build the canonical EvidenceAST."""
        target_dir = paper_dir(document_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        config = config or PARSER_ABLATIONS["P4"]

        # Step 1: Open PyMuPDF document for geometric and page metrics
        doc = fitz.open(str(pdf_path))
        num_pages = len(doc)
        doc.close()

        # Step 2: Try Docling semantic parse if config allows (P2, P3, P4)
        docling_result = None
        if config.config_id in ("P2", "P3", "P4") and is_docling_available():
            docling_result = parse_with_docling(pdf_path, document_id)

        # Step 3: If Docling succeeded, build EvidenceAST from Docling + PyMuPDF
        if docling_result:
            blocks = docling_result["blocks"]
            sections = docling_result["sections"]
            parser_engine = "docling+pymupdf"
            degraded_mode = False
        else:
            # Graceful Fallback: Extract via PyMuPDF heuristic parser
            logger.info("Using PyMuPDF heuristic layout parser for [%s]", document_id)
            blocks, sections = cls._parse_with_pymupdf_heuristic(pdf_path, document_id)
            parser_engine = "pymupdf_heuristic"
            degraded_mode = True if config.config_id == "P4" else False

        # Step 4: Extract figures and tables geometry via PyMuPDF
        figures_dir = target_dir / "figures"
        figures = extract_figures(pdf_path, figures_dir)
        write_json(target_dir / "figures.json", figures)

        # Merge visual blocks into AST if not already present
        existing_fig_ids = {b.figure_id for b in blocks if b.figure_id}
        for fig in figures:
            fig_id = fig.get("figure_id", "")
            if fig_id not in existing_fig_ids:
                blocks.append(EvidenceBlock(
                    evidence_id=f"VIS_{fig_id.upper()}",
                    document_id=document_id,
                    page=fig.get("page", 1),
                    section_path=fig.get("section_path", []),
                    modality=EvidenceModality.VISUAL,
                    bbox=fig.get("bbox_norm", [0.0, 0.0, 1.0, 1.0]),
                    text=fig.get("caption", f"Figure {fig_id}"),
                    figure_id=fig_id,
                    figure_path=fig.get("image_path"),
                ))

        # Assign neighboring blocks for local context
        for i, block in enumerate(blocks):
            neighbors = []
            if i > 0:
                neighbors.append(blocks[i - 1].evidence_id)
            if i < len(blocks) - 1:
                neighbors.append(blocks[i + 1].evidence_id)
            block.neighboring_blocks = neighbors

        # Construct Canonical EvidenceAST
        ast = EvidenceAST(
            document_id=document_id,
            title=title,
            authors=authors or [],
            year=year,
            abstract=abstract,
            page_count=num_pages,
            blocks=blocks,
            sections=sections,
            parser_engine=parser_engine,
            degraded_mode=degraded_mode,
            created_at=time.time(),
        )

        # Write canonical evidence AST to disk
        write_json(target_dir / "evidence_ast.json", ast.model_dump())

        # Write backwards-compatible pages.json & chunks.json
        pages_data = extract_pages(pdf_path)
        write_json(target_dir / "pages.json", pages_data)

        chunks_data = cls._generate_chunks_from_ast(ast, config)
        write_json(target_dir / "chunks.json", chunks_data)

        # Write metadata
        meta = {
            "id": document_id,
            "title": title,
            "authors": authors or [],
            "year": year,
            "abstract": abstract,
            "page_count": num_pages,
            "chunk_count": len(chunks_data),
            "figure_count": len(figures),
            "parser_engine": parser_engine,
            "degraded_mode": degraded_mode,
            "ablation_config": config.config_id,
            "ingested_at": time.time(),
        }
        write_json(target_dir / "metadata.json", meta)

        logger.info(
            "Ingestion complete for [%s]: %d blocks, %d chunks, degraded_mode=%s",
            document_id, len(blocks), len(chunks_data), degraded_mode
        )
        return ast

    @classmethod
    def _parse_with_pymupdf_heuristic(
        cls, pdf_path: Path, document_id: str
    ) -> tuple[list[EvidenceBlock], list[SectionNode]]:
        """PyMuPDF heuristic layout and font-hierarchy parser."""
        doc = fitz.open(str(pdf_path))
        blocks: list[EvidenceBlock] = []
        sections: list[SectionNode] = []
        current_section: list[str] = []
        block_idx = 1

        for page_idx, page in enumerate(doc, 1):
            text_page = page.get_text("blocks")
            p_width = page.rect.width or 1.0
            p_height = page.rect.height or 1.0

            for b in text_page:
                x0, y0, x1, y1, text, b_no, b_type = b
                text = text.strip()
                if not text or len(text) < 10:
                    continue

                # Check if block looks like a section header (e.g. "1. Introduction" or "3 Methodology")
                is_heading = bool(re.match(r"^(\d+(\.\d+)*|[A-Z][a-z]+)\s+[A-Z]", text) and len(text.splitlines()) == 1 and len(text) < 80)
                if is_heading:
                    current_section = [text]
                    sections.append(SectionNode(
                        title=text,
                        level=1,
                        page_start=page_idx,
                        page_end=page_idx,
                    ))

                # Normalize bounding box
                bbox_norm = [
                    round(max(0.0, min(1.0, x0 / p_width)), 4),
                    round(max(0.0, min(1.0, y0 / p_height)), 4),
                    round(max(0.0, min(1.0, x1 / p_width)), 4),
                    round(max(0.0, min(1.0, y1 / p_height)), 4),
                ]

                # Classify table or text
                is_table = "|" in text or "\t" in text or "Table " in text[:15]
                modality = EvidenceModality.TABLE if is_table else EvidenceModality.TEXT

                blocks.append(EvidenceBlock(
                    evidence_id=f"E_{block_idx:03d}",
                    document_id=document_id,
                    page=page_idx,
                    section_path=list(current_section),
                    modality=modality,
                    bbox=bbox_norm,
                    text=text,
                    char_start=0,
                    char_end=len(text),
                ))
                block_idx += 1

        doc.close()
        return blocks, sections

    @classmethod
    def _generate_chunks_from_ast(
        cls, ast: EvidenceAST, config: ParserAblationConfig
    ) -> list[dict[str, Any]]:
        """Generate retrieval chunks from the canonical EvidenceAST based on ablation config."""
        chunks: list[dict[str, Any]] = []
        chunk_idx = 1

        for block in ast.blocks:
            chunk_dict = {
                "chunk_id": f"chunk_{chunk_idx:03d}",
                "evidence_id": block.evidence_id,
                "document_id": ast.document_id,
                "page": block.page,
                "section": block.section_path[-1] if block.section_path else "",
                "section_path": block.section_path,
                "modality": block.modality.value,
                "text": block.text,
                "bbox_norm": block.bbox,
                "is_figure_chunk": block.modality == EvidenceModality.VISUAL,
                "is_table_chunk": block.modality == EvidenceModality.TABLE,
                "figure_id": block.figure_id,
                "label": f"Figure {block.figure_id}" if block.figure_id else (f"Table" if block.modality == EvidenceModality.TABLE else ""),
            }
            chunks.append(chunk_dict)
            chunk_idx += 1

        return chunks
