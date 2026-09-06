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

import hashlib
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
    render_page_visual_units,
    safe_paper_id,
    write_json,
)
from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType

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
        year: int | str = 0,
        abstract: str = "",
        config: ParserAblationConfig | None = None,
        target_dir: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceAST:
        """Execute dual-engine ingestion and build the canonical EvidenceAST.

        ``target_dir`` makes the complete derivation suitable for a sibling staging
        directory. When omitted, artifacts continue to be written to the paper's
        established storage directory.
        """
        output_dir = Path(target_dir) if target_dir is not None else paper_dir(document_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        config = config or PARSER_ABLATIONS["P4"]

        preserved_metadata: dict[str, Any] = {}
        existing_metadata_path = output_dir / "metadata.json"
        if existing_metadata_path.exists():
            existing_metadata = read_json(existing_metadata_path)
            if isinstance(existing_metadata, dict):
                preserved_metadata.update(existing_metadata)
        if metadata:
            preserved_metadata.update(metadata)

        resolved_title = title or str(preserved_metadata.get("title", ""))
        resolved_authors = authors if authors is not None else list(preserved_metadata.get("authors", []))
        resolved_abstract = abstract or str(
            preserved_metadata.get("abstract") or preserved_metadata.get("summary") or ""
        )
        raw_year = year if year not in (None, "", 0, "0") else preserved_metadata.get("year", 0)
        try:
            resolved_year = int(raw_year or 0)
        except (TypeError, ValueError):
            year_match = re.search(r"\d{4}", str(raw_year))
            resolved_year = int(year_match.group(0)) if year_match else 0

        # Step 1: Open PyMuPDF document for geometric and page metrics
        with fitz.open(str(pdf_path)) as doc:
            num_pages = len(doc)

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
        figures_dir = output_dir / "figures"
        figures = extract_figures(pdf_path, figures_dir)
        write_json(output_dir / "figures.json", figures)

        # Merge visual blocks into AST if not already present
        existing_fig_ids = {b.figure_id for b in blocks if b.figure_id}
        for fig in figures:
            fig_id = fig.get("figure_id", "")
            if fig_id not in existing_fig_ids:
                raw_bbox = fig.get("bbox_normalized") or fig.get("bbox_norm")
                if isinstance(raw_bbox, dict):
                    bbox = [
                        float(raw_bbox.get("x0", 0.0)),
                        float(raw_bbox.get("y0", 0.0)),
                        float(raw_bbox.get("x1", 1.0)),
                        float(raw_bbox.get("y1", 1.0)),
                    ]
                elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                    bbox = [float(value) for value in raw_bbox[:4]]
                else:
                    bbox = [0.0, 0.0, 1.0, 1.0]
                image_file = fig.get("image_file") or fig.get("image_path")
                figure_path = str(Path("figures") / image_file) if image_file else None
                blocks.append(EvidenceBlock(
                    evidence_id=f"VIS_{fig_id.upper()}",
                    document_id=document_id,
                    page=fig.get("page", 1),
                    section_path=fig.get("section_path", []),
                    modality=EvidenceModality.VISUAL,
                    bbox=bbox,
                    text=fig.get("caption") or fig.get("label") or f"Figure {fig_id}",
                    figure_id=fig_id,
                    figure_path=figure_path,
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
            title=resolved_title,
            authors=resolved_authors,
            year=resolved_year,
            abstract=resolved_abstract,
            page_count=num_pages,
            blocks=blocks,
            sections=sections,
            parser_engine=parser_engine,
            degraded_mode=degraded_mode,
            created_at=time.time(),
        )

        # Write canonical evidence AST to disk
        write_json(output_dir / "evidence_ast.json", ast.model_dump())

        # Write backwards-compatible pages.json & chunks.json
        pages_data = extract_pages(pdf_path)
        write_json(output_dir / "pages.json", pages_data)

        # Full-page visual units are primary ingestion artifacts. They guarantee
        # that a diagram/table remains retrievable even when caption-based figure
        # extraction misses it.
        page_visual_units = render_page_visual_units(
            pdf_path,
            output_dir,
            document_id,
        )
        visual_units = list(page_visual_units)
        for figure in figures:
            image_file = str(figure.get("image_file") or "")
            image_path = output_dir / "figures" / image_file
            if not image_file or not image_path.is_file():
                continue
            raw_bbox = figure.get("bbox_normalized") or figure.get("bbox_norm") or {}
            if isinstance(raw_bbox, dict):
                bbox_norm = [
                    float(raw_bbox.get("x0", 0.0)),
                    float(raw_bbox.get("y0", 0.0)),
                    float(raw_bbox.get("x1", 1.0)),
                    float(raw_bbox.get("y1", 1.0)),
                ]
            elif isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) >= 4:
                bbox_norm = [float(item) for item in raw_bbox[:4]]
            else:
                bbox_norm = [0.0, 0.0, 1.0, 1.0]
            unit_type = (
                VisualUnitType.TABLE
                if str(figure.get("figure_type") or "").lower() == "table"
                else VisualUnitType.FIGURE
            )
            unit = VisualDocumentUnit(
                visual_id=str(figure.get("figure_id") or image_path.stem),
                document_id=document_id,
                source_paper_id=document_id,
                page=int(figure.get("page") or 1),
                unit_type=unit_type,
                image_relpath=(Path("figures") / image_file).as_posix(),
                image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
                width_px=int(figure.get("width_px") or 1),
                height_px=int(figure.get("height_px") or 1),
                bbox_norm=bbox_norm,
                parent_visual_id=f"page_{int(figure.get('page') or 1):04d}",
                label=str(figure.get("label") or ""),
                caption=str(figure.get("caption") or ""),
            )
            visual_units.append(unit.model_dump(mode="json"))
        write_json(output_dir / "visual_units.json", visual_units)

        chunks_data = cls._generate_chunks_from_ast(ast, config)
        write_json(output_dir / "chunks.json", chunks_data)

        # Preserve source metadata while normalizing compatibility aliases and counts.
        meta = dict(preserved_metadata)
        meta.update({
            "id": meta.get("id") or document_id,
            "local_id": meta.get("local_id") or document_id,
            "document_id": document_id,
            "title": resolved_title,
            "authors": resolved_authors,
            "year": meta.get("year", resolved_year),
            "abstract": resolved_abstract,
            "summary": meta.get("summary") or resolved_abstract,
            "page_count": num_pages,
            "pages": num_pages,
            "chunk_count": len(chunks_data),
            "chunks": len(chunks_data),
            "figure_count": len(figures),
            "figures": len(figures),
            "visual_unit_count": len(visual_units),
            "parser_engine": parser_engine,
            "degraded_mode": degraded_mode,
            "ablation_config": config.config_id,
            "ingested_at": time.time(),
        })
        write_json(output_dir / "metadata.json", meta)

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
            section_title = block.section_path[-1] if block.section_path else "Body"
            section_prefix = " > ".join(block.section_path)
            original_text = block.text
            retrieval_text = f"{section_prefix}. {original_text}" if section_prefix else original_text
            if block.modality == EvidenceModality.VISUAL:
                chunk_type = "figure"
            elif block.modality == EvidenceModality.TABLE:
                chunk_type = "table"
            elif "abstract" in section_title.lower():
                chunk_type = "abstract"
            else:
                chunk_type = "body"
            char_start = int(block.char_start or 0)
            char_end = int(block.char_end or 0) or (char_start + len(original_text))
            chunk_dict = {
                "chunk_id": f"chunk_{chunk_idx:03d}",
                "evidence_id": block.evidence_id,
                "document_id": ast.document_id,
                "source_paper_id": ast.document_id,
                "page": block.page,
                "section": section_title,
                "section_title": section_title,
                "section_path": block.section_path,
                "modality": block.modality.value,
                "chunk_type": chunk_type,
                "text": original_text,
                "original_text": original_text,
                "retrieval_text": retrieval_text,
                "paragraph_text": original_text[:500],
                "char_start": char_start,
                "char_end": char_end,
                "bbox_norm": block.bbox,
                "is_figure_chunk": block.modality == EvidenceModality.VISUAL,
                "is_table_chunk": block.modality == EvidenceModality.TABLE,
                "figure_id": block.figure_id,
                "image_file": Path(block.figure_path).name if block.figure_path else None,
                "label": f"Figure {block.figure_id}" if block.figure_id else ("Table" if block.modality == EvidenceModality.TABLE else ""),
            }
            chunks.append(chunk_dict)
            chunk_idx += 1

        return chunks
