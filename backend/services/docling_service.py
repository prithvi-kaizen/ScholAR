"""Docling Semantic Document Parsing Service.

Parses scientific PDFs into structured semantic hierarchies:
- Reading order
- Section / subsection tree
- Paragraphs & inline typography
- Structured tables with cell grids
- Figure associations and captions

Provides graceful fallback detection if Docling is not installed or encounters an unparseable PDF layout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from backend.schemas.evidence import (
    EvidenceBlock,
    EvidenceModality,
    SectionNode,
    TableCell,
    TableData,
)

logger = logging.getLogger("scholar.docling")

_DOCLING_AVAILABLE: bool | None = None


def is_docling_available() -> bool:
    """Check whether Docling is installed and importable."""
    global _DOCLING_AVAILABLE
    if _DOCLING_AVAILABLE is None:
        try:
            import docling  # noqa: F401
            from docling.document_converter import DocumentConverter  # noqa: F401
            _DOCLING_AVAILABLE = True
            logger.info("Docling document converter is available.")
        except (ImportError, Exception) as exc:
            _DOCLING_AVAILABLE = False
            logger.info("Docling is not available (falling back to PyMuPDF): %s", exc)
    return _DOCLING_AVAILABLE


def parse_with_docling(pdf_path: Path, document_id: str) -> dict[str, Any] | None:
    """Parse a scientific PDF using Docling DocumentConverter.

    Returns a structured dict with parsed sections, text blocks, and tables,
    or None if parsing fails / Docling is unavailable.
    """
    if not is_docling_available():
        return None

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        doc = result.document

        parsed_blocks: list[EvidenceBlock] = []
        parsed_sections: list[SectionNode] = []
        current_section_path: list[str] = []

        block_idx = 1
        table_idx = 1
        fig_idx = 1

        # Iterate over Docling body items
        for item, level in doc.iterate_items():
            item_type = getattr(item, "label", type(item).__name__)
            page_no = getattr(item, "page_no", 1) or 1

            # Extract bounding box if present and normalize
            bbox_norm = [0.0, 0.0, 1.0, 1.0]
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                p0 = prov[0]
                page_no = getattr(p0, "page_no", page_no) or page_no
                raw_bbox = getattr(p0, "bbox", None)
                if raw_bbox:
                    # Get page dimensions from Docling doc
                    p_obj = doc.pages.get(page_no) if hasattr(doc, "pages") and isinstance(doc.pages, dict) else None
                    p_size = getattr(p_obj, "size", None)
                    p_w = float(getattr(p_size, "width", 600.0) or 600.0)
                    p_h = float(getattr(p_size, "height", 800.0) or 800.0)

                    # Docling BoundingBox: l, t, r, b or x0, y0, x1, y1
                    l = getattr(raw_bbox, "l", getattr(raw_bbox, "x0", 0.0))
                    t = getattr(raw_bbox, "t", getattr(raw_bbox, "y0", 0.0))
                    r = getattr(raw_bbox, "r", getattr(raw_bbox, "x1", p_w))
                    b = getattr(raw_bbox, "b", getattr(raw_bbox, "y1", p_h))
                    
                    bbox_norm = [
                        round(max(0.0, min(1.0, float(l) / p_w)), 4),
                        round(max(0.0, min(1.0, float(t) / p_h)), 4),
                        round(max(0.0, min(1.0, float(r) / p_w)), 4),
                        round(max(0.0, min(1.0, float(b) / p_h)), 4),
                    ]

            # 1. Section Header
            if item_type in ("section_header", "title", "heading"):
                header_text = getattr(item, "text", "").strip()
                if header_text:
                    if level <= 1:
                        current_section_path = [header_text]
                    elif len(current_section_path) >= level:
                        current_section_path = current_section_path[:level - 1] + [header_text]
                    else:
                        current_section_path.append(header_text)

                    parsed_sections.append(SectionNode(
                        title=header_text,
                        level=level or 1,
                        page_start=page_no,
                        page_end=page_no,
                    ))

            # 2. Table Item
            elif item_type in ("table", "TableItem"):
                table_md = getattr(item, "export_to_markdown", lambda: "")() or getattr(item, "text", "")
                cells: list[TableCell] = []
                headers: list[str] = []
                rows_data: list[list[str]] = []

                # Extract cell data if available
                table_grid = getattr(item, "data", None)
                if table_grid and hasattr(table_grid, "grid"):
                    for r_idx, row in enumerate(table_grid.grid):
                        row_vals = []
                        for c_idx, cell in enumerate(row):
                            val = getattr(cell, "text", str(cell)).strip()
                            row_vals.append(val)
                            cells.append(TableCell(
                                row=r_idx,
                                col=c_idx,
                                value=val,
                                is_header=r_idx == 0,
                            ))
                        if r_idx == 0:
                            headers = row_vals
                        else:
                            rows_data.append(row_vals)

                caption = getattr(item, "caption", "") or f"Table {table_idx}"
                t_data = TableData(
                    table_id=f"tab_{table_idx:03d}",
                    headers=headers,
                    rows=rows_data,
                    cells=cells,
                    caption=str(caption),
                    raw_markdown=table_md,
                    num_rows=len(rows_data) + (1 if headers else 0),
                    num_cols=len(headers) if headers else (len(rows_data[0]) if rows_data else 0),
                )

                parsed_blocks.append(EvidenceBlock(
                    evidence_id=f"E_TAB_{table_idx:02d}",
                    document_id=document_id,
                    page=page_no,
                    section_path=list(current_section_path),
                    modality=EvidenceModality.TABLE,
                    bbox=bbox_norm,
                    text=table_md or str(caption),
                    table_data=t_data,
                ))
                table_idx += 1

            # 3. Figure Item
            elif item_type in ("picture", "figure", "PictureItem"):
                caption = getattr(item, "caption", "") or f"Figure {fig_idx}"
                parsed_blocks.append(EvidenceBlock(
                    evidence_id=f"VIS_F{fig_idx}",
                    document_id=document_id,
                    page=page_no,
                    section_path=list(current_section_path),
                    modality=EvidenceModality.VISUAL,
                    bbox=bbox_norm,
                    text=str(caption),
                    figure_id=f"fig_{fig_idx:03d}",
                ))
                fig_idx += 1

            # 4. Paragraph / Text Item
            else:
                text_content = getattr(item, "text", "").strip()
                if text_content and len(text_content) > 10:
                    parsed_blocks.append(EvidenceBlock(
                        evidence_id=f"E_{block_idx:03d}",
                        document_id=document_id,
                        page=page_no,
                        section_path=list(current_section_path),
                        modality=EvidenceModality.TEXT,
                        bbox=bbox_norm,
                        text=text_content,
                    ))
                    block_idx += 1

        logger.info(
            "Docling parsed [%s]: %d blocks, %d sections",
            document_id, len(parsed_blocks), len(parsed_sections)
        )
        return {
            "blocks": parsed_blocks,
            "sections": parsed_sections,
            "parser_engine": "docling+pymupdf",
            "degraded_mode": False,
        }

    except Exception as exc:
        logger.warning("Docling parse failed for [%s]: %s (Falling back to PyMuPDF)", document_id, exc)
        return None
