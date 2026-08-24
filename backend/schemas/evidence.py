"""Canonical Evidence AST Data Model for ScholAR.

Defines the core software-owned provenance structures:
- EvidenceBlock: atomic multimodality evidence item (text, table, visual)
- EvidenceAST: complete hierarchical document representation
- TableData & TableCell: structured table representation
- ParserAblationConfig: evaluation matrix across parser/chunking configurations
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class EvidenceModality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    VISUAL = "visual"


class TableCell(BaseModel):
    """Atomic cell inside a structured table."""
    row: int
    col: int
    value: str
    is_header: bool = False
    row_span: int = 1
    col_span: int = 1


class TableData(BaseModel):
    """Structured table representation with headers, cell grid, and raw Markdown."""
    table_id: str
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    caption: str = ""
    raw_markdown: str = ""
    num_rows: int = 0
    num_cols: int = 0


class EvidenceBlock(BaseModel):
    """Canonical atomic unit of inspectable evidence across text, tables, and visuals."""
    evidence_id: str                          # e.g., "E_SEC4_P7_T01" or "VIS_F4_R01"
    document_id: str                          # Local paper ID / PDF SHA
    page: int                                 # 1-indexed page number
    section_path: list[str] = Field(default_factory=list)  # ["4 Experiments", "4.2 Ablation Study"]
    modality: EvidenceModality = EvidenceModality.TEXT
    bbox: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])  # [x0, y0, x1, y1] normalized to [0, 1]
    text: str = ""                            # Block text content or table Markdown
    table_data: TableData | None = None       # Populated if modality == TABLE
    figure_id: str | None = None              # Populated if modality == VISUAL
    figure_path: str | None = None            # Path to rendered figure PNG if visual
    parent_id: str | None = None              # Hierarchical parent section/figure block ID
    neighboring_blocks: list[str] = Field(default_factory=list)
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SectionNode(BaseModel):
    """Hierarchical section node in document outline."""
    title: str
    level: int                                # 1 for main section, 2 for subsection, etc.
    page_start: int
    page_end: int
    block_ids: list[str] = Field(default_factory=list)
    subsections: list[SectionNode] = Field(default_factory=list)


class EvidenceAST(BaseModel):
    """Complete canonical document AST containing all evidence blocks and hierarchy."""
    document_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int = 0
    abstract: str = ""
    page_count: int = 0
    blocks: list[EvidenceBlock] = Field(default_factory=list)
    sections: list[SectionNode] = Field(default_factory=list)
    parser_engine: str = "docling+pymupdf"    # e.g. "docling+pymupdf", "pymupdf_heuristic", "pymupdf_fixed"
    degraded_mode: bool = False               # True if fallback parser was engaged
    created_at: float = Field(default_factory=time.time)

    def get_block(self, evidence_id: str) -> EvidenceBlock | None:
        """Lookup an evidence block by its canonical ID."""
        for b in self.blocks:
            if b.evidence_id == evidence_id:
                return b
        return None

    def get_blocks_for_page(self, page: int) -> list[EvidenceBlock]:
        """Get all evidence blocks located on a specific page."""
        return [b for b in self.blocks if b.page == page]

    def get_text_blocks(self) -> list[EvidenceBlock]:
        return [b for b in self.blocks if b.modality == EvidenceModality.TEXT]

    def get_table_blocks(self) -> list[EvidenceBlock]:
        return [b for b in self.blocks if b.modality == EvidenceModality.TABLE]

    def get_visual_blocks(self) -> list[EvidenceBlock]:
        return [b for b in self.blocks if b.modality == EvidenceModality.VISUAL]


class ParserAblationConfig(BaseModel):
    """Parser ablation configuration for empirical comparison."""
    config_id: Literal["P0", "P1", "P2", "P3", "P4"]
    parser_name: str
    chunking_strategy: Literal["fixed_token", "heuristic_ast", "docling_fixed", "docling_semantic", "provenance_ast"]
    chunk_token_size: int = 512
    chunk_overlap: int = 64
    description: str
