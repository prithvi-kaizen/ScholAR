from __future__ import annotations

import math
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator


CoordinateSpace = Literal["pdf_points", "normalized_page", "render_pixels"]


class BoundingBox(BaseModel):
    """Immutable bounding box with explicit coordinate space."""
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: float | None = None
    page_height: float | None = None
    coordinate_space: CoordinateSpace = "normalized_page"

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> BoundingBox:
        # Guarantee min/max ordering
        min_x = min(self.x0, self.x1)
        max_x = max(self.x0, self.x1)
        min_y = min(self.y0, self.y1)
        max_y = max(self.y0, self.y1)
        self.x0 = round(min_x, 4)
        self.x1 = round(max_x, 4)
        self.y0 = round(min_y, 4)
        self.y1 = round(max_y, 4)
        return self

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def is_valid(self, min_area: float = 1e-6) -> bool:
        return not (math.isnan(self.x0) or math.isnan(self.y0) or math.isnan(self.x1) or math.isnan(self.y1)) and self.area >= min_area

    def clip_to_bounds(self, max_x: float = 1.0, max_y: float = 1.0, min_x: float = 0.0, min_y: float = 0.0) -> BoundingBox:
        """Clip coordinates to prevent floating point overflow."""
        cx0 = max(min_x, min(self.x0, max_x))
        cy0 = max(min_y, min(self.y0, max_y))
        cx1 = max(min_x, min(self.x1, max_x))
        cy1 = max(min_y, min(self.y1, max_y))
        return BoundingBox(
            x0=cx0,
            y0=cy0,
            x1=cx1,
            y1=cy1,
            page_width=self.page_width,
            page_height=self.page_height,
            coordinate_space=self.coordinate_space,
        )

    def to_normalized(self, page_width: float, page_height: float) -> BoundingBox:
        if self.coordinate_space == "normalized_page":
            return self
        if page_width <= 0 or page_height <= 0:
            raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")
        return BoundingBox(
            x0=self.x0 / page_width,
            y0=self.y0 / page_height,
            x1=self.x1 / page_width,
            y1=self.y1 / page_height,
            page_width=page_width,
            page_height=page_height,
            coordinate_space="normalized_page",
        )

    def to_pdf_points(self, page_width: float, page_height: float) -> BoundingBox:
        if self.coordinate_space == "pdf_points":
            return self
        if page_width <= 0 or page_height <= 0:
            raise ValueError(f"Invalid page dimensions: {page_width}x{page_height}")
        return BoundingBox(
            x0=self.x0 * page_width,
            y0=self.y0 * page_height,
            x1=self.x1 * page_width,
            y1=self.y1 * page_height,
            page_width=page_width,
            page_height=page_height,
            coordinate_space="pdf_points",
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union (IoU) between two bounding boxes."""
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)

        inter_w = max(0.0, inter_x1 - inter_x0)
        inter_h = max(0.0, inter_y1 - inter_y0)
        inter_area = inter_w * inter_h

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def contains(self, other: BoundingBox, tolerance: float = 0.01) -> bool:
        """Check if this bounding box contains another box within a tolerance margin."""
        return (
            (self.x0 - tolerance) <= other.x0
            and (self.y0 - tolerance) <= other.y0
            and (self.x1 + tolerance) >= other.x1
            and (self.y1 + tolerance) >= other.y1
        )

    def as_list(self) -> list[float]:
        return [self.x0, self.y0, self.x1, self.y1]


class CoordinateTransform(BaseModel):
    """6-value affine transformation matrix: [a, b, c, d, e, f]
    where: x' = a*x + c*y + e
           y' = b*x + d*y + f
    """
    source_space: str
    target_space: str
    matrix: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    def apply(self, x: float, y: float) -> tuple[float, float]:
        a, b, c, d, e, f = self.matrix
        x_prime = a * x + c * y + e
        y_prime = b * x + d * y + f
        return round(x_prime, 4), round(y_prime, 4)

    def apply_box(self, box: BoundingBox, target_space: CoordinateSpace) -> BoundingBox:
        x0_p, y0_p = self.apply(box.x0, box.y0)
        x1_p, y1_p = self.apply(box.x1, box.y1)
        return BoundingBox(
            x0=x0_p,
            y0=y0_p,
            x1=x1_p,
            y1=y1_p,
            page_width=box.page_width,
            page_height=box.page_height,
            coordinate_space=target_space,
        )

    def inverse(self) -> CoordinateTransform:
        a, b, c, d, e, f = self.matrix
        det = a * d - b * c
        if abs(det) < 1e-9:
            raise ValueError("Matrix is singular; inverse transform undefined.")
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det
        inv_e = (c * f - d * e) / det
        inv_f = (b * e - a * f) / det
        return CoordinateTransform(
            source_space=self.target_space,
            target_space=self.source_space,
            matrix=[inv_a, inv_b, inv_c, inv_d, inv_e, inv_f],
        )


class DocumentMetadata(BaseModel):
    document_id: str
    source_sha256: str
    filename: str
    page_count: int
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    parser_name: str = "pymupdf"
    parser_version: str = "1.25.0"
    render_dpi: int = 160


class SectionNode(BaseModel):
    section_id: str
    title: str
    level: int = 1
    section_path: list[str] = Field(default_factory=list)
    parent_section_id: str | None = None
    child_section_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceBlock(BaseModel):
    evidence_id: str
    document_id: str
    block_type: Literal[
        "paragraph", "list_item", "caption", "equation_text",
        "table", "table_row", "figure_reference", "other"
    ] = "paragraph"
    original_text: str
    retrieval_text: str
    page: int
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    ordinal: int = 0
    previous_evidence_id: str | None = None
    next_evidence_id: str | None = None
    source_element_ref: str | None = None
    source_paper_id: str | None = None


class TableCell(BaseModel):
    row_index: int
    column_index: int
    row_span: int = 1
    column_span: int = 1
    text: str
    bbox: BoundingBox | None = None


class TableBlock(BaseModel):
    table_id: str
    evidence_id: str
    document_id: str
    page: int
    section_path: list[str] = Field(default_factory=list)
    caption: str | None = None
    headers: list[str] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    markdown: str = ""
    bbox: BoundingBox | None = None
    image_evidence_id: str | None = None


class VisualEvidence(BaseModel):
    evidence_id: str
    document_id: str
    visual_type: Literal[
        "figure", "chart", "diagram", "table_image", "equation_image",
        "page_region", "full_page"
    ] = "figure"
    page: int
    section_id: str | None = None
    section_path: list[str] = Field(default_factory=list)
    bbox_pdf: BoundingBox | None = None
    bbox_normalized: BoundingBox
    image_path: str
    image_sha256: str
    pixel_width: int
    pixel_height: int
    figure_label: str | None = None
    caption: str | None = None
    nearby_text_ids: list[str] = Field(default_factory=list)
    ocr_text: str | None = None
    parent_visual_id: str | None = None
    region_role: str | None = None


class VisualRegion(BaseModel):
    region_id: str
    parent_evidence_id: str
    document_id: str
    page: int
    role: Literal[
        "legend", "plot", "axis", "label", "bar", "curve", "cell_group",
        "diagram_node", "diagram_edge", "other"
    ] | None = None
    bbox_parent_normalized: BoundingBox
    bbox_page_normalized: BoundingBox
    bbox_pdf: BoundingBox | None = None
    image_path: str = ""
    image_sha256: str = ""
    proposal_source: Literal["deterministic", "parser", "vlm"] = "vlm"
    proposer_model_id: str | None = None
    proposer_revision: str | None = None
    verification: Literal["SUPPORTED", "UNCERTAIN", "CONTRADICTED"] = "UNCERTAIN"


class PageRender(BaseModel):
    page: int
    image_path: str
    image_sha256: str
    pixel_width: int
    pixel_height: int
    pdf_width_points: float
    pdf_height_points: float
    pdf_to_render: CoordinateTransform
    render_to_pdf: CoordinateTransform


class ScientificDocument(BaseModel):
    metadata: DocumentMetadata
    sections: list[SectionNode] = Field(default_factory=list)
    evidence_blocks: list[EvidenceBlock] = Field(default_factory=list)
    tables: list[TableBlock] = Field(default_factory=list)
    visual_evidence: list[VisualEvidence] = Field(default_factory=list)
    page_renders: list[PageRender] = Field(default_factory=list)
