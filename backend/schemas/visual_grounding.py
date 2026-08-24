"""Visual Grounding & Sub-Figure Anchor Schemas for ScholAR.

Defines:
- VisualAnchor: Sub-figure panel or diagram region with high-res crop coordinates
- VisualInspectionReport: VLM-based visual claim verification report
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class VisualAnchor(BaseModel):
    """Sub-figure panel or diagram region with normalized bounding box and rendered crop."""
    figure_id: str                          # e.g., "fig_001", "VIS_F2"
    panel_label: str = ""                   # e.g., "(a)", "(b)", "Left", "Right"
    bbox_norm: list[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    caption: str = ""
    image_crop_path: str = ""
    visual_summary: str = ""


class VisualInspectionReport(BaseModel):
    """Closed-loop visual verification report comparing generated claim against image crop."""
    anchor: VisualAnchor
    claim_text: str
    is_visually_grounded: bool = True
    trend_description: str = ""
    confidence: float = 0.95
    rationale: str = ""
