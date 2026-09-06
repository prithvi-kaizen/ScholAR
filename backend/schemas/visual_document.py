"""Canonical persisted visual-document units used by page and crop retrieval."""

from __future__ import annotations

from enum import Enum
from pathlib import PurePosixPath

from pydantic import BaseModel, Field, field_validator


class VisualUnitType(str, Enum):
    PAGE = "page"
    FIGURE = "figure"
    TABLE = "table"


class VisualDocumentUnit(BaseModel):
    """One immutable, source-scoped image that can be retrieved and inspected."""

    visual_id: str
    document_id: str
    source_paper_id: str
    page: int = Field(ge=1)
    unit_type: VisualUnitType
    image_relpath: str
    image_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    bbox_norm: list[float] = Field(min_length=4, max_length=4)
    parent_visual_id: str | None = None
    label: str = ""
    caption: str = ""

    @field_validator("image_relpath")
    @classmethod
    def validate_image_relpath(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("image_relpath must be a safe paper-relative path")
        return path.as_posix()

    @field_validator("bbox_norm")
    @classmethod
    def validate_bbox(cls, value: list[float]) -> list[float]:
        x0, y0, x1, y1 = (float(item) for item in value)
        if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
            raise ValueError("bbox_norm values must lie in [0, 1]")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox_norm must have positive area")
        return [x0, y0, x1, y1]
