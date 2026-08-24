"""Canonical schemas and data contracts for ScholAR."""

from backend.schemas.capabilities import CapabilityMode, ModelCapabilities, ModelRegistry
from backend.schemas.document import (
    BoundingBox,
    CoordinateSpace,
    CoordinateTransform,
    DocumentMetadata,
    EvidenceBlock,
    PageRender,
    ScientificDocument,
    SectionNode,
    TableBlock,
    TableCell,
    VisualEvidence,
    VisualRegion,
)

__all__ = [
    "CapabilityMode",
    "ModelCapabilities",
    "ModelRegistry",
    "BoundingBox",
    "CoordinateSpace",
    "CoordinateTransform",
    "DocumentMetadata",
    "SectionNode",
    "EvidenceBlock",
    "TableCell",
    "TableBlock",
    "VisualEvidence",
    "VisualRegion",
    "PageRender",
    "ScientificDocument",
]
