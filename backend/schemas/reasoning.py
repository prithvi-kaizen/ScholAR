"""Multi-Level Reasoning Taxonomy & Question Analysis Schemas for ScholAR.

Defines:
- ReasoningLevel (L1 Direct Lookup to L5 Multi-Hop Synthesis)
- TargetModality (TEXT, TABLE, FIGURE, MULTIMODAL)
- SubQuery (bounded atomic subquery)
- QuestionAnalysis (structured query decomposition and planning)
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class ReasoningLevel(str, Enum):
    """5-Level Scientific Reasoning Taxonomy."""
    L1_DIRECT_LOOKUP = "L1_DIRECT_LOOKUP"            # Fact / hyperparameter / single value
    L2_SAME_SECTION = "L2_SAME_SECTION"              # Explanation within a single section
    L3_CROSS_SECTION = "L3_CROSS_SECTION"            # Joint reasoning across distinct sections (e.g. Method <-> Results)
    L4_CROSS_MODAL = "L4_CROSS_MODAL"                # Reasoning across Text <-> Table or Text <-> Figure
    L5_MULTI_HOP_SYNTHESIS = "L5_MULTI_HOP_SYNTHESIS"  # Complex comparative synthesis, causal chains, multi-factor analysis


class TargetModality(str, Enum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    MULTIMODAL = "multimodal"


class SubQuery(BaseModel):
    """Atomic subquery generated during bounded multi-hop decomposition."""
    subquery_id: str                          # e.g., "SQ1", "SQ2"
    query_text: str
    target_sections: list[str] = Field(default_factory=list)
    target_modality: TargetModality = TargetModality.TEXT
    priority: int = 1                         # 1 (high), 2 (medium), 3 (low)


class QuestionAnalysis(BaseModel):
    """Structured question analysis and execution plan."""
    original_query: str
    reasoning_level: ReasoningLevel = ReasoningLevel.L1_DIRECT_LOOKUP
    target_modalities: list[TargetModality] = Field(default_factory=lambda: [TargetModality.TEXT])
    requires_arithmetic: bool = False         # True if question asks for delta, %, ratio, etc.
    requires_visual: bool = False             # True if question asks for figure/plot/diagram inspection
    target_sections: list[str] = Field(default_factory=list)
    subqueries: list[SubQuery] = Field(default_factory=list)  # Max 3 atomic subqueries
    confidence: float = 1.0
    rationale: str = ""
