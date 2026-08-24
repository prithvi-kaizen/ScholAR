"""Evidence Graph & Reasoning Path Schemas for ScholAR.

Defines:
- EvidenceNode: Graph node representing an EvidenceBlock with its semantic role
- EvidenceEdge: Semantic directed relationship (supports_mechanism, ablation_evidence, explains_result)
- EvidenceGraph: Directed evidence graph representing multi-hop paper reasoning
- ReasoningPath: Ordered evidence chain (E1 -> E2 -> E3) for transparent explanation
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class EvidenceRelation(str, Enum):
    """Semantic relationship types between evidence blocks in an evidence graph."""
    SUPPORTS_MECHANISM = "supports_mechanism"      # E1 (Methodology) -> E2 (Architecture Details)
    ABLATION_EVIDENCE = "ablation_evidence"        # E1 (Methodology) -> E2 (Ablation Table/Text)
    EXPLAINS_RESULT = "explains_result"            # E2 (Ablation) -> E3 (Final Benchmark Result)
    CONTRADICTS_BASELINE = "contradicts_baseline"  # E1 (Proposed) -> E2 (Prior Art Failure)
    DEFINES_SYMBOL = "defines_symbol"              # E1 (Equation/Notation) -> E2 (Usage in Text)
    CROSS_MODAL_GROUNDING = "cross_modal_grounding" # E1 (Prose Claim) -> E2 (Table Cell / Figure Region)
    SECTION_SEQUENCE = "section_sequence"          # Linear reading order bridge


class EvidenceNode(BaseModel):
    """Node in the reasoning evidence graph."""
    node_id: str                              # Canonical evidence_id (e.g., "E_001", "E_TAB_02", "VIS_F1")
    document_id: str
    page: int
    section: str = ""
    modality: str = "text"
    text_preview: str = ""
    reasoning_role: str = "primary_evidence"  # method_definition, ablation_support, final_result, etc.
    score: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceEdge(BaseModel):
    """Directed semantic edge connecting two evidence nodes."""
    source_id: str
    target_id: str
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS_MECHANISM
    weight: float = 1.0
    description: str = ""


class EvidenceGraph(BaseModel):
    """Complete directed graph representing multi-level scientific reasoning structure."""
    nodes: list[EvidenceNode] = Field(default_factory=list)
    edges: list[EvidenceEdge] = Field(default_factory=list)
    graph_id: str = ""
    query: str = ""
    reasoning_level: str = "L1_DIRECT_LOOKUP"

    def get_node(self, node_id: str) -> EvidenceNode | None:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None


class ReasoningPathStep(BaseModel):
    """One step in an ordered evidence path (E1 -> E2 -> E3)."""
    step_index: int
    evidence_id: str
    section: str = ""
    page: int = 1
    modality: str = "text"
    role: str = "primary_evidence"
    claim_contribution: str = ""


class ReasoningPath(BaseModel):
    """Ordered multi-hop evidence path for synthesis and audit trail."""
    query: str
    reasoning_level: str
    steps: list[ReasoningPathStep] = Field(default_factory=list)
    graph: EvidenceGraph | None = None
    synthesized_rationale: str = ""
