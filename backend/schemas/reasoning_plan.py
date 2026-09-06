"""Explicit data contracts for multi-level reasoning plans, execution nodes, and answer drafts.

Follows ScholAR Reasoning Architecture:
- Explicit typed operations (no arbitrary shell or unverified prose)
- Node dependency DAG with validation
- Shared AnswerDraft contract for all synthesis and fallback paths
- Source-scoped evidence requirements and bundles
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator


class NodeOperation(str, Enum):
    """Finite allowlist of deterministic and bounded plan node operations."""
    RETRIEVE_TEXT = "retrieve_text"
    LOOKUP_STRUCTURE = "lookup_structure"
    RESOLVE_REFERENCE = "resolve_reference"
    INSPECT_VISUAL = "inspect_visual"
    SELECT_TABLE_CELLS = "select_table_cells"
    CALCULATE = "calculate"
    CHECK_COMPARABILITY = "check_comparability"
    VERIFY_CLAIM = "verify_claim"
    SYNTHESIZE = "synthesize"


class NodeStatus(str, Enum):
    """Execution and verification status for plan nodes."""
    PENDING = "pending"
    RUNNING = "running"
    SUPPORTED = "supported"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"
    CONFLICTING = "conflicting"
    ERROR = "error"
    SKIPPED = "skipped"


class EvidenceRequirement(BaseModel):
    """Specific evidence constraint or fact needed by a plan node."""
    requirement_id: str
    description: str = ""
    target_modality: str = "text"
    source_scope: list[str] = Field(default_factory=list)
    is_mandatory: bool = True


class EvidenceBundle(BaseModel):
    """Complete valid support bundle for a claim or deduction."""
    bundle_id: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_paper_ids: list[str] = Field(default_factory=list)
    modality: str = "text"
    rationale: str = ""


class PlanNode(BaseModel):
    """Executable step in a structured scientific reasoning plan."""
    node_id: str
    operation: NodeOperation
    depends_on: list[str] = Field(default_factory=list)
    input_bindings: dict[str, Any] = Field(default_factory=dict)
    source_scope: list[str] = Field(default_factory=list)
    target_modality: str = "text"
    required_output_schema: str = "dict"
    evidence_requirements: list[EvidenceRequirement] = Field(default_factory=list)
    mandatory: bool = True
    max_attempts: int = 2


class NodeResult(BaseModel):
    """Deterministic result and verification state of an executed plan node."""
    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    typed_value: Any = None
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 1
    elapsed_ms: float = 0.0


class ReasoningPlan(BaseModel):
    """Validated directed acyclic reasoning plan."""
    plan_id: str
    schema_version: str = "v1.0"
    query: str
    corpus_id: str = "default"
    requirements: list[EvidenceRequirement] = Field(default_factory=list)
    nodes: list[PlanNode] = Field(default_factory=list)
    final_requirements: list[str] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_plan_dag(self) -> ReasoningPlan:
        """Validate node IDs uniqueness and acyclicity."""
        node_ids = {n.node_id for n in self.nodes}
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node_id found in ReasoningPlan nodes")

        # Check dependencies exist
        for n in self.nodes:
            for dep in n.depends_on:
                if dep not in node_ids:
                    raise ValueError(f"Node {n.node_id} depends on unknown node {dep}")
                if dep == n.node_id:
                    raise ValueError(f"Node {n.node_id} cannot depend on itself")

        # Cycle check (topological sort)
        in_degree = {nid: 0 for nid in node_ids}
        adj: dict[str, list[str]] = {nid: [] for nid in node_ids}
        for n in self.nodes:
            for dep in n.depends_on:
                adj[dep].append(n.node_id)
                in_degree[n.node_id] += 1

        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        visited_count = 0
        while queue:
            curr = queue.pop(0)
            visited_count += 1
            for neighbor in adj[curr]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if visited_count != len(node_ids):
            raise ValueError("ReasoningPlan contains cyclic dependencies")
        return self


class ClaimDraft(BaseModel):
    """Atomic claim statement with exact source bindings in an answer draft."""
    claim_id: str
    text_span: str
    support_evidence_ids: list[str] = Field(default_factory=list)
    computed_result_ids: list[str] = Field(default_factory=list)
    is_supported: bool = True
    start_offset: int | None = None
    end_offset: int | None = None


class AnswerDraft(BaseModel):
    """Shared intermediate answer contract for all synthesis and fallback paths."""
    draft_id: str
    query: str
    text: str
    generation_mode: str = "model_backed"  # "model_backed", "extractive_fallback", "caption_fallback"
    claims: list[ClaimDraft] = Field(default_factory=list)
    support_ids: list[str] = Field(default_factory=list)
    computed_result_ids: list[str] = Field(default_factory=list)
    unresolved_requirements: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    is_fallback: bool = False
