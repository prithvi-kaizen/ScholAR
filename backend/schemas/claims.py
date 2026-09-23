"""Auditable claim-verification and selective-repair contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EntailmentStatus(str, Enum):
    """Canonical claim-support vocabulary, including unresolved pixel evidence."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIED_VISUAL = "UNVERIFIED_VISUAL"


class EvidenceOrigin(str, Enum):
    """Whether evidence is a source artifact or a model/application derivative."""

    SOURCE_TEXT = "SOURCE_TEXT"
    SOURCE_TABLE = "SOURCE_TABLE"
    SOURCE_PIXELS = "SOURCE_PIXELS"
    MODEL_VISUAL_OBSERVATION = "MODEL_VISUAL_OBSERVATION"
    APPLICATION_IMPUTED = "APPLICATION_IMPUTED"


class RepairAction(str, Enum):
    """Conservative, evidence-preserving repair actions."""

    NONE = "none"
    CITATION_REMAP = "citation_remap"
    CLAIM_NARROWING = "claim_narrowing"
    NUMERIC_CORRECTION = "numeric_correction"
    CLAIM_DELETION = "claim_deletion"
    ABSTAIN = "abstain"


class CitationSpan(BaseModel):
    """A zero-based, half-open citation-marker span in the containing answer."""

    start: int
    end: int
    marker: str
    reference_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class EvidenceProvenance(BaseModel):
    """Source-scoped evidence resolved for one atomic claim."""

    evidence_id: str
    ref_id: int | None = None
    source_paper_id: str | None = None
    document_id: str | None = None
    page: int | None = None
    region: dict[str, Any] | list[float] | None = None
    origin: EvidenceOrigin = EvidenceOrigin.SOURCE_TEXT
    derived_artifact_ids: list[str] = Field(default_factory=list)


class DerivedEvidenceArtifact(BaseModel):
    """Model/application output linked to source evidence but never replacing it."""

    artifact_id: str = Field(min_length=1)
    origin: EvidenceOrigin
    derived_from_evidence_id: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def require_derived_origin(self) -> "DerivedEvidenceArtifact":
        if self.origin not in {
            EvidenceOrigin.MODEL_VISUAL_OBSERVATION,
            EvidenceOrigin.APPLICATION_IMPUTED,
        }:
            raise ValueError("derived evidence must have a model/application origin")
        return self

    @classmethod
    def model_observation(
        cls,
        *,
        artifact_id: str,
        derived_from_evidence_id: str,
        content_sha256: str,
        model_id: str | None,
    ) -> "DerivedEvidenceArtifact":
        return cls(
            artifact_id=artifact_id,
            origin=EvidenceOrigin.MODEL_VISUAL_OBSERVATION,
            derived_from_evidence_id=derived_from_evidence_id,
            content_sha256=content_sha256,
            model_id=model_id,
        )


class ClaimRepairRecord(BaseModel):
    """One real text edit, addressed against the pre-repair answer."""

    claim_id: str
    action: RepairAction
    original_start: int
    original_end: int
    original_text: str
    replacement_text: str
    initial_status: EntailmentStatus
    second_pass_status: EntailmentStatus | None = None
    original_evidence_ids: list[str] = Field(default_factory=list)
    resolved_evidence_ids: list[str] = Field(default_factory=list)
    remap_attempted: bool = False


class SupportScorerMetadata(BaseModel):
    """Traceable scorer identity; thresholds are not claimed calibrated without labels."""

    backend: str = "lexical-overlap"
    version: str = "lexical-support-v2"
    thresholds_calibrated: bool = False
    supported_threshold: float = 0.50
    partial_threshold: float = 0.25
    threshold_profile_id: str | None = None


class AtomicClaim(BaseModel):
    """Atomic scientific statement with exact answer and citation spans."""

    claim_id: str
    text: str
    cited_evidence_ids: list[str] = Field(default_factory=list)
    entailment_status: EntailmentStatus = EntailmentStatus.SUPPORTED
    confidence_score: float = 1.0
    rationale: str = ""
    repaired_text: str | None = None
    repair_action: RepairAction = RepairAction.NONE

    # Offsets are zero-based, half-open Unicode-character offsets. For a report
    # over answer A, A[start:end] is exactly text.
    start: int | None = None
    end: int | None = None
    citation_spans: list[CitationSpan] = Field(default_factory=list)
    normalized_text: str = ""
    claim_type: str = "factual"
    resolved_evidence: list[EvidenceProvenance] = Field(default_factory=list)
    first_pass_status: EntailmentStatus | None = None
    second_pass_status: EntailmentStatus | None = None
    final_start: int | None = None
    final_end: int | None = None
    scorer: SupportScorerMetadata = Field(default_factory=SupportScorerMetadata)
    requires_human_pixel_judgment: bool = False


class VerificationReport(BaseModel):
    """Aggregated claim support for one exact answer string."""

    claims: list[AtomicClaim] = Field(default_factory=list)
    overall_supported: bool = True
    supported_count: int = 0
    partial_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    unverified_visual_count: int = 0
    has_abstained: bool = False
    abstention_reason: str | None = None
    final_verified_response: str = ""
    edits: list[ClaimRepairRecord] = Field(default_factory=list)
    second_pass_completed: bool = False
    scorer: SupportScorerMetadata = Field(default_factory=SupportScorerMetadata)
