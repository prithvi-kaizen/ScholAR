"""Versioned, auditable execution contract for the ScholAR answer pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from backend.schemas.capabilities import CapabilityMode, EvidenceBudget, ModelCapabilities
from backend.schemas.claims import ClaimRepairRecord, VerificationReport
from backend.schemas.evidence_graph import EvidenceGraph, ReasoningPathStep
from backend.schemas.numeric_plan import NumericExecutionResult, NumericPlan
from backend.schemas.reasoning import QuestionAnalysis, SubQuery


ANSWER_TRACE_SCHEMA_VERSION = "1.0"
ANSWER_PIPELINE_VERSION = "answer-pipeline-v1"
ANSWER_PROMPT_VERSION = "grounded-answer-v1"
RETRIEVER_VERSION = "hybrid-document-visual-rrf-v3"
VERIFIER_VERSION = "lexical-claim-verifier-v2"


class ExecutionPolicy(str, Enum):
    """Controls whether a measured run may silently degrade."""

    ALLOW_EXTRACTIVE_FALLBACK = "ALLOW_EXTRACTIVE_FALLBACK"
    REQUIRE_LOCAL_MODEL = "REQUIRE_LOCAL_MODEL"


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


class GenerationMode(str, Enum):
    LOCAL_MODEL = "LOCAL_MODEL"
    VISION_MODEL = "VISION_MODEL"
    EXTRACTIVE_FALLBACK = "EXTRACTIVE_FALLBACK"
    NO_GENERATION = "NO_GENERATION"


class CitationOrigin(str, Enum):
    MODEL_EMITTED = "MODEL_EMITTED"
    APPLICATION_IMPUTED = "APPLICATION_IMPUTED"
    REMAPPED = "REMAPPED"
    VISION_SERVICE = "VISION_SERVICE"
    EXTRACTIVE_SERVICE = "EXTRACTIVE_SERVICE"


class RepairMode(str, Enum):
    """Frozen post-generation intervention applied to an answer."""

    NONE = "NONE"
    CITATION_REMAP_ONLY = "CITATION_REMAP_ONLY"
    SELECTIVE = "SELECTIVE"


class InterventionControls(BaseModel):
    repair_mode: RepairMode = RepairMode.SELECTIVE
    abstain_on_no_supported_claims: bool = True

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def validate_combination(self) -> "InterventionControls":
        if self.repair_mode != RepairMode.SELECTIVE and self.abstain_on_no_supported_claims:
            raise ValueError(
                "abstain_on_no_supported_claims is only valid for SELECTIVE repair"
            )
        return self


class DecodingOptions(BaseModel):
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    num_ctx: int = Field(default=16000, ge=256)
    num_predict: int = Field(default=1650, ge=1)

    model_config = ConfigDict(extra="forbid")


class EvaluationContext(BaseModel):
    release_id: str
    run_id: str
    system_name: str
    case_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    prompt_hashes: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    @model_serializer(mode="wrap")
    def omit_unset_corpus_hash(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if self.corpus_sha256 is None:
            payload.pop("corpus_sha256", None)
        return payload


class AnswerPipelineRequest(BaseModel):
    paper_id: str
    query: str
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    secondary_paper_ids: list[str] = Field(default_factory=list, max_length=25)
    requested_model: str | None = None
    generation_seed: int | None = Field(default=None, ge=0, le=2147483647)
    capability_mode: CapabilityMode = CapabilityMode.AUTO
    execution_policy: ExecutionPolicy = ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK
    intervention: InterventionControls = Field(default_factory=InterventionControls)
    decoding: DecodingOptions = Field(default_factory=DecodingOptions)
    evaluation_context: EvaluationContext | None = None
    snippet_id: str | None = None
    snippet_page: int | None = None
    snippet_bbox: list[float] | None = None
    snippet_text: str | None = None
    experiment_id: str | None = None
    visual_page_backend: Literal[
        "configured", "auto", "colqwen2", "clip", "disabled"
    ] = "configured"

    model_config = ConfigDict(extra="forbid")


class EvidenceIdentity(BaseModel):
    source_id: str
    local_id_kind: str
    local_id: str

    @property
    def global_id(self) -> str:
        return f"{self.source_id}::{self.local_id_kind}::{self.local_id}"


class RetrievalQueryChannelTrace(BaseModel):
    retrieval_query: str
    subquery_id: str
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    modality_score: float | None = None
    modality_rank: int | None = None
    image_embedding_score: float | None = None
    image_embedding_rank: int | None = None
    image_embedding_eligible: bool = False
    image_embedding_threshold: float | None = None
    image_embedding_corroborated: bool = False
    page_image_score: float | None = None
    page_image_rank: int | None = None
    page_image_eligible: bool = False
    page_image_threshold: float | None = None
    page_image_corroborated: bool = False
    visual_retrieval_backend: str | None = None
    visual_retrieval_model: str | None = None
    visual_inspection_candidate: bool = False
    rrf_score: float | None = None
    rerank_score: float | None = None


class RetrievalHitTrace(BaseModel):
    identity: EvidenceIdentity
    retrieval_queries: list[str] = Field(default_factory=list)
    subquery_ids: list[str] = Field(default_factory=list)
    query_channel_results: list[RetrievalQueryChannelTrace] = Field(default_factory=list)
    final_rank: int | None = None
    page: int | None = None
    section: str = ""
    modality: str = "text"
    bm25_score: float | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    modality_score: float | None = None
    modality_rank: int | None = None
    image_embedding_score: float | None = None
    image_embedding_rank: int | None = None
    image_embedding_eligible: bool = False
    image_embedding_threshold: float | None = None
    image_embedding_corroborated: bool = False
    page_image_score: float | None = None
    page_image_rank: int | None = None
    page_image_eligible: bool = False
    page_image_threshold: float | None = None
    page_image_corroborated: bool = False
    visual_retrieval_backend: str | None = None
    visual_retrieval_model: str | None = None
    candidate_regions: list[dict[str, Any]] = Field(default_factory=list)
    visual_inspection_candidate: bool = False
    rrf_score: float | None = None
    rerank_score: float | None = None
    selected_for_context: bool = False
    shown_to_generator: bool = False
    text_preview: str = ""


class PromptEvidenceTrace(BaseModel):
    prompt_evidence_id: str
    identity: EvidenceIdentity
    page: int | None = None
    section: str = ""
    modality: str = "text"
    quote: str
    content_sha256: str


class StageTiming(BaseModel):
    stage: str
    duration_ms: float
    status: Literal["ok", "error"] = "ok"
    error: str | None = None


class LocalGenerationMetadata(BaseModel):
    provider: str = "ollama"
    requested_model: str | None = None
    resolved_model: str | None = None
    model_digest: str | None = None
    quantization: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    prompt_version: str = ANSWER_PROMPT_VERSION
    prompt_sha256: str | None = None
    mode: GenerationMode = GenerationMode.NO_GENERATION
    prompt_eval_count: int | None = None
    eval_count: int | None = None
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_eval_duration_ns: int | None = None
    eval_duration_ns: int | None = None
    error: str | None = None


class CitationTrace(BaseModel):
    ref_id: int
    page: int | None = None
    chunk_id: str = ""
    section_title: str | None = None
    chunk_type: str | None = None
    quote: str = ""
    source_paper_id: str | None = None
    document_id: str | None = None
    source_evidence_id: str | None = None
    verification: str | None = None
    confidence: float | None = None
    origin: CitationOrigin = CitationOrigin.MODEL_EMITTED
    identity: EvidenceIdentity | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={"origin", "identity", "extra"}, exclude_none=True)
        payload.update(self.extra)
        return payload


class VerificationTrace(BaseModel):
    backend: str = "lexical-overlap"
    version: str = VERIFIER_VERSION
    initial_report: VerificationReport | None = None
    report: VerificationReport | None = None
    repair_requested: bool = False
    repair_actions_recorded: list[str] = Field(default_factory=list)
    edits: list[ClaimRepairRecord] = Field(default_factory=list)
    answer_text_changed: bool = False
    reverified: bool = False


class InterventionExecutionTrace(BaseModel):
    requested: InterventionControls = Field(default_factory=InterventionControls)
    executed_repair_mode: RepairMode = RepairMode.NONE
    verification_reached: bool = False

    model_config = ConfigDict(extra="forbid")


class AbstentionTrace(BaseModel):
    abstained: bool = False
    stage: str | None = None
    reason_code: str | None = None
    user_message: str | None = None


class RunIdentity(BaseModel):
    pipeline_version: str = ANSWER_PIPELINE_VERSION
    prompt_version: str = ANSWER_PROMPT_VERSION
    retriever_version: str = RETRIEVER_VERSION
    verifier_version: str = VERIFIER_VERSION
    git_revision: str | None = None
    git_dirty: bool | None = None
    experiment_id: str | None = None
    acquisition_mode: str = "local-prepared-artifacts"


class AnswerTrace(BaseModel):
    schema_version: Literal["1.0"] = ANSWER_TRACE_SCHEMA_VERSION
    trace_id: str
    timestamp: float
    status: PipelineStatus = PipelineStatus.SUCCESS
    paper_id: str
    query: str
    request: AnswerPipelineRequest
    run_identity: RunIdentity
    capabilities: ModelCapabilities
    route_budget: dict[str, Any] = Field(default_factory=dict)
    evidence_budget: EvidenceBudget
    analysis: QuestionAnalysis
    reasoning_level: str
    target_modalities: list[str] = Field(default_factory=list)
    subqueries: list[SubQuery] = Field(default_factory=list)
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)
    retrieval_hits: list[RetrievalHitTrace] = Field(default_factory=list)
    prompt_evidence: list[PromptEvidenceTrace] = Field(default_factory=list)
    evidence_graph: EvidenceGraph | None = None
    reasoning_path: list[ReasoningPathStep] = Field(default_factory=list)
    numeric_execution_plan: NumericPlan | None = None
    numeric_plan_used_for_generation: bool = False
    numeric_plan: NumericExecutionResult | None = None
    generation: LocalGenerationMetadata = Field(default_factory=LocalGenerationMetadata)
    raw_answer: str = ""
    normalized_answer: str = ""
    final_answer: str = ""
    citations: list[CitationTrace] = Field(default_factory=list)
    intervention: InterventionExecutionTrace = Field(default_factory=InterventionExecutionTrace)
    verification: VerificationTrace = Field(default_factory=VerificationTrace)
    verification_report: VerificationReport | None = None
    abstention: AbstentionTrace = Field(default_factory=AbstentionTrace)
    timings: list[StageTiming] = Field(default_factory=list)
    latency_ms: float = 0.0
    hardware_tier: str = ""
    persistence_succeeded: bool = False
    response_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def to_chat_response(self) -> dict[str, Any]:
        response: dict[str, Any] = {
            "answer": self.final_answer,
            "citations": [citation.to_api_dict() for citation in self.citations],
            "model": self.generation.resolved_model or self.generation.requested_model,
            "route_type": self.route_budget.get("route_type"),
            "capability_mode": self.capabilities.capability_mode.value,
            "reasoning_level": self.reasoning_level,
            "reasoning_steps": [step.model_dump() for step in self.reasoning_path],
            "numeric_plan": self.numeric_plan.model_dump() if self.numeric_plan else None,
            "verification_report": self.verification_report.model_dump() if self.verification_report else None,
            "abstained": self.abstention.abstained,
            "uncertainty_reason": self.abstention.reason_code,
            "trace_id": self.trace_id,
            "trace_schema_version": self.schema_version,
            "trace": self.model_dump(mode="json", by_alias=False),
        }
        response.update(self.response_metadata)
        if self.status == PipelineStatus.ERROR:
            response["error"] = True
            response["message"] = self.generation.error or "Answer generation failed."
        return response
