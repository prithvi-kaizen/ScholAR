"""Pydantic contracts for backward-compatible v1 and claim-bearing v2 releases."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from backend.schemas.answer_trace import (
    AnswerTrace,
    DecodingOptions,
    ExecutionPolicy,
    InterventionControls,
    PipelineStatus,
    RetrievalControls,
)


RELEASE_SCHEMA_VERSION = "1.0"
LATEST_RELEASE_SCHEMA_VERSION = "2.0"
ReleaseSchemaVersion = Literal["1.0", "2.0"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReleaseStudyStatus(str, Enum):
    NOT_READY = "NOT_READY"
    READY = "READY"


class RowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    ABSTAINED = "ABSTAINED"
    ERROR = "ERROR"


class CanonicalKey(StrictModel):
    system: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    seed: int
    case_id: str = Field(min_length=1, max_length=200)

    def as_tuple(self) -> tuple[str, str, int, str]:
        return self.system, self.model, self.seed, self.case_id

    def as_string(self) -> str:
        return "::".join((self.system, self.model, str(self.seed), self.case_id))


class DatasetSpec(StrictModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    cases_path: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    corpus_manifest_path: str | None = None
    evidence_class: Literal["measured", "proxy", "toy"]
    claim_status: Literal["current", "non_release"]
    paper_disjoint_from_development: bool = False

    @model_serializer(mode="wrap")
    def omit_unset_corpus_hash(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if self.corpus_sha256 is None:
            payload.pop("corpus_sha256", None)
        if self.corpus_manifest_path is None:
            payload.pop("corpus_manifest_path", None)
        return payload


class SystemOptions(StrictModel):
    execution_policy: ExecutionPolicy = ExecutionPolicy.REQUIRE_LOCAL_MODEL
    intervention: InterventionControls = Field(default_factory=InterventionControls)
    decoding: DecodingOptions = Field(default_factory=DecodingOptions)
    retrieval: RetrievalControls | None = None

    @model_serializer(mode="wrap")
    def omit_unset_retrieval(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if self.retrieval is None:
            payload.pop("retrieval", None)
        return payload


class SystemSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    runner: Literal["scholar_http", "fixture"] = "scholar_http"
    description: str = ""
    options: SystemOptions = Field(default_factory=SystemOptions)


class ModelSpec(StrictModel):
    tag: str = Field(min_length=1, max_length=200)
    digest: str | None = Field(default=None, min_length=8)
    quantization: str | None = None


class ComponentIdentity(StrictModel):
    """Stable identity for one non-generator experiment component."""

    component_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HardwareIdentity(StrictModel):
    tier: str = Field(min_length=1)
    exact_device: str = Field(min_length=1)
    profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FallbackPolicy(StrictModel):
    silent_fallback_allowed: Literal[False] = False
    missing_component_action: Literal["ERROR_ROW"] = "ERROR_ROW"
    condition_mismatch_action: Literal["ERROR_ROW"] = "ERROR_ROW"


class ExperimentIdentitySpec(StrictModel):
    """Full experiment identity required by schema v2 measured releases."""

    corpus_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ingestion_policy: ComponentIdentity
    chunking_policy: ComponentIdentity
    dense: ComponentIdentity
    reranker: ComponentIdentity
    clip: ComponentIdentity
    colqwen: ComponentIdentity
    verifier: ComponentIdentity
    calibration_artifact_hashes: dict[str, str] = Field(min_length=1)
    hardware: HardwareIdentity
    visual_observation_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    answer_prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fallback_policy: FallbackPolicy = Field(default_factory=FallbackPolicy)

    @model_validator(mode="after")
    def validate_calibration_hashes(self) -> "ExperimentIdentitySpec":
        invalid = [
            name for name, digest in self.calibration_artifact_hashes.items()
            if not name or not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ]
        if invalid:
            raise ValueError(f"invalid calibration artifact hashes: {invalid}")
        return self


MetricSource = Literal[
    "success_rate",
    "abstention_rate",
    "error_rate",
    "latency_ms",
    "supported_claim_rate",
    "partial_claim_rate",
    "contradiction_rate",
    "retained_claim_rate",
    "answer_word_count",
    "page_recall_at_1",
    "page_recall_at_3",
    "page_recall_at_5",
    "mrr",
    "region_recall_iou_0_5",
    "citation_precision",
    "citation_page_accuracy",
    "answerable_coverage",
    "unanswerable_abstention",
]


class MetricSpec(StrictModel):
    name: str = Field(min_length=1, max_length=100)
    source: MetricSource
    denominator: Literal[
        "all_expected",
        "eligible_only",
        "all_answerable_cases",
        "all_answerable_visual_cases",
        "all_unanswerable_cases",
        "all_emitted_citations",
    ] = "all_expected"
    on_error: float | None = None
    on_abstained: float | None = None
    description: str = ""


class ReleaseConfig(StrictModel):
    schema_version: ReleaseSchemaVersion = RELEASE_SCHEMA_VERSION
    release_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    run_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    study_status: ReleaseStudyStatus
    output_dir: str = Field(min_length=1)
    dataset: DatasetSpec
    systems: list[SystemSpec] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    seeds: list[int] = Field(min_length=1)
    metrics: list[MetricSpec] = Field(min_length=1)
    backend_url: str = "http://127.0.0.1:8000"
    command: list[str] = Field(default_factory=list)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    protocol_path: str | None = None
    protocol_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    experiment: ExperimentIdentitySpec | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> "ReleaseConfig":
        for label, values in (
            ("system", [item.name for item in self.systems]),
            ("model", [item.tag for item in self.models]),
            ("seed", self.seeds),
            ("metric", [item.name for item in self.metrics]),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} values are forbidden")
        if self.study_status == ReleaseStudyStatus.READY:
            if not self.dataset.sha256:
                raise ValueError("READY release config requires the frozen dataset sha256")
            if self.dataset.evidence_class != "toy" and not self.dataset.paper_disjoint_from_development:
                raise ValueError("READY measured/proxy releases require a paper-disjoint dataset")
            if self.dataset.evidence_class != "toy" and not self.dataset.corpus_sha256:
                raise ValueError(
                    "READY measured/proxy releases require the frozen corpus sha256"
                )
            missing = [item.tag for item in self.models if not item.digest or not item.quantization]
            if missing:
                raise ValueError(f"READY release config requires model digests and quantization: {missing}")
            if self.dataset.evidence_class != "toy":
                if not self.protocol_path or not self.protocol_sha256:
                    raise ValueError("READY empirical release config requires a frozen protocol")
                if not self.prompt_hashes:
                    raise ValueError("READY empirical release config requires prompt hashes")
                invalid_hashes = [
                    name for name, digest in self.prompt_hashes.items()
                    if not isinstance(digest, str) or not digest.startswith("sha256:") or len(digest.removeprefix("sha256:")) != 64
                ]
                if invalid_hashes:
                    raise ValueError(f"READY empirical release config has invalid prompt hashes: {invalid_hashes}")
                if self.release_id in {"eacl_industry_v1", "eacl_industry_v2"}:
                    self._validate_eacl_condition_matrix()
                if self.schema_version == LATEST_RELEASE_SCHEMA_VERSION:
                    if self.experiment is None:
                        raise ValueError("READY schema-v2 empirical release requires full experiment identity")
                    if self.dataset.corpus_sha256 != self.experiment.corpus_manifest_sha256:
                        raise ValueError("dataset corpus hash differs from v2 corpus manifest identity")
                    if not self.dataset.corpus_manifest_path:
                        raise ValueError("READY schema-v2 empirical release requires corpus_manifest_path")
                    expected_prompt_hashes = {
                        "answer": "sha256:" + self.experiment.answer_prompt_sha256,
                        "visual_observation": "sha256:" + self.experiment.visual_observation_prompt_sha256,
                    }
                    for name, digest in expected_prompt_hashes.items():
                        if self.prompt_hashes.get(name) != digest:
                            raise ValueError(f"v2 prompt hash {name!r} differs from experiment identity")
            elif self.schema_version == LATEST_RELEASE_SCHEMA_VERSION and self.experiment is not None:
                raise ValueError("toy schema-v2 releases must not carry claim-bearing experiment identity")
        for metric in self.metrics:
            if metric.denominator == "all_expected" and metric.on_error is None:
                raise ValueError(f"all_expected metric {metric.name!r} requires on_error")
        return self

    def _validate_eacl_condition_matrix(self) -> None:
        expected = {
            "S0": (False, "disabled", False, "NONE"),
            "S1": (True, "disabled", False, "NONE"),
            "S2": (False, "clip", False, "NONE"),
            "S3": (False, "colqwen2", False, "NONE"),
            "S4": (False, "colqwen2", True, "NONE"),
            "S5": (False, "colqwen2", True, "SELECTIVE"),
        }
        if {system.name for system in self.systems} != set(expected):
            raise ValueError("EACL release requires exactly the S0-S5 condition matrix")
        for system in self.systems:
            controls = system.options.retrieval
            if controls is None or controls.condition_id != system.name:
                raise ValueError(f"{system.name} requires matching typed retrieval controls")
            actual = (
                controls.include_crop_image_channel,
                controls.visual_page_backend,
                controls.pixel_inspection,
                system.options.intervention.repair_mode.value,
            )
            if actual != expected[system.name]:
                raise ValueError(f"{system.name} differs from the preregistered EACL condition")
            if controls.include_modality_channel:
                raise ValueError(f"{system.name} must keep the uncontrolled modality channel disabled")


class EvidenceRegion(StrictModel):
    page: int = Field(ge=1)
    bbox_norm: tuple[
        float,
        float,
        float,
        float,
    ]

    @model_validator(mode="after")
    def validate_bbox(self) -> "EvidenceRegion":
        x0, y0, x1, y1 = self.bbox_norm
        if not all(0.0 <= value <= 1.0 for value in self.bbox_norm):
            raise ValueError("normalized region coordinates must be in [0, 1]")
        if x0 >= x1 or y0 >= y1:
            raise ValueError("normalized region must have positive area")
        return self


class CaseRecord(StrictModel):
    case_id: str = Field(min_length=1, max_length=200)
    slot_id: str | None = None
    paper_id: str = Field(min_length=1, max_length=200)
    query: str = Field(min_length=1, max_length=8000)
    secondary_paper_ids: list[str] = Field(default_factory=list)
    split: Literal["development", "test"] | None = None
    pair_id: str | None = None
    formulation: Literal["explicit", "implicit", "not_applicable"] | None = None
    evidence_modality: Literal["text", "table", "plot", "diagram", "mixed"] | None = None
    answerable: bool | None = None
    gold_pages: list[int] = Field(default_factory=list)
    gold_regions: list[EvidenceRegion] = Field(default_factory=list)
    required_key_points: list[str] = Field(default_factory=list)
    acceptable_answers: list[str] = Field(default_factory=list)
    source_license: str | None = None
    redistribution_status: Literal["allowed", "restricted", "unknown"] | None = None
    annotation_status: Literal["UNANNOTATED", "DOUBLE_ANNOTATED", "ADJUDICATED"] = "UNANNOTATED"
    annotator_count: int = Field(default=0, ge=0)
    adjudication_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def measured_readiness_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.slot_id:
            errors.append("slot_id is required")
        if self.split != "test":
            errors.append("split must be test")
        for label, value in (
            ("formulation", self.formulation),
            ("evidence_modality", self.evidence_modality),
            ("answerable", self.answerable),
            ("source_license", self.source_license),
            ("redistribution_status", self.redistribution_status),
        ):
            if value is None or value == "":
                errors.append(f"{label} is required")
        if self.answerable is True:
            if not self.gold_pages:
                errors.append("answerable case requires gold_pages")
            if not self.required_key_points:
                errors.append("answerable case requires required_key_points")
            if not self.acceptable_answers:
                errors.append("answerable case requires acceptable_answers")
        if any(page < 1 for page in self.gold_pages):
            errors.append("gold_pages must be positive")
        if self.formulation in {"explicit", "implicit"} and not self.pair_id:
            errors.append("explicit/implicit case requires pair_id")
        if self.evidence_modality in {"table", "plot", "diagram", "mixed"} and self.answerable is True:
            if not self.gold_regions:
                errors.append("visual answerable case requires gold_regions")
        if self.annotation_status != "ADJUDICATED":
            errors.append("annotation_status must be ADJUDICATED")
        if self.annotator_count < 2:
            errors.append("at least two annotators are required")
        if not self.adjudication_sha256:
            errors.append("adjudication_sha256 is required")
        return errors


class ExpectedKeySet(StrictModel):
    schema_version: ReleaseSchemaVersion = RELEASE_SCHEMA_VERSION
    release_id: str
    run_id: str
    dataset_sha256: str
    n_expected: int = Field(ge=1)
    keys: list[CanonicalKey] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_keys(self) -> "ExpectedKeySet":
        tuples = [item.as_tuple() for item in self.keys]
        if len(tuples) != len(set(tuples)):
            raise ValueError("expected key universe contains duplicates")
        if self.n_expected != len(self.keys):
            raise ValueError("n_expected differs from the frozen key universe")
        if tuples != sorted(tuples):
            raise ValueError("expected keys must use canonical sorted order")
        return self


class ReleaseError(StrictModel):
    error_type: str = Field(min_length=1)
    message: str = Field(min_length=1)
    stage: str = "generation"


class FrozenRowIdentity(StrictModel):
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    case_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    paper_id: str
    query: str
    secondary_paper_ids: list[str] = Field(default_factory=list)
    system_name: str
    system_options: SystemOptions
    model_tag: str
    model_digest: str
    quantization: str
    seed: int
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    protocol_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    experiment: ExperimentIdentitySpec | None = None
    retrieval_controls_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    generator_identity_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    git_revision: str | None = None
    git_dirty: bool | None = None
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_serializer(mode="wrap")
    def omit_unset_corpus_hash(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if self.corpus_sha256 is None:
            payload.pop("corpus_sha256", None)
        return payload


class RawReleaseRow(StrictModel):
    schema_version: ReleaseSchemaVersion = RELEASE_SCHEMA_VERSION
    release_id: str
    run_id: str
    key: CanonicalKey
    identity: FrozenRowIdentity
    status: RowStatus
    trace: dict[str, Any] | None = None
    error: ReleaseError | None = None
    recorded_at: str

    @model_validator(mode="after")
    def validate_payload(self) -> "RawReleaseRow":
        if self.schema_version == LATEST_RELEASE_SCHEMA_VERSION:
            missing = [
                name for name, value in (
                    ("experiment", self.identity.experiment),
                    ("retrieval_controls_sha256", self.identity.retrieval_controls_sha256),
                    ("generator_identity_sha256", self.identity.generator_identity_sha256),
                ) if value is None
            ]
            if missing:
                raise ValueError(f"schema-v2 row lacks frozen identity fields: {missing}")
        if self.status == RowStatus.ERROR:
            if self.error is None:
                raise ValueError("ERROR rows require an error record")
            if self.trace is not None:
                trace = AnswerTrace.model_validate(self.trace)
                if trace.status != PipelineStatus.ERROR:
                    raise ValueError("ERROR release row contains a non-error AnswerTrace")
                if trace.request.execution_policy.value != "REQUIRE_LOCAL_MODEL":
                    raise ValueError("claim-bearing error traces must require the local model")
            return self
        if self.trace is None or self.error is not None:
            raise ValueError("SUCCESS/ABSTAINED rows require trace and forbid error")
        trace = AnswerTrace.model_validate(self.trace)
        expected = PipelineStatus(self.status.value)
        if trace.status != expected:
            raise ValueError("release-row status differs from AnswerTrace status")
        if trace.request.execution_policy.value != "REQUIRE_LOCAL_MODEL":
            raise ValueError("claim-bearing release rows must require the local model")
        if trace.generation.mode.value == "EXTRACTIVE_FALLBACK":
            raise ValueError("claim-bearing release rows forbid extractive fallback")
        return self


class ScoredReleaseRow(StrictModel):
    schema_version: ReleaseSchemaVersion = RELEASE_SCHEMA_VERSION
    release_id: str
    run_id: str
    key: CanonicalKey
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: RowStatus
    metrics: dict[str, float | None]
    metric_weights: dict[str, int] = Field(default_factory=dict)

    @model_serializer(mode="wrap")
    def omit_empty_weights(self, handler: Any) -> dict[str, Any]:
        payload = handler(self)
        if not self.metric_weights:
            payload.pop("metric_weights", None)
        return payload


class ArtifactHash(StrictModel):
    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class ReleaseManifest(StrictModel):
    schema_version: ReleaseSchemaVersion = RELEASE_SCHEMA_VERSION
    release_id: str
    run_id: str
    evidence_class: Literal["measured", "proxy", "toy"]
    claim_status: Literal["current", "non_release"]
    lifecycle_status: Literal["GENERATING", "RAW_COMPLETE", "SCORED", "AGGREGATED", "VALIDATED"]
    git_revision: str | None = None
    git_dirty: bool | None = None
    exact_command: list[str]
    dataset: dict[str, Any]
    systems: list[dict[str, Any]]
    models: list[dict[str, Any]]
    prompt_hashes: dict[str, str]
    experiment: ExperimentIdentitySpec | None = None
    seeds: list[int]
    hardware: dict[str, Any]
    software: dict[str, Any]
    started_at: str
    updated_at: str
    completed_at: str | None = None
    n_expected: int
    status_counts: dict[str, int]
    artifacts: list[ArtifactHash] = Field(default_factory=list)


class StudyTemplateEnvelope(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    artifact_type: Literal["template"] = "template"
    study_status: Literal["NOT_STARTED"] = "NOT_STARTED"
    contains_completed_data: Literal[False] = False
    records: list[Any] = Field(default_factory=list, max_length=0)
    instrument_id: str
    purpose: str


class ArtifactEvidenceRef(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)
    schema_name: str = Field(min_length=1)
    schema_version: str = Field(min_length=1)
    evidence_class: Literal["measured", "proxy", "toy", "governance", "official"]
    claim_status: Literal["current", "non_release"]
    allowed_use: str = Field(min_length=1)


class GateRecord(StrictModel):
    id: str = Field(min_length=1)
    phase: Literal["PRE_GENERATION", "PRE_SCORING", "PRE_PAPER", "POST_BUILD"]
    status: Literal["PENDING", "CLEARED", "FAILED"]
    evidence: list[ArtifactEvidenceRef] = Field(default_factory=list)
    reason: str | None = None

    @model_validator(mode="after")
    def require_evidence_when_cleared(self) -> "GateRecord":
        if self.status == "CLEARED" and not self.evidence:
            raise ValueError("CLEARED gates require typed hashed evidence")
        if self.status != "CLEARED" and self.evidence:
            raise ValueError("only CLEARED gates may attach supporting evidence")
        return self


class GateRegistry(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    release_id: str
    status: Literal["BLOCKED", "READY"]
    gates: list[GateRecord]

    @model_validator(mode="after")
    def derive_status(self) -> "GateRegistry":
        ids = [item.id for item in self.gates]
        if len(ids) != len(set(ids)):
            raise ValueError("gate registry contains duplicate IDs")
        expected = "READY" if self.gates and all(item.status == "CLEARED" for item in self.gates) else "BLOCKED"
        if self.status != expected:
            raise ValueError(f"gate registry status must be derived as {expected}")
        return self


class FrozenOutputRef(StrictModel):
    release_id: str
    run_id: str
    key: CanonicalKey
    condition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_id: str
    raw_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    final_answer_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HumanSupportLabel(StrictModel):
    output: FrozenOutputRef
    claim_id: str
    claim_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label: Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"]
    annotator_id: str
    round: Literal["INITIAL", "ADJUDICATION"] = "INITIAL"
    instrument_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FrozenKeyPoint(StrictModel):
    case_id: str
    key_point_id: str
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    required: bool = True


class KeyPointCoverageLabel(StrictModel):
    output: FrozenOutputRef
    key_point_id: str
    covered: bool
    annotator_id: str
    round: Literal["INITIAL", "ADJUDICATION"] = "INITIAL"
    instrument_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class HumanEvaluationBundle(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    artifact_type: Literal["independent_human_evaluation"] = "independent_human_evaluation"
    evidence_class: Literal["measured", "toy"]
    claim_status: Literal["current", "non_release"]
    release_id: str
    run_id: str
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_independent_annotators: int = Field(default=2, ge=2)
    support_labels: list[HumanSupportLabel]
    key_points: list[FrozenKeyPoint]
    coverage_labels: list[KeyPointCoverageLabel]

    @model_validator(mode="after")
    def validate_classification(self) -> "HumanEvaluationBundle":
        if self.evidence_class == "toy" and self.claim_status != "non_release":
            raise ValueError("toy human-evaluation bundles must be non_release")
        if self.evidence_class == "measured" and self.claim_status != "current":
            raise ValueError("measured human-evaluation bundles must be current")
        return self


class PairedComparisonSpec(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    control_system: str
    treatment_system: str
    support_improvement_threshold: float = 0.05
    maximum_coverage_loss: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = Field(default=10000, ge=100)
    bootstrap_seed: int = 2027


class PrimaryGateResult(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    artifact_type: Literal["primary_support_coverage_gate"] = "primary_support_coverage_gate"
    evidence_class: Literal["measured", "toy"]
    claim_status: Literal["current", "non_release"]
    decision: Literal["PASS", "FAIL", "BLOCKED"]
    support_delta: float | None = None
    support_interval_low: float | None = None
    support_interval_high: float | None = None
    coverage_delta: float | None = None
    n_complete_pairs: int = Field(ge=0)
    reasons: list[str] = Field(default_factory=list)
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ReleaseTableSeal(StrictModel):
    schema_version: Literal["1.0"] = RELEASE_SCHEMA_VERSION
    artifact_type: Literal["validated_release_table_seal"] = "validated_release_table_seal"
    release_id: str
    run_id: str
    evidence_class: Literal["measured"]
    claim_status: Literal["current"]
    release_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    release_checksums_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary_table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gate_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    primary_gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    claim_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    validator_version: Literal["release-table-seal-v1"] = "release-table-seal-v1"


class ClaimEvidenceRef(StrictModel):
    artifact: ArtifactEvidenceRef
    selector: str = Field(min_length=1)
    expected_decision: Literal["PASS", "FAIL", "DESCRIPTIVE"]
