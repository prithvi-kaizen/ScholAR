"""Build the deterministic non-release release-v1 fixture."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas.answer_trace import (  # noqa: E402
    AbstentionTrace,
    AnswerPipelineRequest,
    AnswerTrace,
    ExecutionPolicy,
    GenerationMode,
    InterventionExecutionTrace,
    LocalGenerationMetadata,
    PipelineStatus,
    RepairMode,
    RunIdentity,
    VerificationTrace,
)
from backend.schemas.capabilities import EvidenceBudget, ModelCapabilities  # noqa: E402
from backend.schemas.claims import AtomicClaim, EntailmentStatus, VerificationReport  # noqa: E402
from backend.schemas.reasoning import QuestionAnalysis  # noqa: E402
from evaluation.aggregate_release import run as aggregate_run  # noqa: E402
from evaluation.release.identity import build_frozen_identity, expected_context  # noqa: E402
from evaluation.release.io import load_config, read_jsonl, resolve_repo_path, write_jsonl  # noqa: E402
from evaluation.release.manifest import load_manifest, update_manifest  # noqa: E402
from evaluation.release.schemas import CanonicalKey, CaseRecord, RawReleaseRow, ReleaseConfig, ReleaseError, RowStatus  # noqa: E402
from evaluation.run_release_suite import run_release  # noqa: E402
from evaluation.score_release import run as score_run  # noqa: E402
from evaluation.validate_release import run as validate_run  # noqa: E402

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "fixture_config.json"


def _trace(config: ReleaseConfig, key: CanonicalKey, case: CaseRecord, status: RowStatus) -> AnswerTrace:
    system = next(item for item in config.systems if item.name == key.system)
    manifest = load_manifest(resolve_repo_path(config.output_dir) / "manifest.json")
    request = AnswerPipelineRequest(
        paper_id=case.paper_id,
        query=case.query,
        requested_model=key.model,
        execution_policy=system.options.execution_policy,
        intervention=system.options.intervention,
        decoding=system.options.decoding,
        evaluation_context=expected_context(config, key),
        generation_seed=key.seed,
        experiment_id=f"{config.run_id}:{key.system}",
    )
    claim = AtomicClaim(
        claim_id="C1",
        text="The fixture exercises deterministic release accounting.",
        normalized_text="The fixture exercises deterministic release accounting.",
        start=0,
        end=55,
        final_start=0,
        final_end=55,
        entailment_status=EntailmentStatus.SUPPORTED,
        first_pass_status=EntailmentStatus.SUPPORTED,
        second_pass_status=EntailmentStatus.SUPPORTED,
    )
    report = VerificationReport(
        claims=[claim],
        overall_supported=True,
        supported_count=1,
        final_verified_response=claim.text,
        second_pass_completed=True,
    )
    abstained = status == RowStatus.ABSTAINED
    final_answer = (
        "The fixture has insufficient evidence and abstains."
        if abstained else claim.text
    )
    return AnswerTrace(
        trace_id=f"trace_fixture_{key.seed}_{case.case_id}",
        timestamp=1700000000.0 + key.seed,
        status=PipelineStatus(status.value),
        paper_id=case.paper_id,
        query=case.query,
        request=request,
        run_identity=RunIdentity(
            experiment_id=f"{config.run_id}:{key.system}",
            git_revision=manifest.git_revision,
            git_dirty=manifest.git_dirty,
        ),
        capabilities=ModelCapabilities(model_id=key.model, display_name="Fixture Model"),
        evidence_budget=EvidenceBudget(),
        analysis=QuestionAnalysis(original_query=case.query),
        reasoning_level="L1_DIRECT_LOOKUP",
        generation=LocalGenerationMetadata(
            requested_model=key.model,
            resolved_model=key.model,
            model_digest="sha256:fixture-model-v1",
            quantization="none",
            options={**system.options.decoding.model_dump(mode="json"), "seed": key.seed},
            mode=GenerationMode.NO_GENERATION if abstained else GenerationMode.LOCAL_MODEL,
        ),
        raw_answer="" if abstained else final_answer,
        normalized_answer="" if abstained else final_answer,
        final_answer=final_answer,
        verification=VerificationTrace(
            initial_report=None if abstained else report,
            report=None if abstained else report,
            repair_requested=(
                not abstained and system.options.intervention.repair_mode != RepairMode.NONE
            ),
            answer_text_changed=False,
            reverified=not abstained,
        ),
        intervention=InterventionExecutionTrace(
            requested=system.options.intervention,
            executed_repair_mode=(
                RepairMode.NONE if abstained else system.options.intervention.repair_mode
            ),
            verification_reached=not abstained,
        ),
        verification_report=None if abstained else report,
        abstention=AbstentionTrace(
            abstained=abstained,
            stage="sufficiency" if abstained else None,
            reason_code="FIXTURE_INSUFFICIENT" if abstained else None,
            user_message=final_answer if abstained else None,
        ),
        latency_ms=float(10 + key.seed),
        hardware_tier="fixture",
        persistence_succeeded=True,
    )


def _generate(config: ReleaseConfig, key: CanonicalKey, raw_case: object) -> RawReleaseRow:
    case = CaseRecord.model_validate(raw_case)
    manifest = load_manifest(resolve_repo_path(config.output_dir) / "manifest.json")
    identity = build_frozen_identity(
        config,
        key,
        case,
        git_revision=manifest.git_revision,
        git_dirty=manifest.git_dirty,
    )
    recorded = "2026-01-01T00:00:00Z"
    if key.case_id == "case_beta" and key.seed == 29:
        return RawReleaseRow(
            release_id=config.release_id,
            run_id=config.run_id,
            key=key,
            identity=identity,
            status=RowStatus.ERROR,
            error=ReleaseError(error_type="FixtureError", message="Intentional immutable toy failure."),
            recorded_at=recorded,
        )
    status = RowStatus.ABSTAINED if key.case_id == "case_beta" else RowStatus.SUCCESS
    trace = _trace(config, key, case, status)
    return RawReleaseRow(
        release_id=config.release_id,
        run_id=config.run_id,
        key=key,
        identity=identity,
        status=status,
        trace=trace.model_dump(mode="json"),
        recorded_at=recorded,
    )


def main() -> int:
    release_dir = run_release(CONFIG, generator=_generate)
    raw_path = release_dir / "raw/rows.jsonl"
    rows = read_jsonl(raw_path, RawReleaseRow)
    # Deliberately non-canonical order proves downstream stages are order independent.
    write_jsonl(raw_path, [rows[2], rows[0], rows[3], rows[1]])
    manifest = load_manifest(release_dir / "manifest.json")
    update_manifest(manifest, release_dir, raw_rows=rows, lifecycle_status="RAW_COMPLETE")
    score_run(CONFIG)
    aggregate_run(CONFIG)
    errors = validate_run(release_dir, mark_validated=True)
    if errors:
        raise RuntimeError("fixture validation failed:\n- " + "\n- ".join(errors))
    print(f"release-v1 fixture rebuilt and validated: {release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
