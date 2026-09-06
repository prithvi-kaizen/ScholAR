"""Pure construction and validation of frozen per-row release identity."""

from __future__ import annotations

from backend.schemas.answer_trace import AnswerTrace, EvaluationContext, GenerationMode, RepairMode
from evaluation.release.io import canonical_json_bytes, sha256_bytes
from evaluation.release.schemas import (
    CanonicalKey,
    CaseRecord,
    FrozenRowIdentity,
    ModelSpec,
    RawReleaseRow,
    ReleaseConfig,
    ReleaseManifest,
    SystemSpec,
)


def case_sha256(case: CaseRecord) -> str:
    return sha256_bytes(canonical_json_bytes(case))


def _condition_digest(identity: FrozenRowIdentity | dict[str, object]) -> str:
    payload = (
        identity.model_dump(
            mode="json",
            exclude={"condition_sha256"},
            exclude_none=True,
        )
        if isinstance(identity, FrozenRowIdentity)
        else identity
    )
    return sha256_bytes(canonical_json_bytes(payload))


def build_frozen_identity(
    config: ReleaseConfig,
    key: CanonicalKey,
    case: CaseRecord,
    *,
    git_revision: str | None,
    git_dirty: bool | None,
) -> FrozenRowIdentity:
    system = next(item for item in config.systems if item.name == key.system)
    model = next(item for item in config.models if item.tag == key.model)
    payload: dict[str, object] = {
        "dataset_sha256": config.dataset.sha256 or "",
        "case_sha256": case_sha256(case),
        "paper_id": case.paper_id,
        "query": case.query,
        "secondary_paper_ids": case.secondary_paper_ids,
        "system_name": system.name,
        "system_options": system.options.model_dump(mode="json"),
        "model_tag": model.tag,
        "model_digest": model.digest or "",
        "quantization": model.quantization or "",
        "seed": key.seed,
        "prompt_hashes": config.prompt_hashes,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
    }
    if config.dataset.corpus_sha256:
        payload["corpus_sha256"] = config.dataset.corpus_sha256
    payload["condition_sha256"] = _condition_digest(payload)
    return FrozenRowIdentity.model_validate(payload)


def expected_context(config: ReleaseConfig, key: CanonicalKey) -> EvaluationContext:
    return EvaluationContext(
        release_id=config.release_id,
        run_id=config.run_id,
        system_name=key.system,
        case_id=key.case_id,
        dataset_sha256=config.dataset.sha256 or "",
        corpus_sha256=config.dataset.corpus_sha256,
        prompt_hashes=config.prompt_hashes,
    )


def _compare_trace(
    trace: AnswerTrace,
    identity: FrozenRowIdentity,
    config: ReleaseConfig,
    key: CanonicalKey,
    case: CaseRecord,
    system: SystemSpec,
    model: ModelSpec,
) -> list[str]:
    errors: list[str] = []
    request = trace.request
    for label, actual, expected in (
        ("trace paper_id", trace.paper_id, case.paper_id),
        ("trace query", trace.query, case.query),
        ("request paper_id", request.paper_id, case.paper_id),
        ("request query", request.query, case.query),
        ("secondary paper IDs", request.secondary_paper_ids, case.secondary_paper_ids),
        ("requested model", request.requested_model, model.tag),
        ("generation seed", request.generation_seed, key.seed),
        ("execution policy", request.execution_policy, system.options.execution_policy),
        ("intervention controls", request.intervention, system.options.intervention),
        ("decoding options", request.decoding, system.options.decoding),
        ("evaluation context", request.evaluation_context, expected_context(config, key)),
        ("trace requested intervention", trace.intervention.requested, system.options.intervention),
        ("git revision", trace.run_identity.git_revision, identity.git_revision),
        ("git dirty state", trace.run_identity.git_dirty, identity.git_dirty),
    ):
        if actual != expected:
            errors.append(f"{label} differs from frozen condition")

    expected_executed = (
        system.options.intervention.repair_mode
        if trace.intervention.verification_reached
        else RepairMode.NONE
    )
    if trace.intervention.executed_repair_mode != expected_executed:
        errors.append("executed intervention differs from frozen condition")

    if trace.generation.mode in {GenerationMode.LOCAL_MODEL, GenerationMode.VISION_MODEL}:
        expected_options = system.options.decoding.model_dump(mode="json")
        expected_options["seed"] = key.seed
        for label, actual, expected in (
            ("generation requested model", trace.generation.requested_model, model.tag),
            ("generation resolved model", trace.generation.resolved_model, model.tag),
            ("model digest", trace.generation.model_digest, model.digest),
            ("model quantization", trace.generation.quantization, model.quantization),
            ("generation options", trace.generation.options, expected_options),
        ):
            if actual != expected:
                errors.append(f"{label} differs from frozen condition")
    return errors


def validate_row_against_condition(
    row: RawReleaseRow,
    config: ReleaseConfig,
    case: CaseRecord,
    manifest: ReleaseManifest,
) -> list[str]:
    errors: list[str] = []
    try:
        system = next(item for item in config.systems if item.name == row.key.system)
        model = next(item for item in config.models if item.tag == row.key.model)
    except StopIteration:
        return ["row key references an unknown system or model"]
    expected = build_frozen_identity(
        config,
        row.key,
        case,
        git_revision=manifest.git_revision,
        git_dirty=manifest.git_dirty,
    )
    if row.identity != expected:
        errors.append("row frozen identity differs from config/case/manifest")
    if row.identity.condition_sha256 != _condition_digest(row.identity):
        errors.append("row condition digest is invalid")
    if row.trace is not None:
        try:
            trace = AnswerTrace.model_validate(row.trace)
        except Exception as exc:
            return errors + [f"invalid AnswerTrace: {exc}"]
        errors.extend(_compare_trace(trace, expected, config, row.key, case, system, model))
    return errors
