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
    payload = case.model_dump(mode="json")
    legacy_defaults = {
        "slot_id": None,
        "split": None,
        "pair_id": None,
        "formulation": None,
        "evidence_modality": None,
        "answerable": None,
        "gold_pages": [],
        "gold_regions": [],
        "required_key_points": [],
        "acceptable_answers": [],
        "source_license": None,
        "redistribution_status": None,
        "annotation_status": "UNANNOTATED",
        "annotator_count": 0,
        "adjudication_sha256": None,
    }
    for field, default in legacy_defaults.items():
        if payload.get(field) == default:
            payload.pop(field, None)
    return sha256_bytes(canonical_json_bytes(payload))


def _condition_digest(identity: FrozenRowIdentity | dict[str, object]) -> str:
    payload = (
        identity.model_dump(
            mode="json",
            exclude={"condition_sha256"},
            exclude_none=True,
        )
        if isinstance(identity, FrozenRowIdentity)
        else {
            key: value
            for key, value in identity.items()
            if key != "condition_sha256" and value is not None
        }
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
    if config.protocol_sha256:
        payload["protocol_sha256"] = config.protocol_sha256
    if config.schema_version == "2.0":
        payload["experiment"] = (
            config.experiment.model_dump(mode="json") if config.experiment else None
        )
        payload["retrieval_controls_sha256"] = sha256_bytes(
            canonical_json_bytes(system.options.retrieval.model_dump(mode="json"))
        ) if system.options.retrieval is not None else None
        payload["generator_identity_sha256"] = sha256_bytes(canonical_json_bytes({
            "tag": model.tag,
            "digest": model.digest,
            "quantization": model.quantization,
        }))
    payload["condition_sha256"] = _condition_digest(payload)
    return FrozenRowIdentity.model_validate(payload)


def expected_context(config: ReleaseConfig, key: CanonicalKey) -> EvaluationContext:
    experiment_identity_sha256 = (
        sha256_bytes(canonical_json_bytes(config.experiment))
        if config.experiment is not None else None
    )
    return EvaluationContext(
        release_id=config.release_id,
        run_id=config.run_id,
        system_name=key.system,
        case_id=key.case_id,
        dataset_sha256=config.dataset.sha256 or "",
        corpus_sha256=config.dataset.corpus_sha256,
        prompt_hashes=config.prompt_hashes,
        experiment_identity_sha256=experiment_identity_sha256,
    )


def _compare_trace(
    trace: AnswerTrace,
    identity: FrozenRowIdentity,
    config: ReleaseConfig,
    key: CanonicalKey,
    case: CaseRecord,
    system: SystemSpec,
    model: ModelSpec,
    corpus_manifest: dict[str, object] | None = None,
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
        ("retrieval controls", request.retrieval, system.options.retrieval),
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

    controls = system.options.retrieval
    if controls is not None:
        if not controls.pixel_inspection and trace.generation.mode == GenerationMode.VISION_MODEL:
            errors.append("pixel inspection executed while disabled by frozen condition")
        if not controls.include_crop_image_channel and any(
            hit.image_embedding_eligible for hit in trace.retrieval_hits
        ):
            errors.append("crop-image retrieval executed while disabled by frozen condition")
        if controls.visual_page_backend == "disabled" and any(
            hit.page_image_eligible for hit in trace.retrieval_hits
        ):
            errors.append("page-image retrieval executed while disabled by frozen condition")
        if controls.strict_components:
            reranker = trace.retrieval_metadata.get("reranker")
            if not isinstance(reranker, dict) or reranker.get("model_loaded") is not True:
                errors.append("frozen cross-encoder reranker did not execute")
            statuses = trace.retrieval_metadata.get("visual_page_retrieval")
            if controls.visual_page_backend != "disabled":
                if not isinstance(statuses, list) or not statuses:
                    errors.append("frozen page retriever lacks execution status")
                else:
                    for status in statuses:
                        if (
                            not isinstance(status, dict)
                            or status.get("requested_backend") != controls.visual_page_backend
                            or status.get("succeeded") is not True
                        ):
                            errors.append("frozen page retriever did not execute exactly as configured")
                            break

        if config.schema_version == "2.0":
            executions = trace.retrieval_metadata.get("channel_execution")
            if not isinstance(executions, list) or not executions:
                errors.append("schema-v2 trace lacks channel execution records")
            else:
                expected_channels = {
                    "bm25": True,
                    "dense": True,
                    "modality": controls.include_modality_channel,
                    "crop_image": controls.include_crop_image_channel,
                    "page_image": controls.visual_page_backend != "disabled",
                    "reranker": True,
                }
                for invocation in executions:
                    channels = invocation.get("channels") if isinstance(invocation, dict) else None
                    if not isinstance(channels, dict):
                        errors.append("schema-v2 trace has malformed channel execution record")
                        continue
                    for channel, enabled in expected_channels.items():
                        status = channels.get(channel)
                        if not isinstance(status, dict):
                            errors.append(f"schema-v2 trace lacks {channel} execution status")
                            continue
                        if status.get("requested") is not enabled:
                            errors.append(f"{channel} requested state differs from frozen condition")
                        if status.get("executed") is not enabled:
                            errors.append(f"{channel} execution differs from frozen condition")

                experiment = config.experiment
                if experiment is not None:
                    first = executions[0].get("channels", {}) if isinstance(executions[0], dict) else {}
                    expected_components = {
                        "dense": experiment.dense,
                        "reranker": experiment.reranker,
                    }
                    if controls.include_crop_image_channel:
                        expected_components["crop_image"] = experiment.clip
                    if controls.visual_page_backend == "clip":
                        expected_components["page_image"] = experiment.clip
                    elif controls.visual_page_backend == "colqwen2":
                        expected_components["page_image"] = experiment.colqwen
                    for channel, component in expected_components.items():
                        status = first.get(channel) if isinstance(first, dict) else None
                        identity_status = status.get("component_identity") if isinstance(status, dict) else None
                        actual_fingerprint = (
                            identity_status.get("encoder_fingerprint")
                            if isinstance(identity_status, dict) else None
                        )
                        if actual_fingerprint != component.identity_sha256:
                            errors.append(f"{channel} component identity differs from frozen condition")
                        if channel in {"crop_image", "page_image"} and (
                            not isinstance(identity_status, dict)
                            or identity_status.get("threshold_calibrated") is not True
                        ):
                            errors.append(f"{channel} used an uncalibrated retrieval threshold")

                    verifier_digest = sha256_bytes(canonical_json_bytes({
                        "backend": trace.verification.backend,
                        "version": trace.verification.version,
                    }))
                    if verifier_digest != experiment.verifier.identity_sha256:
                        errors.append("verifier identity differs from frozen condition")
                    scorer = (
                        trace.verification.report.scorer
                        if trace.verification.report is not None else None
                    )
                    if scorer is None or scorer.thresholds_calibrated is not True:
                        errors.append("verifier used thresholds without a frozen calibration profile")
                    elif not scorer.threshold_profile_id:
                        errors.append("verifier trace lacks the frozen calibration profile identity")
                    if trace.hardware_tier != experiment.hardware.tier:
                        errors.append("hardware tier differs from frozen condition")
                    if trace.hardware_device != experiment.hardware.exact_device:
                        errors.append("exact hardware device differs from frozen condition")

            source_rows = trace.response_metadata.get("source_bundle_identities")
            if not isinstance(source_rows, list):
                errors.append("schema-v2 trace lacks source-bundle identities")
            elif not isinstance(corpus_manifest, dict):
                errors.append("schema-v2 validation lacks the frozen corpus manifest")
            else:
                manifest_papers = corpus_manifest.get("papers")
                by_paper = {
                    str(item.get("paper_id")): item
                    for item in manifest_papers
                    if isinstance(item, dict) and item.get("paper_id")
                } if isinstance(manifest_papers, list) else {}
                traced = {
                    str(item.get("paper_id")): item
                    for item in source_rows
                    if isinstance(item, dict) and item.get("paper_id")
                }
                required_sources = {case.paper_id, *case.secondary_paper_ids}
                if set(traced) != required_sources:
                    errors.append("trace source-bundle universe differs from frozen case")
                for source_id in sorted(required_sources):
                    source_trace = traced.get(source_id)
                    source_manifest = by_paper.get(source_id)
                    if not isinstance(source_trace, dict) or not isinstance(source_manifest, dict):
                        errors.append(f"source {source_id} is absent from trace or frozen corpus")
                        continue
                    artifacts = source_manifest.get("source_artifacts")
                    ingestion_hash = next((
                        item.get("sha256") for item in artifacts
                        if isinstance(item, dict) and item.get("path") == "ingestion_manifest.json"
                    ), None) if isinstance(artifacts, list) else None
                    for label, actual, expected_value in (
                        ("ingestion manifest", source_trace.get("ingestion_manifest_sha256"), ingestion_hash),
                        ("PDF", source_trace.get("pdf_sha256"), source_manifest.get("pdf_sha256")),
                        ("chunks", source_trace.get("chunks_sha256"), source_manifest.get("chunks_sha256")),
                        ("parser", source_trace.get("parser_engine"), source_manifest.get("parser_engine")),
                        (
                            "ingestion policy",
                            source_trace.get("ingestion_policy_identity_sha256"),
                            source_manifest.get("ingestion_policy_identity_sha256"),
                        ),
                        (
                            "chunking policy",
                            source_trace.get("chunking_policy_identity_sha256"),
                            source_manifest.get("chunking_policy_identity_sha256"),
                        ),
                    ):
                        if actual != expected_value:
                            errors.append(f"source {source_id} {label} identity differs from frozen corpus")
                    experiment = config.experiment
                    if experiment is not None:
                        if (
                            source_trace.get("ingestion_policy_identity_sha256")
                            != experiment.ingestion_policy.identity_sha256
                        ):
                            errors.append(f"source {source_id} uses a different ingestion policy")
                        if (
                            source_trace.get("chunking_policy_identity_sha256")
                            != experiment.chunking_policy.identity_sha256
                        ):
                            errors.append(f"source {source_id} uses a different chunking policy")

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
    corpus_manifest: dict[str, object] | None = None,
) -> list[str]:
    errors: list[str] = []
    if row.schema_version != config.schema_version:
        errors.append("row schema version differs from release config")
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
        errors.extend(_compare_trace(
            trace, expected, config, row.key, case, system, model, corpus_manifest
        ))
    return errors
