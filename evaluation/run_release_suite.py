"""Freeze expected keys and resumably generate immutable release raw rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.gates import validate_gate_registry  # noqa: E402
from evaluation.release.identity import (  # noqa: E402
    build_frozen_identity,
    expected_context,
    validate_row_against_condition,
)
from evaluation.release.io import (  # noqa: E402
    atomic_write,
    build_expected_keys,
    key_index,
    load_cases,
    load_corpus_manifest,
    load_config,
    read_json,
    read_jsonl,
    resolve_repo_path,
    sha256_file,
    write_json,
    write_jsonl,
)
from evaluation.release.manifest import create_manifest, load_manifest, update_manifest, utc_now  # noqa: E402
from evaluation.release.schemas import (  # noqa: E402
    CanonicalKey,
    ExpectedKeySet,
    RawReleaseRow,
    ReleaseConfig,
    ReleaseError,
    ReleaseStudyStatus,
    RowStatus,
)
from evaluation.scholar_runner import run_scholar_http  # noqa: E402
from evaluation.protocol_governance import snapshot_protocol  # noqa: E402
from backend.services.network_policy_service import NetworkPolicyService  # noqa: E402


Generator = Callable[[ReleaseConfig, CanonicalKey, object], RawReleaseRow]
DIRECTORIES = ("configs", "data_cards", "raw", "scored", "aggregates", "tables", "figures", "human", "logs")


def _generate_scholar(config: ReleaseConfig, key: CanonicalKey, case: object) -> RawReleaseRow:
    from evaluation.release.schemas import CaseRecord

    item = CaseRecord.model_validate(case)
    release_dir = resolve_repo_path(config.output_dir)
    manifest = load_manifest(release_dir / "manifest.json")
    frozen_identity = build_frozen_identity(
        config,
        key,
        item,
        git_revision=manifest.git_revision,
        git_dirty=manifest.git_dirty,
    )
    system = next(value for value in config.systems if value.name == key.system)
    try:
        result = run_scholar_http(
            config.backend_url,
            item.paper_id,
            item.query,
            key.model,
            require_local_model=system.options.execution_policy.value == "REQUIRE_LOCAL_MODEL",
            secondary_paper_ids=item.secondary_paper_ids,
            experiment_id=f"{config.run_id}:{key.system}",
            generation_seed=key.seed,
            intervention=system.options.intervention,
            decoding=system.options.decoding,
            retrieval=system.options.retrieval,
            evaluation_context=expected_context(config, key),
            allow_error_trace=True,
        )
        trace = result.trace
        status = RowStatus(trace.status.value)
        if status == RowStatus.ERROR:
            message = trace.generation.error or "Answer pipeline returned an error trace"
            return RawReleaseRow(
                schema_version=config.schema_version,
                release_id=config.release_id,
                run_id=config.run_id,
                key=key,
                identity=frozen_identity,
                status=status,
                trace=trace.model_dump(mode="json"),
                error=ReleaseError(error_type="AnswerPipelineError", message=message),
                recorded_at=utc_now(),
            )
        return RawReleaseRow(
            schema_version=config.schema_version,
            release_id=config.release_id,
            run_id=config.run_id,
            key=key,
            identity=frozen_identity,
            status=status,
            trace=trace.model_dump(mode="json"),
            recorded_at=utc_now(),
        )
    except Exception as exc:
        return RawReleaseRow(
            schema_version=config.schema_version,
            release_id=config.release_id,
            run_id=config.run_id,
            key=key,
            identity=frozen_identity,
            status=RowStatus.ERROR,
            error=ReleaseError(error_type=type(exc).__name__, message=str(exc) or repr(exc)),
            recorded_at=utc_now(),
        )


def _verify_local_runtime(config: ReleaseConfig) -> dict[str, str]:
    """Return condition errors so each expected key can receive a terminal ERROR row."""
    errors: dict[str, str] = {}
    if not NetworkPolicyService.is_loopback_url(config.backend_url):
        return {"*": "release backend must be a loopback URL"}
    try:
        with urllib.request.urlopen(
            f"{config.backend_url.rstrip('/')}/api/system/network-policy", timeout=5
        ) as response:
            policy = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        return {"*": f"release backend preflight failed: {type(exc).__name__}: {exc}"}
    if policy.get("mode") != "strict-local" or policy.get("external_network_allowed") is not False:
        return {"*": "release backend must run in strict-local mode"}

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    if not NetworkPolicyService.is_loopback_url(ollama_url):
        return {"*": "release Ollama endpoint must be a loopback URL"}
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as response:
            installed = json.loads(response.read().decode("utf-8")).get("models", [])
    except Exception as exc:
        return {"*": f"Ollama model preflight failed: {type(exc).__name__}: {exc}"}
    by_name = {item.get("name") or item.get("model"): item for item in installed}
    for model in config.models:
        item = by_name.get(model.tag)
        if item is None:
            errors[model.tag] = f"frozen release model is not installed: {model.tag}"
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        if str(item.get("digest") or "") != model.digest:
            errors[model.tag] = f"installed digest differs from frozen config for {model.tag}"
        elif str(details.get("quantization_level") or "") != model.quantization:
            errors[model.tag] = f"installed quantization differs from frozen config for {model.tag}"
    return errors


def _condition_error_row(
    config: ReleaseConfig,
    key: CanonicalKey,
    case: object,
    manifest: object,
    message: str,
) -> RawReleaseRow:
    from evaluation.release.schemas import CaseRecord, ReleaseManifest

    item = CaseRecord.model_validate(case)
    frozen_manifest = ReleaseManifest.model_validate(manifest)
    identity = build_frozen_identity(
        config,
        key,
        item,
        git_revision=frozen_manifest.git_revision,
        git_dirty=frozen_manifest.git_dirty,
    )
    return RawReleaseRow(
        schema_version=config.schema_version,
        release_id=config.release_id,
        run_id=config.run_id,
        key=key,
        identity=identity,
        status=RowStatus.ERROR,
        error=ReleaseError(
            error_type="FrozenConditionError",
            message=message,
            stage="condition_validation",
        ),
        recorded_at=utc_now(),
    )


def run_release(config_path: Path, generator: Generator | None = None) -> Path:
    config = load_config(config_path)
    if config.study_status != ReleaseStudyStatus.READY:
        raise ValueError("release generation is blocked until config study_status is READY")
    if config.dataset.evidence_class != "toy":
        gates_path = resolve_repo_path(config.output_dir) / "gates.json"
        registry, gate_errors = validate_gate_registry(
            gates_path,
            required_phase="PRE_GENERATION",
        )
        if registry is None or registry.release_id != config.release_id or gate_errors:
            details = gate_errors or ["gate registry release_id differs from config"]
            raise ValueError("release generation blocked by evidence gates:\n- " + "\n- ".join(details))
    runtime_errors = _verify_local_runtime(config) if generator is None else {}
    release_dir = resolve_repo_path(config.output_dir)
    for name in DIRECTORIES:
        (release_dir / name).mkdir(parents=True, exist_ok=True)
    if config.dataset.evidence_class != "toy":
        if not config.protocol_path or not config.protocol_sha256:
            raise ValueError("empirical release requires a frozen protocol path and hash")
        snapshot_protocol(
            resolve_repo_path(config.protocol_path),
            release_dir,
            config.protocol_sha256,
        )
        if config.schema_version == "2.0":
            if not config.dataset.corpus_manifest_path or not config.dataset.corpus_sha256:
                raise ValueError("schema-v2 empirical release requires a frozen corpus manifest")
            source_manifest = resolve_repo_path(config.dataset.corpus_manifest_path)
            if sha256_file(source_manifest) != config.dataset.corpus_sha256:
                raise ValueError("schema-v2 corpus manifest differs from its frozen hash")
            corpus_snapshot = release_dir / "configs/corpus_manifest.json"
            if corpus_snapshot.exists():
                if corpus_snapshot.read_bytes() != source_manifest.read_bytes():
                    raise ValueError("release corpus-manifest snapshot is immutable and differs")
            else:
                atomic_write(corpus_snapshot, source_manifest.read_bytes())

    snapshot_path = release_dir / "configs/release_config.json"
    if snapshot_path.exists():
        existing = ReleaseConfig.model_validate(read_json(snapshot_path))
        if existing != config:
            raise ValueError("release config snapshot is immutable and differs from requested config")
    else:
        write_json(snapshot_path, config)

    cases = load_cases(config)
    corpus_manifest = load_corpus_manifest(config)
    expected = build_expected_keys(config, cases)
    expected_path = release_dir / "expected_keys.json"
    if expected_path.exists():
        frozen = ExpectedKeySet.model_validate(read_json(expected_path))
        if frozen != expected:
            raise ValueError("frozen expected-key universe differs from current config/dataset")
    else:
        write_json(expected_path, expected)

    raw_path = release_dir / "raw/rows.jsonl"
    rows = read_jsonl(raw_path, RawReleaseRow) if raw_path.exists() else []
    indexed = key_index(rows)
    expected_set = {key.as_tuple() for key in expected.keys}
    if not set(indexed).issubset(expected_set):
        raise ValueError("raw rows contain keys outside the frozen expected universe")

    manifest_path = release_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_manifest(manifest_path)
    else:
        manifest = create_manifest(config, release_dir, expected.n_expected, ROOT)
        write_json(manifest_path, manifest)
    if (manifest.release_id, manifest.run_id) != (config.release_id, config.run_id):
        raise ValueError("existing manifest belongs to a different release/run")

    case_by_id = {case.case_id: case for case in cases}
    system_by_name = {system.name: system for system in config.systems}
    for existing_row in rows:
        identity_errors = validate_row_against_condition(
            existing_row,
            config,
            case_by_id[existing_row.key.case_id],
            manifest,
            corpus_manifest,
        )
        if identity_errors:
            raise ValueError(
                f"immutable raw row identity mismatch for {existing_row.key.as_string()}: "
                + "; ".join(identity_errors)
            )
    chosen_generator = generator or _generate_scholar
    for key in expected.keys:
        if key.as_tuple() in indexed:
            continue  # SUCCESS, ABSTAINED, and ERROR are all immutable terminal rows.
        system = system_by_name[key.system]
        if generator is None and system.runner != "scholar_http":
            raise ValueError(f"CLI generation cannot execute runner {system.runner!r}")
        runtime_error = runtime_errors.get("*") or runtime_errors.get(key.model)
        if runtime_error:
            row = _condition_error_row(
                config, key, case_by_id[key.case_id], manifest, runtime_error
            )
        else:
            row = chosen_generator(config, key, case_by_id[key.case_id])
        if row.key != key or row.release_id != config.release_id or row.run_id != config.run_id:
            row = _condition_error_row(
                config,
                key,
                case_by_id[key.case_id],
                manifest,
                "generator returned a row for the wrong canonical key/release",
            )
        identity_errors = validate_row_against_condition(
            row,
            config,
            case_by_id[key.case_id],
            manifest,
            corpus_manifest,
        )
        if identity_errors:
            row = _condition_error_row(
                config,
                key,
                case_by_id[key.case_id],
                manifest,
                "generator returned a mismatched frozen condition: "
                + "; ".join(identity_errors),
            )
        rows.append(row)
        rows.sort(key=lambda item: item.key.as_tuple())
        indexed[row.key.as_tuple()] = row
        write_jsonl(raw_path, rows)
        update_manifest(manifest, release_dir, raw_rows=rows, lifecycle_status="GENERATING")

    if set(indexed) != expected_set:
        raise RuntimeError("generation ended without one terminal row per expected key")
    update_manifest(manifest, release_dir, raw_rows=rows, lifecycle_status="RAW_COMPLETE")
    return release_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    release_dir = run_release(args.config)
    print(f"raw release complete: {release_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
