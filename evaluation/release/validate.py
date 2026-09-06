"""Fail-closed validation for complete release-v1 directories."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from backend.schemas.answer_trace import AnswerTrace
from evaluation.release.identity import validate_row_against_condition
from evaluation.release.io import (
    key_index,
    load_cases,
    read_checksums,
    read_json,
    read_jsonl,
    sha256_file,
)
from evaluation.release.schemas import (
    ExpectedKeySet,
    RawReleaseRow,
    ReleaseConfig,
    ReleaseManifest,
    ScoredReleaseRow,
)


REQUIRED_DIRECTORIES = ("configs", "data_cards", "raw", "scored", "aggregates", "tables", "figures", "human", "logs")
REQUIRED_FILES = (
    "manifest.json",
    "checksums.sha256",
    "expected_keys.json",
    "configs/release_config.json",
    "raw/rows.jsonl",
    "scored/rows.jsonl",
    "aggregates/summary.json",
    "tables/summary.csv",
    "tables/summary.tex",
    "tables/provenance.json",
)
FORBIDDEN_RELEASE_TOKENS = (
    "legacy_non_empirical",
    "baseline_comparison_results.json",
    "ablation_study_results.json",
    "user_study_results.json",
)
ANONYMITY_PATTERNS = (
    re.compile(r"/Users/", re.I),
    re.compile(r"github\.com/prithvi-kaizen", re.I),
    re.compile(r"prithviraj", re.I),
)


def _error(errors: list[str], message: str) -> None:
    errors.append(message)


def validate_release_directory(release_dir: Path) -> list[str]:
    root = Path(release_dir)
    errors: list[str] = []
    for name in REQUIRED_DIRECTORIES:
        if not (root / name).is_dir():
            _error(errors, f"missing release directory: {name}")
    for name in REQUIRED_FILES:
        if not (root / name).is_file():
            _error(errors, f"missing release file: {name}")
    if errors:
        return errors

    try:
        config = ReleaseConfig.model_validate(read_json(root / "configs/release_config.json"))
        expected = ExpectedKeySet.model_validate(read_json(root / "expected_keys.json"))
        raw_rows = read_jsonl(root / "raw/rows.jsonl", RawReleaseRow)
        scored_rows = read_jsonl(root / "scored/rows.jsonl", ScoredReleaseRow)
        manifest = ReleaseManifest.model_validate(read_json(root / "manifest.json"))
        aggregate = read_json(root / "aggregates/summary.json")
        provenance = read_json(root / "tables/provenance.json")
    except Exception as exc:
        return [f"release schema/parse failure: {exc}"]

    identities = {(config.release_id, config.run_id), (expected.release_id, expected.run_id), (manifest.release_id, manifest.run_id)}
    identities.update((row.release_id, row.run_id) for row in raw_rows)
    identities.update((row.release_id, row.run_id) for row in scored_rows)
    identities.add((aggregate.get("release_id"), aggregate.get("run_id")))
    if len(identities) != 1:
        _error(errors, f"release/run identity mismatch: {sorted(identities)}")
    if config.study_status.value != "READY":
        _error(errors, "complete release config is not READY")
    if manifest.lifecycle_status not in {"AGGREGATED", "VALIDATED"}:
        _error(errors, f"manifest lifecycle is incomplete: {manifest.lifecycle_status}")
    if config.dataset.evidence_class != manifest.evidence_class or config.dataset.claim_status != manifest.claim_status:
        _error(errors, "manifest evidence classification differs from config")
    if manifest.evidence_class != "toy" and manifest.claim_status != "current":
        _error(errors, "empirical release must be classified current")
    if manifest.evidence_class == "toy" and manifest.claim_status != "non_release":
        _error(errors, "toy fixture must be classified non_release")
    if expected.dataset_sha256 != config.dataset.sha256:
        _error(errors, "expected-key dataset hash differs from release config")
    if manifest.dataset != config.dataset.model_dump(mode="json"):
        _error(errors, "manifest dataset snapshot differs from release config")
    if manifest.systems != [item.model_dump(mode="json") for item in config.systems]:
        _error(errors, "manifest system/intervention snapshot differs from release config")
    if manifest.models != [item.model_dump(mode="json") for item in config.models]:
        _error(errors, "manifest model snapshot differs from release config")
    if manifest.seeds != config.seeds or manifest.prompt_hashes != config.prompt_hashes:
        _error(errors, "manifest seed or prompt identity differs from release config")
    try:
        cases = load_cases(config)
        cases_by_id = {case.case_id: case for case in cases}
    except Exception as exc:
        _error(errors, f"frozen dataset validation failed: {exc}")
        cases_by_id = {}

    try:
        raw_index = key_index(raw_rows)
        scored_index = key_index(scored_rows)
    except ValueError as exc:
        _error(errors, str(exc))
        raw_index, scored_index = {}, {}
    expected_set = {key.as_tuple() for key in expected.keys}
    if set(raw_index) != expected_set:
        _error(errors, "raw key universe differs from frozen expected keys")
    if set(scored_index) != expected_set:
        _error(errors, "scored key universe differs from frozen expected keys")
    for key in sorted(set(raw_index) & set(scored_index)):
        if raw_index[key].status.value != scored_index[key].status.value:
            _error(errors, f"raw/scored status mismatch for {key}")
        if raw_index[key].identity.condition_sha256 != scored_index[key].condition_sha256:
            _error(errors, f"raw/scored condition identity mismatch for {key}")

    actual_status = dict(sorted(Counter(row.status.value for row in raw_rows).items()))
    if manifest.n_expected != expected.n_expected or aggregate.get("n_expected") != expected.n_expected:
        _error(errors, "n_expected accounting differs across expected keys, manifest, or aggregate")
    if manifest.status_counts != actual_status or aggregate.get("status_counts") != actual_status:
        _error(errors, "status counts do not account for every raw outcome")
    metric_specs = {item.name: item for item in config.metrics}
    if aggregate.get("metric_names") != list(metric_specs):
        _error(errors, "aggregate metric order differs from release config")
    for group in aggregate.get("groups", []):
        if group.get("n_expected", 0) <= 0:
            _error(errors, "aggregate group has no expected rows")
        for name, spec in metric_specs.items():
            if spec.denominator == "all_expected" and group.get("metric_case_denominators", {}).get(name) != group.get("n_cases"):
                _error(errors, f"all_expected metric {name!r} dropped a case denominator")

    aggregate_hash = sha256_file(root / "aggregates/summary.json")
    if provenance.get("aggregate_sha256") != aggregate_hash:
        _error(errors, "table provenance does not match aggregate JSON")
    for table_name in ("summary.csv", "summary.tex"):
        text = (root / "tables" / table_name).read_text(encoding="utf-8")
        if aggregate_hash not in text:
            _error(errors, f"table lacks aggregate provenance hash: {table_name}")

    try:
        declared = read_checksums(root / "checksums.sha256")
        actual_paths = sorted(
            path for path in root.rglob("*")
            if path.is_file()
            and path.name not in {"checksums.sha256", ".gitkeep", ".DS_Store"}
            and path.suffix != ".pyc"
            and "__pycache__" not in path.parts
        )
        actual_names = {path.relative_to(root).as_posix() for path in actual_paths}
        if set(declared) != actual_names:
            _error(errors, "checksum inventory differs from release file inventory")
        for name, expected_hash in declared.items():
            path = root / name
            if path.is_file() and sha256_file(path) != expected_hash:
                _error(errors, f"checksum mismatch: {name}")
    except Exception as exc:
        _error(errors, f"checksum validation failed: {exc}")

    artifact_hashes = {item.path: item.sha256 for item in manifest.artifacts}
    for name in ("raw/rows.jsonl", "scored/rows.jsonl", "aggregates/summary.json", "tables/summary.csv", "tables/summary.tex"):
        if artifact_hashes.get(name) != sha256_file(root / name):
            _error(errors, f"manifest artifact hash missing or stale: {name}")

    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv", ".tex", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if '"artifact_type":"template"' in text.replace(" ", "") or '"study_status":"NOT_STARTED"' in text.replace(" ", ""):
            _error(errors, f"template-only artifact appears inside empirical release: {path.relative_to(root)}")
        for token in FORBIDDEN_RELEASE_TOKENS:
            if token in text:
                _error(errors, f"release references forbidden legacy evidence in {path.relative_to(root)}: {token}")
        for pattern in ANONYMITY_PATTERNS:
            if pattern.search(text):
                _error(errors, f"release anonymity hygiene failed in {path.relative_to(root)}")

    for row in raw_rows:
        case = cases_by_id.get(row.key.case_id)
        if case is None:
            _error(errors, f"row references missing frozen case: {row.key.as_string()}")
            continue
        for identity_error in validate_row_against_condition(row, config, case, manifest):
            _error(errors, f"row identity {row.key.as_string()}: {identity_error}")
        if row.trace is not None:
            try:
                AnswerTrace.model_validate(row.trace)
            except Exception as exc:
                _error(errors, f"invalid AnswerTrace for {row.key.as_string()}: {exc}")
    return sorted(set(errors))


def assert_valid_release(release_dir: Path) -> None:
    errors = validate_release_directory(release_dir)
    if errors:
        raise ValueError("release validation failed:\n- " + "\n- ".join(errors))
