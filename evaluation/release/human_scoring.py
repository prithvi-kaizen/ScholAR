"""Deterministic consumer for independent support and key-point annotations."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from typing import Iterable

from backend.schemas.answer_trace import AnswerTrace
from evaluation.release.io import canonical_json_bytes, sha256_bytes
from evaluation.release.schemas import (
    FrozenOutputRef,
    HumanEvaluationBundle,
    PairedComparisonSpec,
    PrimaryGateResult,
    RawReleaseRow,
    RowStatus,
)


def _row_sha256(row: RawReleaseRow) -> str:
    return sha256_bytes(canonical_json_bytes(row))


def _answer_sha256(trace: AnswerTrace) -> str:
    return hashlib.sha256(trace.final_answer.encode("utf-8")).hexdigest()


def _validate_output_ref(reference: FrozenOutputRef, row: RawReleaseRow) -> list[str]:
    errors: list[str] = []
    if (reference.release_id, reference.run_id, reference.key) != (
        row.release_id,
        row.run_id,
        row.key,
    ):
        errors.append("annotation output release/run/key differs from raw row")
    if reference.condition_sha256 != row.identity.condition_sha256:
        errors.append("annotation output condition digest differs from raw row")
    if reference.raw_row_sha256 != _row_sha256(row):
        errors.append("annotation output raw-row digest differs from raw row")
    if row.trace is None:
        errors.append("annotation refers to a trace-less row")
    else:
        trace = AnswerTrace.model_validate(row.trace)
        if reference.trace_id != trace.trace_id:
            errors.append("annotation output trace_id differs from raw row")
        if reference.final_answer_sha256 != _answer_sha256(trace):
            errors.append("annotation output answer digest differs from raw row")
    return errors


def _resolve_binary(
    values: Iterable[tuple[str, str, bool]],
    minimum_annotators: int,
) -> tuple[float | None, str | None]:
    initial = [(annotator, value) for annotator, round_name, value in values if round_name == "INITIAL"]
    if len({annotator for annotator, _ in initial}) < minimum_annotators:
        return None, "fewer than the required independent initial annotators"
    distinct = {value for _, value in initial}
    if len(distinct) == 1:
        return (1.0 if next(iter(distinct)) else 0.0), None
    adjudicated = [value for _, round_name, value in values if round_name == "ADJUDICATION"]
    if len(adjudicated) != 1:
        return None, "disagreement lacks exactly one adjudication label"
    return (1.0 if adjudicated[0] else 0.0), None


def _interval(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    if not values:
        raise ValueError("paired interval requires at least one value")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(
        sum(values[rng.randrange(len(values))] for _ in values) / len(values)
        for _ in range(samples)
    )
    low_index = int(0.025 * (samples - 1))
    high_index = int(0.975 * (samples - 1))
    return means[low_index], means[high_index]


def score_primary_gate(
    bundle: HumanEvaluationBundle,
    spec: PairedComparisonSpec,
    rows: list[RawReleaseRow],
) -> PrimaryGateResult:
    input_hash = sha256_bytes(canonical_json_bytes(bundle))
    spec_hash = sha256_bytes(canonical_json_bytes(spec))
    reasons: list[str] = []
    indexed = {row.key.as_tuple(): row for row in rows}
    if len(indexed) != len(rows):
        reasons.append("raw rows contain duplicate canonical keys")
    if any((row.release_id, row.run_id) != (bundle.release_id, bundle.run_id) for row in rows):
        reasons.append("raw row release/run differs from annotation bundle")

    relevant = [
        row for row in rows
        if row.key.system in {spec.control_system, spec.treatment_system}
    ]
    pair_index: dict[tuple[str, int, str], dict[str, RawReleaseRow]] = defaultdict(dict)
    for row in relevant:
        pair_index[(row.key.model, row.key.seed, row.key.case_id)][row.key.system] = row
    expected_systems = {spec.control_system, spec.treatment_system}
    incomplete = [key for key, systems in pair_index.items() if set(systems) != expected_systems]
    if not pair_index or incomplete:
        reasons.append("control/treatment rows do not form a complete paired universe")

    support_groups: dict[tuple[tuple[str, str, int, str], str], list[tuple[str, str, bool]]] = defaultdict(list)
    for label in bundle.support_labels:
        row = indexed.get(label.output.key.as_tuple())
        if row is None:
            reasons.append("support label references an unknown raw row")
            continue
        reasons.extend(_validate_output_ref(label.output, row))
        support_groups[(label.output.key.as_tuple(), label.claim_id)].append(
            (label.annotator_id, label.round, label.label == "SUPPORTED")
        )

    key_points_by_case: dict[str, set[str]] = defaultdict(set)
    for point in bundle.key_points:
        if point.required:
            key_points_by_case[point.case_id].add(point.key_point_id)
    coverage_groups: dict[tuple[tuple[str, str, int, str], str], list[tuple[str, str, bool]]] = defaultdict(list)
    for label in bundle.coverage_labels:
        row = indexed.get(label.output.key.as_tuple())
        if row is None:
            reasons.append("coverage label references an unknown raw row")
            continue
        reasons.extend(_validate_output_ref(label.output, row))
        coverage_groups[(label.output.key.as_tuple(), label.key_point_id)].append(
            (label.annotator_id, label.round, label.covered)
        )

    scores: dict[tuple[str, str, int, str], tuple[float, float]] = {}
    for row in relevant:
        key = row.key.as_tuple()
        if row.status in {RowStatus.ERROR, RowStatus.ABSTAINED}:
            scores[key] = (0.0, 0.0)
            continue
        claim_groups = [values for (row_key, _), values in support_groups.items() if row_key == key]
        if not claim_groups:
            reasons.append(f"successful row lacks support labels: {row.key.as_string()}")
            continue
        support_values: list[float] = []
        for values in claim_groups:
            value, error = _resolve_binary(values, bundle.minimum_independent_annotators)
            if error:
                reasons.append(f"support labels for {row.key.as_string()}: {error}")
            elif value is not None:
                support_values.append(value)
        required_points = key_points_by_case.get(row.key.case_id, set())
        if not required_points:
            reasons.append(f"case lacks frozen required key points: {row.key.case_id}")
            continue
        coverage_values: list[float] = []
        for point_id in sorted(required_points):
            values = coverage_groups.get((key, point_id), [])
            value, error = _resolve_binary(values, bundle.minimum_independent_annotators)
            if error:
                reasons.append(f"coverage labels for {row.key.as_string()}/{point_id}: {error}")
            elif value is not None:
                coverage_values.append(value)
        if len(coverage_values) != len(required_points) or len(support_values) != len(claim_groups):
            continue
        scores[key] = (
            sum(support_values) / len(support_values),
            sum(coverage_values) / len(coverage_values),
        )

    support_deltas: list[float] = []
    coverage_deltas: list[float] = []
    for pair_key, systems in sorted(pair_index.items()):
        if set(systems) != expected_systems:
            continue
        control_key = systems[spec.control_system].key.as_tuple()
        treatment_key = systems[spec.treatment_system].key.as_tuple()
        if control_key not in scores or treatment_key not in scores:
            reasons.append(f"paired outputs lack complete annotations: {pair_key}")
            continue
        control_support, control_coverage = scores[control_key]
        treatment_support, treatment_coverage = scores[treatment_key]
        support_deltas.append(treatment_support - control_support)
        coverage_deltas.append(treatment_coverage - control_coverage)

    if reasons or len(support_deltas) != len(pair_index):
        return PrimaryGateResult(
            evidence_class=bundle.evidence_class,
            claim_status=bundle.claim_status,
            decision="BLOCKED",
            n_complete_pairs=len(support_deltas),
            reasons=sorted(set(reasons)),
            input_sha256=input_hash,
            spec_sha256=spec_hash,
        )

    support_delta = sum(support_deltas) / len(support_deltas)
    coverage_delta = sum(coverage_deltas) / len(coverage_deltas)
    low, high = _interval(support_deltas, spec.bootstrap_samples, spec.bootstrap_seed)
    passed = (
        support_delta >= spec.support_improvement_threshold
        and low > 0.0
        and coverage_delta >= -spec.maximum_coverage_loss
    )
    return PrimaryGateResult(
        evidence_class=bundle.evidence_class,
        claim_status=bundle.claim_status,
        decision="PASS" if passed else "FAIL",
        support_delta=round(support_delta, 6),
        support_interval_low=round(low, 6),
        support_interval_high=round(high, 6),
        coverage_delta=round(coverage_delta, 6),
        n_complete_pairs=len(support_deltas),
        reasons=[] if passed else ["predeclared support/interval/coverage decision rule was not met"],
        input_sha256=input_hash,
        spec_sha256=spec_hash,
    )
