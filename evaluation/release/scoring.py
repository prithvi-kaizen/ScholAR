"""Pure score-only transformation from immutable raw AnswerTrace rows."""

from __future__ import annotations

import re
from pathlib import Path

from backend.schemas.answer_trace import AnswerTrace
from backend.schemas.claims import EntailmentStatus
from evaluation.release.io import key_index, read_jsonl, write_jsonl
from evaluation.release.schemas import MetricSpec, RawReleaseRow, RowStatus, ScoredReleaseRow


def _claim_rates(trace: AnswerTrace) -> dict[str, float]:
    report = trace.verification.report or trace.verification_report
    claims = report.claims if report else []
    count = len(claims)
    if not count:
        return {
            "supported_claim_rate": 0.0,
            "partial_claim_rate": 0.0,
            "contradiction_rate": 0.0,
        }
    status_count = {status: 0 for status in EntailmentStatus}
    for claim in claims:
        status_count[claim.entailment_status] += 1
    return {
        "supported_claim_rate": status_count[EntailmentStatus.SUPPORTED] / count,
        "partial_claim_rate": status_count[EntailmentStatus.PARTIAL] / count,
        "contradiction_rate": status_count[EntailmentStatus.CONTRADICTED] / count,
    }


def _metric_from_trace(row: RawReleaseRow, source: str) -> float | None:
    if source == "success_rate":
        return 1.0 if row.status == RowStatus.SUCCESS else 0.0
    if source == "abstention_rate":
        return 1.0 if row.status == RowStatus.ABSTAINED else 0.0
    if source == "error_rate":
        return 1.0 if row.status == RowStatus.ERROR else 0.0
    if row.trace is None:
        return None
    trace = AnswerTrace.model_validate(row.trace)
    if source == "latency_ms":
        return float(trace.latency_ms)
    if source in {"supported_claim_rate", "partial_claim_rate", "contradiction_rate"}:
        return _claim_rates(trace)[source]
    if source == "retained_claim_rate":
        initial = trace.verification.initial_report
        final = trace.verification.report or trace.verification_report
        if initial is None or not initial.claims:
            return 1.0 if final and final.claims else 0.0
        return min(1.0, len(final.claims if final else []) / len(initial.claims))
    if source == "answer_word_count":
        return float(len(re.findall(r"\b\w+\b", trace.final_answer)))
    raise ValueError(f"unsupported metric source: {source}")


def score_metric(row: RawReleaseRow, metric: MetricSpec) -> float | None:
    if row.status == RowStatus.ERROR:
        value = metric.on_error
    elif row.status == RowStatus.ABSTAINED and metric.on_abstained is not None:
        value = metric.on_abstained
    else:
        value = _metric_from_trace(row, metric.source)
    if metric.denominator == "all_expected" and value is None:
        raise ValueError(
            f"all_expected metric {metric.name!r} produced no value for {row.key.as_string()}"
        )
    return round(float(value), 6) if value is not None else None


def score_rows(rows: list[RawReleaseRow], metrics: list[MetricSpec]) -> list[ScoredReleaseRow]:
    key_index(rows)
    output = [
        ScoredReleaseRow(
            release_id=row.release_id,
            run_id=row.run_id,
            key=row.key,
            condition_sha256=row.identity.condition_sha256,
            status=row.status,
            metrics={metric.name: score_metric(row, metric) for metric in metrics},
        )
        for row in rows
    ]
    return sorted(output, key=lambda row: row.key.as_tuple())


def score_release(raw_path: Path, scored_path: Path, metrics: list[MetricSpec]) -> list[ScoredReleaseRow]:
    """Read raw rows and write scores. This function has no generation adapter."""
    rows = read_jsonl(raw_path, RawReleaseRow)
    scored = score_rows(rows, metrics)
    write_jsonl(scored_path, scored)
    return scored
