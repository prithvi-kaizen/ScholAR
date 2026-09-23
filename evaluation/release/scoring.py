"""Pure score-only transformation from immutable raw AnswerTrace rows."""

from __future__ import annotations

import re
from pathlib import Path

from backend.schemas.answer_trace import AnswerTrace, PipelineStatus
from backend.schemas.claims import EntailmentStatus
from evaluation.release.io import key_index, read_jsonl, write_jsonl
from evaluation.release.schemas import CaseRecord, MetricSpec, RawReleaseRow, RowStatus, ScoredReleaseRow


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


def _ranked_pages(trace: AnswerTrace) -> list[int]:
    ordered = sorted(
        (hit for hit in trace.retrieval_hits if hit.page is not None),
        key=lambda hit: (hit.final_rank is None, hit.final_rank or 10**9, hit.identity.global_id),
    )
    pages: list[int] = []
    for hit in ordered:
        assert hit.page is not None
        if hit.page not in pages:
            pages.append(hit.page)
    return pages


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    lx0, ly0, lx1, ly1 = left
    rx0, ry0, rx1, ry1 = right
    width = max(0.0, min(lx1, rx1) - max(lx0, rx0))
    height = max(0.0, min(ly1, ry1) - max(ly0, ry0))
    intersection = width * height
    union = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_bbox(value: object) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        return None
    bbox = tuple(float(item) for item in value)
    x0, y0, x1, y1 = bbox
    return bbox if 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 else None


def _objective_metric(trace: AnswerTrace, case: CaseRecord, source: str) -> tuple[float | None, int]:
    if source in {"page_recall_at_1", "page_recall_at_3", "page_recall_at_5", "mrr"}:
        if case.answerable is not True:
            return None, 0
        pages = _ranked_pages(trace)
        gold = set(case.gold_pages)
        first = next((rank for rank, page in enumerate(pages, 1) if page in gold), None)
        if source == "mrr":
            return (0.0 if first is None else 1.0 / first), 1
        cutoff = int(source.rsplit("_", 1)[1])
        return float(first is not None and first <= cutoff), 1
    if source == "region_recall_iou_0_5":
        if case.answerable is not True or not case.gold_regions:
            return None, 0
        for hit in trace.retrieval_hits:
            if hit.page is None:
                continue
            gold_regions = [gold for gold in case.gold_regions if gold.page == hit.page]
            for candidate in hit.candidate_regions:
                bbox = _candidate_bbox(candidate.get("bbox_norm") if isinstance(candidate, dict) else None)
                if bbox is not None and any(_iou(bbox, gold.bbox_norm) >= 0.5 for gold in gold_regions):
                    return 1.0, 1
        return 0.0, 1
    if source in {"citation_precision", "citation_page_accuracy"}:
        if case.answerable is not True or not trace.citations:
            return None, 0
        if source == "citation_precision":
            correct = sum(
                citation.page in set(case.gold_pages)
                and (citation.source_paper_id or citation.document_id or trace.paper_id) == case.paper_id
                for citation in trace.citations
            )
        else:
            retrieved = {
                (hit.identity.global_id, hit.page) for hit in trace.retrieval_hits
            }
            correct = sum(
                citation.identity is not None
                and (citation.identity.global_id, citation.page) in retrieved
                for citation in trace.citations
            )
        return correct / len(trace.citations), len(trace.citations)
    if source == "answerable_coverage":
        return (float(trace.status == PipelineStatus.SUCCESS), 1) if case.answerable is True else (None, 0)
    if source == "unanswerable_abstention":
        return (float(trace.status == PipelineStatus.ABSTAINED), 1) if case.answerable is False else (None, 0)
    raise ValueError(f"unsupported objective metric source: {source}")


_OBJECTIVE_SOURCES = {
    "page_recall_at_1", "page_recall_at_3", "page_recall_at_5", "mrr",
    "region_recall_iou_0_5", "citation_precision", "citation_page_accuracy",
    "answerable_coverage", "unanswerable_abstention",
}


def _metric_from_trace(
    row: RawReleaseRow, source: str, case: CaseRecord | None = None
) -> tuple[float | None, int]:
    if source == "success_rate":
        return (1.0 if row.status == RowStatus.SUCCESS else 0.0), 1
    if source == "abstention_rate":
        return (1.0 if row.status == RowStatus.ABSTAINED else 0.0), 1
    if source == "error_rate":
        return (1.0 if row.status == RowStatus.ERROR else 0.0), 1
    if source in _OBJECTIVE_SOURCES and case is None:
        raise ValueError(f"metric {source!r} requires its frozen CaseRecord")
    if row.trace is None:
        return None, 0
    trace = AnswerTrace.model_validate(row.trace)
    if source in _OBJECTIVE_SOURCES:
        assert case is not None
        return _objective_metric(trace, case, source)
    if source == "latency_ms":
        return float(trace.latency_ms), 1
    if source in {"supported_claim_rate", "partial_claim_rate", "contradiction_rate"}:
        return _claim_rates(trace)[source], 1
    if source == "retained_claim_rate":
        initial = trace.verification.initial_report
        final = trace.verification.report or trace.verification_report
        if initial is None or not initial.claims:
            return (1.0 if final and final.claims else 0.0), 1
        return min(1.0, len(final.claims if final else []) / len(initial.claims)), 1
    if source == "answer_word_count":
        return float(len(re.findall(r"\b\w+\b", trace.final_answer))), 1
    raise ValueError(f"unsupported metric source: {source}")


def _case_eligible(case: CaseRecord | None, source: str) -> bool:
    if source in {"page_recall_at_1", "page_recall_at_3", "page_recall_at_5", "mrr", "answerable_coverage"}:
        return case is not None and case.answerable is True
    if source == "region_recall_iou_0_5":
        return case is not None and case.answerable is True and bool(case.gold_regions)
    if source in {"citation_precision", "citation_page_accuracy"}:
        return case is not None and case.answerable is True
    if source == "unanswerable_abstention":
        return case is not None and case.answerable is False
    return True


def score_metric_with_weight(
    row: RawReleaseRow, metric: MetricSpec, case: CaseRecord | None = None
) -> tuple[float | None, int]:
    if not _case_eligible(case, metric.source):
        return None, 0
    if row.status == RowStatus.ERROR:
        value = metric.on_error
        weight = 1 if value is not None else 0
    elif row.status == RowStatus.ABSTAINED and metric.on_abstained is not None:
        value = metric.on_abstained
        weight = 1
    else:
        value, weight = _metric_from_trace(row, metric.source, case)
    if metric.denominator == "all_expected" and value is None:
        raise ValueError(
            f"all_expected metric {metric.name!r} produced no value for {row.key.as_string()}"
        )
    return (round(float(value), 6) if value is not None else None), weight


def score_metric(
    row: RawReleaseRow, metric: MetricSpec, case: CaseRecord | None = None
) -> float | None:
    return score_metric_with_weight(row, metric, case)[0]


def score_rows(
    rows: list[RawReleaseRow],
    metrics: list[MetricSpec],
    cases: dict[str, CaseRecord] | None = None,
) -> list[ScoredReleaseRow]:
    key_index(rows)
    output: list[ScoredReleaseRow] = []
    for row in rows:
        case = cases.get(row.key.case_id) if cases is not None else None
        scored = {
            metric.name: score_metric_with_weight(row, metric, case)
            for metric in metrics
        }
        output.append(ScoredReleaseRow(
            schema_version=row.schema_version,
            release_id=row.release_id,
            run_id=row.run_id,
            key=row.key,
            condition_sha256=row.identity.condition_sha256,
            status=row.status,
            metrics={name: value for name, (value, _) in scored.items()},
            metric_weights={
                metric.name: scored[metric.name][1]
                for metric in metrics
                if metric.denominator == "all_emitted_citations"
                and scored[metric.name][1] != 1
            },
        ))
    return sorted(output, key=lambda row: row.key.as_tuple())


def score_release(
    raw_path: Path,
    scored_path: Path,
    metrics: list[MetricSpec],
    cases: dict[str, CaseRecord] | None = None,
) -> list[ScoredReleaseRow]:
    """Read raw rows and write scores. This function has no generation adapter."""
    rows = read_jsonl(raw_path, RawReleaseRow)
    scored = score_rows(rows, metrics, cases)
    write_jsonl(scored_path, scored)
    return scored
