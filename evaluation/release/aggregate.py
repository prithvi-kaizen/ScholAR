"""Deterministic case-balanced aggregation and aggregate-only table rendering."""

from __future__ import annotations

import csv
import io
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evaluation.release.io import (
    key_index,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from evaluation.release.schemas import ExpectedKeySet, ScoredReleaseRow


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def aggregate_rows(
    rows: list[ScoredReleaseRow], expected: ExpectedKeySet, metric_names: list[str]
) -> dict[str, Any]:
    indexed = key_index(rows)
    expected_tuples = {key.as_tuple() for key in expected.keys}
    if set(indexed) != expected_tuples:
        missing = sorted(expected_tuples - set(indexed))
        extra = sorted(set(indexed) - expected_tuples)
        raise ValueError(f"scored key universe mismatch; missing={missing[:3]}, extra={extra[:3]}")

    by_case: dict[tuple[str, str, str], list[ScoredReleaseRow]] = defaultdict(list)
    for row in rows:
        by_case[(row.key.system, row.key.model, row.key.case_id)].append(row)

    case_rows: list[dict[str, Any]] = []
    for (system, model, case_id), repetitions in sorted(by_case.items()):
        repetitions.sort(key=lambda row: row.key.seed)
        case_rows.append({
            "system": system,
            "model": model,
            "case_id": case_id,
            "n_expected_repetitions": len(repetitions),
            "status_counts": dict(sorted(Counter(row.status.value for row in repetitions).items())),
            "metrics": {
                name: _mean([row.metrics[name] for row in repetitions if row.metrics.get(name) is not None])
                for name in metric_names
            },
            "metric_denominators": {
                name: sum(row.metrics.get(name) is not None for row in repetitions)
                for name in metric_names
            },
        })

    groups: list[dict[str, Any]] = []
    group_keys = sorted({(row["system"], row["model"]) for row in case_rows})
    for system, model in group_keys:
        cases = [row for row in case_rows if row["system"] == system and row["model"] == model]
        raw_group = [row for row in rows if row.key.system == system and row.key.model == model]
        groups.append({
            "system": system,
            "model": model,
            "n_expected": len(raw_group),
            "n_cases": len(cases),
            "status_counts": dict(sorted(Counter(row.status.value for row in raw_group).items())),
            "metrics": {
                name: _mean([row["metrics"][name] for row in cases if row["metrics"][name] is not None])
                for name in metric_names
            },
            "metric_case_denominators": {
                name: sum(row["metrics"][name] is not None for row in cases)
                for name in metric_names
            },
        })

    return {
        "schema_version": "1.0",
        "release_id": expected.release_id,
        "run_id": expected.run_id,
        "aggregation_order": ["within_answer", "across_seeds_within_case", "across_cases"],
        "n_expected": expected.n_expected,
        "n_scored": len(rows),
        "status_counts": dict(sorted(Counter(row.status.value for row in rows).items())),
        "metric_names": metric_names,
        "case_rows": case_rows,
        "groups": groups,
    }


def aggregate_release(
    scored_path: Path,
    expected_path: Path,
    aggregate_path: Path,
    metric_names: list[str],
) -> dict[str, Any]:
    rows = read_jsonl(scored_path, ScoredReleaseRow)
    expected = ExpectedKeySet.model_validate(read_json(expected_path))
    summary = aggregate_rows(rows, expected, metric_names)
    write_json(aggregate_path, summary)
    return summary


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def render_tables(aggregate_path: Path, tables_dir: Path) -> list[Path]:
    """Render CSV/TeX using only aggregate JSON as input."""
    aggregate_path = Path(aggregate_path)
    summary = read_json(aggregate_path)
    required = {"release_id", "run_id", "metric_names", "groups"}
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise ValueError("aggregate JSON does not have the release-v1 summary shape")
    aggregate_hash = sha256_file(aggregate_path)
    metric_names = list(summary["metric_names"])
    groups = list(summary["groups"])
    tables_dir.mkdir(parents=True, exist_ok=True)

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["release_id", "run_id", "aggregate_sha256", "system", "model", "n_expected", "n_cases", *metric_names])
    for group in groups:
        writer.writerow([
            summary["release_id"], summary["run_id"], aggregate_hash,
            group["system"], group["model"], group["n_expected"], group["n_cases"],
            *[group["metrics"].get(name) for name in metric_names],
        ])
    csv_path = tables_dir / "summary.csv"
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")

    columns = "llrr" + "r" * len(metric_names)
    header = ["System", "Model", "N", "Cases", *metric_names]
    latex = [
        f"% release_id={summary['release_id']}",
        f"% run_id={summary['run_id']}",
        f"% aggregate_sha256={aggregate_hash}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\toprule",
        " & ".join(_latex_escape(item) for item in header) + r" \\",
        "\\midrule",
    ]
    for group in groups:
        values = [group["system"], group["model"], group["n_expected"], group["n_cases"]]
        values.extend("--" if group["metrics"].get(name) is None else f"{group['metrics'][name]:.3f}" for name in metric_names)
        latex.append(" & ".join(_latex_escape(item) for item in values) + r" \\")
    latex.extend(["\\bottomrule", "\\end{tabular}", ""])
    tex_path = tables_dir / "summary.tex"
    tex_path.write_text("\n".join(latex), encoding="utf-8")

    provenance = {
        "schema_version": "1.0",
        "release_id": summary["release_id"],
        "run_id": summary["run_id"],
        "source": aggregate_path.name,
        "aggregate_sha256": aggregate_hash,
        "tables": ["summary.csv", "summary.tex"],
    }
    provenance_path = tables_dir / "provenance.json"
    write_json(provenance_path, provenance)
    return [csv_path, tex_path, provenance_path]
