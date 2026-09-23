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
from evaluation.release.schemas import CaseRecord, ExpectedKeySet, ScoredReleaseRow
from evaluation.statistics import clustered_mean_interval, paired_cluster_analysis


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _weighted_mean(values: list[tuple[float, int]]) -> float | None:
    denominator = sum(weight for _, weight in values)
    if denominator == 0:
        return None
    return round(sum(value * weight for value, weight in values) / denominator, 6)


def aggregate_rows(
    rows: list[ScoredReleaseRow],
    expected: ExpectedKeySet,
    metric_names: list[str],
    *,
    cases: dict[str, CaseRecord] | None = None,
    statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    indexed = key_index(rows)
    expected_tuples = {key.as_tuple() for key in expected.keys}
    if set(indexed) != expected_tuples:
        missing = sorted(expected_tuples - set(indexed))
        extra = sorted(set(indexed) - expected_tuples)
        raise ValueError(f"scored key universe mismatch; missing={missing[:3]}, extra={extra[:3]}")

    by_case: dict[tuple[str, str, str], list[ScoredReleaseRow]] = defaultdict(list)
    weighted_metric_names = {
        name for row in rows for name in row.metric_weights
    }
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
                name: _weighted_mean([
                    (row.metrics[name], row.metric_weights.get(name, 1))
                    for row in repetitions if row.metrics.get(name) is not None
                ]) for name in metric_names
            },
            "metric_denominators": {
                name: sum(
                    row.metric_weights.get(name, 1)
                    for row in repetitions if row.metrics.get(name) is not None
                )
                for name in metric_names
            },
        })

    groups: list[dict[str, Any]] = []
    group_keys = sorted({(row["system"], row["model"]) for row in case_rows})
    for system, model in group_keys:
        group_cases = [
            row for row in case_rows
            if row["system"] == system and row["model"] == model
        ]
        raw_group = [row for row in rows if row.key.system == system and row.key.model == model]
        groups.append({
            "system": system,
            "model": model,
            "n_expected": len(raw_group),
            "n_cases": len(group_cases),
            "status_counts": dict(sorted(Counter(row.status.value for row in raw_group).items())),
            "metrics": {
                name: _weighted_mean([
                    (
                        row["metrics"][name],
                        row["metric_denominators"][name] if name in weighted_metric_names else 1,
                    )
                    for row in group_cases if row["metrics"][name] is not None
                ])
                for name in metric_names
            },
            "metric_case_denominators": {
                name: sum(row["metrics"][name] is not None for row in group_cases)
                for name in metric_names
            },
        })

    summary = {
        "schema_version": expected.schema_version,
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
    if statistics is not None:
        if cases is None:
            raise ValueError("paper-clustered statistics require frozen case records")
        samples = int(statistics.get("bootstrap_samples", 10000))
        seed = int(statistics.get("bootstrap_seed", 2027))
        confidence = float(statistics.get("confidence_level", 0.95))
        group_intervals: list[dict[str, Any]] = []
        for group in groups:
            for name in metric_names:
                values: dict[str, list[float]] = defaultdict(list)
                for row in case_rows:
                    if row["system"] != group["system"] or row["model"] != group["model"]:
                        continue
                    value = row["metrics"].get(name)
                    if value is not None:
                        weight = max(1, int(row["metric_denominators"].get(name, 1)))
                        values[cases[row["case_id"]].paper_id].extend(
                            [float(value)] * weight
                        )
                interval = clustered_mean_interval(
                    dict(values), samples=samples, seed=seed, confidence_level=confidence
                )
                group_intervals.append({
                    "system": group["system"], "model": group["model"], "metric": name, **interval
                })
        summary["paper_clustered_analysis"] = {
            "independent_cluster": "paper",
            "bootstrap_samples": samples,
            "bootstrap_seed": seed,
            "confidence_level": confidence,
            "group_intervals": group_intervals,
            "paired_comparisons": paired_cluster_analysis(
                case_rows,
                cases,
                metric_names,
                list(statistics.get("paired_comparisons", [])),
                samples=samples,
                seed=seed,
                confidence_level=confidence,
            ),
            "multiplicity_correction": "Holm-Bonferroni",
        }
    return summary


def aggregate_release(
    scored_path: Path,
    expected_path: Path,
    aggregate_path: Path,
    metric_names: list[str],
    *,
    cases: dict[str, CaseRecord] | None = None,
    statistics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(scored_path, ScoredReleaseRow)
    expected = ExpectedKeySet.model_validate(read_json(expected_path))
    summary = aggregate_rows(
        rows, expected, metric_names, cases=cases, statistics=statistics
    )
    write_json(aggregate_path, summary)
    return summary


def _latex_escape(value: object) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def render_tables(
    aggregate_path: Path,
    tables_dir: Path,
    *,
    release_dir: Path | None = None,
) -> list[Path]:
    """Render tables; schema-v2 measured outputs require a passed human gate."""
    aggregate_path = Path(aggregate_path)
    summary = read_json(aggregate_path)
    required = {"release_id", "run_id", "metric_names", "groups"}
    if not isinstance(summary, dict) or not required.issubset(summary):
        raise ValueError("aggregate JSON does not have the release-v1 summary shape")
    manifest_identity: str | None = None
    primary_gate_hash: str | None = None
    if summary.get("schema_version") == "2.0":
        if release_dir is None:
            raise ValueError("schema-v2 table rendering requires the release directory")
        from evaluation.release.manifest import load_manifest, manifest_identity_sha256
        from evaluation.release.human_scoring import score_primary_gate
        from evaluation.release.schemas import (
            HumanEvaluationBundle,
            PairedComparisonSpec,
            PrimaryGateResult,
            RawReleaseRow,
        )

        release_dir = Path(release_dir)
        manifest = load_manifest(release_dir / "manifest.json")
        if manifest.evidence_class != "measured" or manifest.claim_status != "current":
            raise ValueError("schema-v2 claim-bearing tables require a current measured release")
        gate_path = release_dir / "human/primary_gate.json"
        if not gate_path.is_file():
            raise ValueError("schema-v2 table rendering requires the independent-human primary gate")
        primary_gate = PrimaryGateResult.model_validate(read_json(gate_path))
        if (
            primary_gate.decision != "PASS"
            or primary_gate.evidence_class != "measured"
            or primary_gate.claim_status != "current"
        ):
            raise ValueError("schema-v2 claim-bearing tables require a current measured PASS primary gate")
        annotations_path = release_dir / "human/annotations.json"
        spec_path = release_dir / "human/paired_gate_spec.json"
        if not annotations_path.is_file() or not spec_path.is_file():
            raise ValueError("schema-v2 table rendering requires frozen human-gate inputs")
        reproduced = score_primary_gate(
            HumanEvaluationBundle.model_validate(read_json(annotations_path)),
            PairedComparisonSpec.model_validate(read_json(spec_path)),
            read_jsonl(release_dir / "raw/rows.jsonl", RawReleaseRow),
        )
        if reproduced != primary_gate:
            raise ValueError("schema-v2 human primary gate does not reproduce from frozen inputs")
        manifest_identity = manifest_identity_sha256(manifest)
        primary_gate_hash = sha256_file(gate_path)
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
        "schema_version": summary["schema_version"],
        "release_id": summary["release_id"],
        "run_id": summary["run_id"],
        "source": aggregate_path.name,
        "aggregate_sha256": aggregate_hash,
        "tables": ["summary.csv", "summary.tex"],
    }
    if manifest_identity is not None and primary_gate_hash is not None:
        provenance.update({
            "release_manifest_identity_sha256": manifest_identity,
            "primary_gate_sha256": primary_gate_hash,
            "human_annotations_sha256": sha256_file(release_dir / "human/annotations.json"),
            "paired_gate_spec_sha256": sha256_file(release_dir / "human/paired_gate_spec.json"),
        })
    provenance_path = tables_dir / "provenance.json"
    write_json(provenance_path, provenance)
    return [csv_path, tex_path, provenance_path]
