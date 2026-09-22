#!/usr/bin/env python3
"""Freeze the exact 150-question complement of the human study for independent Luna judging.

Ensures fail-closed reference eligibility (113 eligible vs 37 ineligible)
and generates manifest.json and reference_adjudication.jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "evaluation/benchmarks/two_hundred_questions_dataset.json"
DEFAULT_RESULTS = ROOT / "evaluation/results/200_questions_benchmark_results.json"
DEFAULT_HUMAN_SELECTION = (
    ROOT / "evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters/selection_manifest.json"
)
DEFAULT_ANNOTATIONS_ZIP = (
    ROOT / "evaluation/human_eval/private/offline_50_model_trace_v7/annotation_source.zip"
)
DEFAULT_OUT_DIR = ROOT / "evaluation/results/eacl_demo_150_luna"


def sha256_text(val: str) -> str:
    return hashlib.sha256(val.encode("utf-8")).hexdigest()


def sha256_bytes(val: bytes) -> str:
    return hashlib.sha256(val).hexdigest()


def build_manifest(
    dataset_path: Path = DEFAULT_DATASET,
    results_path: Path = DEFAULT_RESULTS,
    human_selection_path: Path = DEFAULT_HUMAN_SELECTION,
    annotations_zip_path: Path = DEFAULT_ANNOTATIONS_ZIP,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    from evaluation.human_eval import notion_ground_truth

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    if not results_path.is_file():
        raise FileNotFoundError(f"Results not found: {results_path}")
    if not human_selection_path.is_file():
        raise FileNotFoundError(f"Human selection manifest not found: {human_selection_path}")
    if not annotations_zip_path.is_file():
        raise FileNotFoundError(f"Annotation zip not found: {annotations_zip_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results_data = json.loads(results_path.read_text(encoding="utf-8"))
    query_results = results_data.get("query_results", [])
    human_manifest = json.loads(human_selection_path.read_text(encoding="utf-8"))

    # Validate dataset and results size
    if len(query_results) != 200:
        raise ValueError(f"Expected 200 query results, got {len(query_results)}")

    results_by_id = {r["id"]: r for r in query_results}
    if len(results_by_id) != 200:
        raise ValueError("Duplicate IDs found in benchmark results")

    dataset_by_id = {q["id"]: q for q in dataset} if isinstance(dataset, list) else {}

    # Extract human 50 IDs
    human_cases = human_manifest.get("cases", [])
    if len(human_cases) != 50:
        raise ValueError(f"Expected 50 human cases in selection manifest, got {len(human_cases)}")
    human_ids = {c["case_id"] for c in human_cases}
    if len(human_ids) != 50:
        raise ValueError("Duplicate IDs found in human study selection")

    if not human_ids.issubset(set(results_by_id)):
        missing = human_ids - set(results_by_id)
        raise ValueError(f"Human IDs not found in results: {missing}")

    # Compute complement
    complement_ids = sorted(set(results_by_id) - human_ids)
    if len(complement_ids) != 150:
        raise ValueError(f"Expected exactly 150 complement cases, got {len(complement_ids)}")
    if set(complement_ids) & human_ids:
        raise ValueError("Fatal: overlap detected between human sample and complement cases")

    # Parse and audit Notion annotations
    annotations, zip_meta = notion_ground_truth.parse_zip(annotations_zip_path)
    audit = notion_ground_truth.audit_pairing(annotations, query_results)
    eligible_ids = audit["eligible_ids"]

    # Manifest cases and adjudication records
    manifest_cases = []
    adjudication_records = []

    for cid in complement_ids:
        row = results_by_id[cid]
        trace = row.get("trace", {})
        ds_item = dataset_by_id.get(cid, {})

        paper_id = row.get("paper_id") or ds_item.get("paper_id", "")
        question = row.get("question") or ds_item.get("question", "")
        category = ds_item.get("category") or row.get("category", "unspecified")
        difficulty = ds_item.get("difficulty") or row.get("difficulty", "unspecified")
        model_answer = trace.get("final_answer") or row.get("model_answer", "")

        citations = trace.get("citations") or row.get("citations", [])
        prompt_evidence = trace.get("prompt_evidence") or row.get("prompt_evidence", [])

        # Check visual evidence
        has_visual = any(
            c.get("identity", {}).get("modality") in ("visual", "figure", "table")
            or "bbox_normalized" in c.get("extra", {})
            or c.get("modality") in ("visual", "figure", "table")
            for c in citations
        )

        is_ref_eligible = cid in eligible_ids
        ann = annotations.get(cid, {})

        if is_ref_eligible:
            ref_status = "eligible"
            ref_answer = ann.get("answer", "").strip()
            ref_page = ann.get("page", "").strip()
            ref_evidence = ann.get("evidence", "").strip()
            adjudication = {
                "case_id": cid,
                "status": "eligible",
                "reason": "exact_question_match_nonempty_answer",
                "reference_page": ref_page,
            }
        else:
            ref_status = "ineligible"
            ref_answer = ""
            ref_page = ""
            ref_evidence = ""
            reason = "empty_answer" if not ann.get("answer", "").strip() else "question_text_mismatch"
            adjudication = {
                "case_id": cid,
                "status": "ineligible",
                "reason": reason,
                "raw_annotated_question": ann.get("question", ""),
                "action": "correctness_marked_not_judgeable_evaluate_grounding_only",
            }

        adjudication_records.append(adjudication)

        case_record = {
            "case_id": cid,
            "paper_id": paper_id,
            "question": question,
            "category": category,
            "difficulty": difficulty,
            "model_answer": model_answer,
            "citations": citations,
            "prompt_evidence": prompt_evidence,
            "has_visual_evidence": has_visual,
            "reference_status": ref_status,
            "reference_answer": ref_answer,
            "reference_page": ref_page,
            "reference_evidence": ref_evidence,
        }

        # Deterministic case payload hash
        case_hash = sha256_text(json.dumps(case_record, sort_keys=True, ensure_ascii=False))
        case_record["input_hash"] = case_hash
        manifest_cases.append(case_record)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    adjudication_path = out_dir / "reference_adjudication.jsonl"

    manifest_data = {
        "schema_version": "eacl_demo_150_luna_manifest_v1",
        "total_cases": len(manifest_cases),
        "eligible_references_count": sum(1 for c in manifest_cases if c["reference_status"] == "eligible"),
        "ineligible_references_count": sum(1 for c in manifest_cases if c["reference_status"] == "ineligible"),
        "visual_cases_count": sum(1 for c in manifest_cases if c["has_visual_evidence"]),
        "human_study_ids_count": len(human_ids),
        "source_dataset_sha256": sha256_bytes(dataset_path.read_bytes()),
        "source_results_sha256": sha256_bytes(results_path.read_bytes()),
        "annotation_zip_sha256": zip_meta["zip_sha256"],
        "cases": manifest_cases,
    }

    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")

    with adjudication_path.open("w", encoding="utf-8") as f:
        for adj in adjudication_records:
            f.write(json.dumps(adj, sort_keys=True, ensure_ascii=False) + "\n")

    return {
        "total_cases": len(manifest_cases),
        "eligible_count": manifest_data["eligible_references_count"],
        "ineligible_count": manifest_data["ineligible_references_count"],
        "manifest_path": str(manifest_path),
        "adjudication_path": str(adjudication_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze 150-case Luna judge manifest")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--human-selection", type=Path, default=DEFAULT_HUMAN_SELECTION)
    parser.add_argument("--annotations-zip", type=Path, default=DEFAULT_ANNOTATIONS_ZIP)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    summary = build_manifest(
        dataset_path=args.dataset,
        results_path=args.results,
        human_selection_path=args.human_selection,
        annotations_zip_path=args.annotations_zip,
        out_dir=args.out_dir,
    )
    print(f"Successfully generated 150-case manifest at {summary['manifest_path']}")
    print(f"Total complement cases: {summary['total_cases']}")
    print(f"Eligible paired references: {summary['eligible_count']}")
    print(f"Ineligible references (marked not_judgeable for correctness): {summary['ineligible_count']}")


if __name__ == "__main__":
    main()
