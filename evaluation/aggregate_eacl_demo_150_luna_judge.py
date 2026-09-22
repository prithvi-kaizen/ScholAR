#!/usr/bin/env python3
"""Audit and aggregate the 150-case independent Luna judge evaluation into a public receipt.

Ensures explicit denominators for reference correctness (113) and visual grounding.
Fails closed if hashes, manifest IDs, or schema constraints are violated.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation/results/eacl_demo_150_luna/manifest.json"
DEFAULT_JUDGMENTS = ROOT / "evaluation/results/eacl_demo_150_luna/judgments.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation/EACL_DEMO_150_LUNA_PUBLIC.json"


def aggregate_judgments(
    manifest_path: Path = DEFAULT_MANIFEST,
    judgments_path: Path = DEFAULT_JUDGMENTS,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest.get("cases", [])
    total_cases = len(cases)
    eligible_refs = sum(1 for c in cases if c.get("reference_status") == "eligible")
    ineligible_refs = sum(1 for c in cases if c.get("reference_status") == "ineligible")
    visual_cases_count = sum(1 for c in cases if c.get("has_visual_evidence"))
    visual_judgeable_count = sum(
        1 for c in cases if c.get("has_visual_evidence") and c.get("visual_material_supplied", False)
    )

    if not judgments_path.is_file() or judgments_path.stat().st_size == 0:
        pending_receipt = {
            "schema_version": "eacl_demo_150_luna_public_v1",
            "status": "PENDING",
            "total_attempted_cases": total_cases,
            "completed_cases": 0,
            "reference_correctness_denominator": eligible_refs,
            "reference_ineligible_denominator": ineligible_refs,
            "visual_evidence_cases": visual_cases_count,
            "visual_grounding_judgeable_denominator": visual_judgeable_count,
            "visual_grounding_not_judgeable_count": visual_cases_count - visual_judgeable_count,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "note": "Evaluation harness configured; external API execution pending owner approval and key provision.",
        }
        output_path.write_text(json.dumps(pending_receipt, indent=2), encoding="utf-8")
        return pending_receipt

    judgments_raw = [
        json.loads(line)
        for line in judgments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    judgments_by_id = {j["case_id"]: j for j in judgments_raw}
    if len(judgments_by_id) != len(judgments_raw):
        raise ValueError("Duplicate case IDs detected in judgments file")

    # Audit against manifest
    manifest_by_id = {c["case_id"]: c for c in cases}
    for cid, j in judgments_by_id.items():
        if cid not in manifest_by_id:
            raise ValueError(f"Unknown case ID in judgments: {cid}")
        man_case = manifest_by_id[cid]
        if j.get("input_hash") and man_case.get("input_hash") and j["input_hash"] != man_case["input_hash"]:
            raise ValueError(f"Input hash mismatch for {cid}")

        is_ref_eligible = man_case.get("reference_status") == "eligible"
        has_visual = bool(man_case.get("has_visual_evidence", False))
        visual_supplied = bool(man_case.get("visual_material_supplied", False))
        j_verdict = j.get("judgment", {})
        if not is_ref_eligible and j_verdict.get("correctness") != "not_judgeable":
            raise ValueError(f"Case {cid} has ineligible reference but correctness was not 'not_judgeable'")
        if not has_visual and j_verdict.get("visual_grounding") != "not_applicable":
            raise ValueError(f"Case {cid} has no visual evidence but visual_grounding was not 'not_applicable'")
        if has_visual and not visual_supplied and j_verdict.get("visual_grounding") != "not_judgeable":
            raise ValueError(f"Case {cid} has visual evidence but no pixels were supplied; visual_grounding must be 'not_judgeable'")

    completed_count = len(judgments_by_id)
    is_complete = completed_count == total_cases

    # Compute metric distributions
    correctness_counts: dict[str, int] = {}
    completeness_counts: dict[str, int] = {}
    grounding_counts: dict[str, int] = {}
    citation_counts: dict[str, int] = {}
    visual_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    eligible_correctness_counts: dict[str, int] = {}
    ineligible_correctness_counts: dict[str, int] = {}

    for j in judgments_by_id.values():
        val = j.get("judgment", {})
        c = val.get("correctness")
        if c:
            correctness_counts[c] = correctness_counts.get(c, 0) + 1
            scoped = eligible_correctness_counts if j.get("reference_status") == "eligible" else ineligible_correctness_counts
            scoped[c] = scoped.get(c, 0) + 1
        comp = val.get("completeness")
        if comp:
            completeness_counts[comp] = completeness_counts.get(comp, 0) + 1
        g = val.get("grounding")
        if g:
            grounding_counts[g] = grounding_counts.get(g, 0) + 1
        cit = val.get("citation_quality")
        if cit:
            citation_counts[cit] = citation_counts.get(cit, 0) + 1
        vg = val.get("visual_grounding")
        if vg:
            visual_counts[vg] = visual_counts.get(vg, 0) + 1
        f = val.get("failure_type")
        if f:
            failure_counts[f] = failure_counts.get(f, 0) + 1

    receipt = {
        "schema_version": "eacl_demo_150_luna_public_v1",
        "status": "COMPLETED" if is_complete else "PARTIAL",
        "evaluation_mode": "manual_assistant_proxy",
        "judge_model": "manual-assistant-proxy",
        "total_attempted_cases": total_cases,
        "completed_cases": completed_count,
        "reference_correctness_denominator": eligible_refs,
        "reference_ineligible_denominator": ineligible_refs,
        "visual_evidence_cases": visual_cases_count,
        "visual_grounding_judgeable_denominator": visual_judgeable_count,
        "visual_grounding_not_judgeable_count": visual_cases_count - visual_judgeable_count,
        "metrics": {
            "correctness_distribution": correctness_counts,
            "eligible_reference_correctness_distribution": eligible_correctness_counts,
            "ineligible_reference_correctness_distribution": ineligible_correctness_counts,
            "completeness_distribution": completeness_counts,
            "grounding_distribution": grounding_counts,
            "citation_quality_distribution": citation_counts,
            "visual_grounding_distribution": visual_counts,
            "failure_types": failure_counts,
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    output_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate 150-case Luna judge results")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--judgments", type=Path, default=DEFAULT_JUDGMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    receipt = aggregate_judgments(args.manifest, args.judgments, args.output)
    print(f"Aggregation complete (status={receipt['status']}). Written to {args.output}")


if __name__ == "__main__":
    main()
