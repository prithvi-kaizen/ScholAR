import json
import pytest
from pathlib import Path

from evaluation.build_eacl_demo_150_judge_manifest import (
    DEFAULT_DATASET,
    DEFAULT_RESULTS,
    DEFAULT_HUMAN_SELECTION,
    DEFAULT_ANNOTATIONS_ZIP,
    build_manifest,
)
from evaluation.run_eacl_demo_150_luna_judge import (
    make_judge_payload,
    validate_judgment,
    run_judge,
)
from evaluation.aggregate_eacl_demo_150_luna_judge import aggregate_judgments


def test_manifest_partition_math_and_eligibility(tmp_path):
    out_dir = tmp_path / "luna_150"
    summary = build_manifest(
        dataset_path=DEFAULT_DATASET,
        results_path=DEFAULT_RESULTS,
        human_selection_path=DEFAULT_HUMAN_SELECTION,
        annotations_zip_path=DEFAULT_ANNOTATIONS_ZIP,
        out_dir=out_dir,
    )
    assert summary["total_cases"] == 150
    assert summary["eligible_count"] == 113
    assert summary["ineligible_count"] == 37

    manifest_file = Path(summary["manifest_path"])
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert manifest["total_cases"] == 150
    assert len(manifest["cases"]) == 150

    case_ids = {c["case_id"] for c in manifest["cases"]}
    assert len(case_ids) == 150

    human_manifest = json.loads(DEFAULT_HUMAN_SELECTION.read_text(encoding="utf-8"))
    human_ids = {c["case_id"] for c in human_manifest["cases"]}
    assert len(human_ids) == 50
    assert len(case_ids & human_ids) == 0


def test_judge_payload_and_validation(tmp_path):
    rubric_path = Path("evaluation/protocols/eacl_demo_150_llm_judge.json")
    rubric = json.loads(rubric_path.read_text(encoding="utf-8"))

    eligible_case = {
        "case_id": "TEST_Q1",
        "question": "What is the optimizer?",
        "reference_status": "eligible",
        "reference_answer": "Adam with beta1=0.9",
        "model_answer": "The optimizer used is Adam.",
        "prompt_evidence": [{"page": 3, "section": "Methods", "quote": "We use Adam"}],
        "citations": [{"page": 3, "quote": "We use Adam"}],
        "has_visual_evidence": False,
    }
    payload = make_judge_payload(eligible_case)
    assert payload["reference_status"] == "eligible"
    assert "Adam with beta1=0.9" in payload["reference_answer"]

    valid_judgment = {
        "correctness": "correct",
        "completeness": "complete",
        "grounding": "supported",
        "citation_quality": "accurate",
        "visual_grounding": "not_applicable",
        "failure_type": "none",
        "brief_reason": "Answer matches reference and is supported by excerpts.",
    }
    errors = validate_judgment(valid_judgment, rubric, is_ref_eligible=True)
    assert not errors

    ineligible_case = {
        "case_id": "TEST_Q2",
        "question": "What is the dataset size?",
        "reference_status": "ineligible",
        "reference_answer": "",
        "model_answer": "10,000 samples",
        "citations": [],
    }
    ineligible_payload = make_judge_payload(ineligible_case)
    assert ineligible_payload["reference_status"] == "ineligible"

    # Testing fail-closed rule: correctness must be not_judgeable if reference ineligible
    invalid_judgment = dict(valid_judgment)
    invalid_judgment["correctness"] = "correct"
    errors = validate_judgment(invalid_judgment, rubric, is_ref_eligible=False)
    assert any("correctness must be not_judgeable" in e for e in errors)

    invalid_judgment["correctness"] = "not_judgeable"
    invalid_judgment["visual_grounding"] = "not_applicable"
    assert not validate_judgment(invalid_judgment, rubric, is_ref_eligible=False)

    visual_case_errors = validate_judgment(
        {**valid_judgment, "visual_grounding": "not_applicable"},
        rubric,
        is_ref_eligible=True,
        has_visual_evidence=True,
        visual_material_supplied=False,
    )
    assert any("must be not_judgeable" in error for error in visual_case_errors)


def test_judge_payload_uses_saved_source_text_without_claiming_visual_input():
    case = {
        "case_id": "VISUAL_TEXT_Q1",
        "question": "What trend does the paper report?",
        "reference_status": "eligible",
        "reference_answer": "The value increases.",
        "model_answer": "The value increases.",
        "prompt_evidence": [],
        "citations": [{
            "page": 4,
            "quote": "",
            "extra": {
                "text": "The measured value increases across all settings.",
                "visual_observation": "The line rises sharply.",
            },
        }],
        "has_visual_evidence": True,
    }

    payload = make_judge_payload(case)
    assert payload["source_excerpts"] == [{
        "page": 4,
        "section": None,
        "quote": "The measured value increases across all settings.",
    }]
    assert payload["citations"][0]["quote"] == "The measured value increases across all settings."
    assert payload["visual_material_supplied"] is False
    assert "visual_observation" not in json.dumps(payload)


def test_dry_run_and_aggregator(tmp_path):
    manifest_path = Path("evaluation/results/eacl_demo_150_luna/manifest.json")
    rubric_path = Path("evaluation/protocols/eacl_demo_150_llm_judge.json")
    output_path = tmp_path / "judgments.jsonl"

    res = run_judge(
        manifest_path=manifest_path,
        output_path=output_path,
        rubric_path=rubric_path,
        dry_run=True,
    )
    assert res["status"] == "DRY_RUN_COMPLETED"
    assert res["total_cases"] == 150
    assert res["pending_cases"] == 150

    public_out = tmp_path / "public.json"
    receipt = aggregate_judgments(manifest_path, output_path, public_out)
    assert receipt["status"] == "PENDING"
    assert receipt["total_attempted_cases"] == 150
    assert receipt["completed_cases"] == 0
    assert receipt["reference_correctness_denominator"] == 113
    assert receipt["visual_evidence_cases"] == 90
    assert receipt["visual_grounding_judgeable_denominator"] == 90
    assert receipt["visual_grounding_not_judgeable_count"] == 0


def test_export_and_import_codex_batches(tmp_path):
    from evaluation.export_luna_codex_batches import export_batches
    from evaluation.import_luna_codex_results import import_codex_results

    manifest_path = Path("evaluation/results/eacl_demo_150_luna/manifest.json")
    rubric_path = Path("evaluation/protocols/eacl_demo_150_llm_judge.json")
    batch_dir = tmp_path / "batches"

    exp_res = export_batches(
        manifest_path=manifest_path,
        rubric_path=rubric_path,
        out_dir=batch_dir,
        batch_size=15,
    )
    assert exp_res["status"] == "BATCHES_EXPORTED"
    assert exp_res["num_batches"] == 10
    assert (batch_dir / "prompt_batch_01.txt").is_file()
    assert (batch_dir / "cases_batch_01.json").is_file()

    # Create mock returns for 2 cases: one eligible, one ineligible
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    eligible_case = next(c for c in manifest["cases"] if c.get("reference_status") == "eligible")
    ineligible_case = next(c for c in manifest["cases"] if c.get("reference_status") == "ineligible")
    eligible_visual = bool(eligible_case.get("has_visual_evidence"))
    ineligible_visual = bool(ineligible_case.get("has_visual_evidence"))

    mock_return = [
        {
            "case_id": eligible_case["case_id"],
            "correctness": "correct",
            "completeness": "complete",
            "grounding": "supported",
            "citation_quality": "accurate",
            "visual_grounding": "not_judgeable" if eligible_visual else "not_applicable",
            "failure_type": "none",
            "brief_reason": "Fully grounded and matches reference.",
        },
        {
            "case_id": ineligible_case["case_id"],
            "correctness": "not_judgeable",
            "completeness": "complete",
            "grounding": "supported",
            "citation_quality": "accurate",
            "visual_grounding": "not_judgeable" if ineligible_visual else "not_applicable",
            "failure_type": "none",
            "brief_reason": "Direct evidence verified.",
        },
    ]

    mock_file = tmp_path / "mock_return.json"
    mock_file.write_text(json.dumps(mock_return), encoding="utf-8")

    judgments_path = tmp_path / "judgments.jsonl"
    public_path = tmp_path / "public.json"

    imp_res = import_codex_results(
        input_paths=[mock_file],
        manifest_path=manifest_path,
        rubric_path=rubric_path,
        judgments_path=judgments_path,
        public_output_path=public_path,
    )

    assert imp_res["new_cases"] == 2
    assert imp_res["total_completed"] == 2
    assert imp_res["status"] == "PARTIAL"

    # Verify fail-closed enforcement
    saved_lines = [json.loads(line) for line in judgments_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    saved_by_id = {s["case_id"]: s for s in saved_lines}

    assert saved_by_id[eligible_case["case_id"]]["judgment"]["correctness"] == "correct"
    assert saved_by_id[ineligible_case["case_id"]]["judgment"]["correctness"] == "not_judgeable"


def test_visual_batches_export_and_update(tmp_path):
    from evaluation.export_luna_visual_batches import export_visual_batches
    from evaluation.update_luna_visual_judgments import update_visual_judgments

    manifest_path = Path("evaluation/results/eacl_demo_150_luna/manifest.json")
    batch_dir = tmp_path / "visual_batches"

    exp = export_visual_batches(
        manifest_path=manifest_path,
        out_dir=batch_dir,
        batch_size=15,
    )
    assert exp["status"] == "VISUAL_BATCHES_EXPORTED"
    assert exp["total_visual_cases"] == 90
    assert exp["num_batches"] == 6
    assert (batch_dir / "prompt_visual_batch_01.txt").is_file()

    # Test update with mock visual return
    mock_visual = [
        {
            "case_id": "ADAM_Q12",
            "visual_grounding": "supported",
            "visual_reason": "Figure 1 shows the convergence curves supporting the statement.",
        }
    ]
    mock_file = tmp_path / "mock_visual.json"
    mock_file.write_text(json.dumps(mock_visual), encoding="utf-8")

    judgments_path = Path("evaluation/results/eacl_demo_150_luna/judgments.jsonl")
    public_path = tmp_path / "public_vis.json"

    # Copy judgments to tmp to avoid mutating real data in unit test
    tmp_judgments = tmp_path / "judgments.jsonl"
    tmp_judgments.write_text(judgments_path.read_text(encoding="utf-8"), encoding="utf-8")

    res = update_visual_judgments(
        input_paths=[mock_file],
        manifest_path=manifest_path,
        judgments_path=tmp_judgments,
        public_output_path=public_path,
    )
    assert res["status"] == "UPDATED"
    assert res["updated_cases"] == 1

