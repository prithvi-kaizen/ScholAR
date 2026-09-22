"""Model-free smoke test for the ScholAR EACL 2027 Demo release.

Verifies that the backend API, dual-engine parser, sample paper, and deterministic
verifier operate correctly without requiring a downloaded LLM or external GPU.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_backend_app_and_routes():
    from backend.main import app
    assert app.title == "ScholAR API"

    route_paths = {route.path for route in app.routes}
    assert "/health" in route_paths
    assert "/api/system/network-policy" in route_paths
    assert "/api/papers/upload" in route_paths
    assert "/api/retrieval/search" in route_paths
    assert "/api/models" in route_paths


def test_sample_paper_parsing_and_layout():
    import fitz
    sample_pdf = ROOT / "examples/eacl_demo/sample_paper.pdf"
    assert sample_pdf.is_file(), f"Sample PDF missing at {sample_pdf}"

    doc = fitz.open(sample_pdf)
    assert len(doc) == 2, f"Expected 2 pages in sample paper, found {len(doc)}"

    p1_text = doc[0].get_text()
    assert "Dynamic Evidence Fusion" in p1_text
    assert "Table 1" in p1_text
    assert "Figure 1" in p1_text

    # Verify text extraction returns valid bounding rectangles
    blocks = doc[0].get_text("blocks")
    assert len(blocks) > 5, "Expected structured layout blocks on page 1"


def test_questions_and_expected_evidence_consistency():
    questions_path = ROOT / "examples/eacl_demo/QUESTIONS.md"
    license_path = ROOT / "examples/eacl_demo/LICENSE.md"
    evidence_path = ROOT / "examples/eacl_demo/EXPECTED_EVIDENCE.json"

    assert questions_path.is_file()
    assert license_path.is_file()
    assert evidence_path.is_file()

    with evidence_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["total_pages"] == 2
    cases = meta["cases"]
    assert len(cases) == 3

    case_types = {c["element_type"] for c in cases}
    assert case_types == {"text", "table", "figure"}


def test_deterministic_verifier_component():
    from backend.services.verifier_service import LexicalSupportScorer, VerificationLabel

    scorer = LexicalSupportScorer()
    source_chunk = "We train our model using Adam with beta1=0.9, beta2=0.999, and a learning rate of 2e-4."
    claim_supported = "The model was trained with Adam and an initial learning rate of 2e-4."

    result = scorer.score(
        claim_id="C1",
        claim_text=claim_supported,
        evidence_texts=[source_chunk],
        modality="text",
        cited_evidence_ids=["E1"],
    )
    assert result.label in {VerificationLabel.SUPPORTED, VerificationLabel.PARTIAL}
    assert result.confidence > 0.4


def test_demo_install_docs_present():
    demo_install = ROOT / "DEMO_INSTALL.md"
    notices = ROOT / "THIRD_PARTY_NOTICES.md"
    assert demo_install.is_file()
    assert notices.is_file()

    content = demo_install.read_text(encoding="utf-8")
    assert "ScholAR_EACL2027_Demo_v1.0.0.zip" in content
    assert "make demo-setup" in content
    assert "make demo-model" in content
    assert "make demo-run" in content
