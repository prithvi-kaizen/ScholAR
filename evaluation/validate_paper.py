"""Validate EACL paper structure, claim provenance, anonymity, and submission gates."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.prepare_paper_tables import validate_seal  # noqa: E402
from evaluation.release.gates import validate_gate_registry  # noqa: E402
from evaluation.release.io import read_json, sha256_file  # noqa: E402
from evaluation.release.schemas import ClaimEvidenceRef, PrimaryGateResult  # noqa: E402
from evaluation.release.validate import validate_release_directory  # noqa: E402
from evaluation.validate_submission_pdf import (  # noqa: E402
    validate_style_provenance,
    validate_submission_pdf,
)

REQUIRED_GATE_IDS = {
    "development_claim_labels",
    "verifier_threshold_calibration",
    "paper_disjoint_heldout_test",
    "judge_validation_annotations",
    "ethics_or_irb_determination",
    "researcher_pilot",
    "model_backed_intervention_runs",
    "available_hardware_profiling",
    "official_template_installed_and_reviewed",
}
FORBIDDEN_TEXT = ("legacy_non_empirical", "paper/manuscript.tex", "manuscript/eacl2027_scholar.tex")
ANONYMITY = (re.compile(r"prithviraj", re.I), re.compile(r"github\.com/prithvi-kaizen", re.I), re.compile(r"/Users/", re.I))


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_empirical_source(source: Any) -> list[str]:
    errors: list[str] = []
    try:
        reference = ClaimEvidenceRef.model_validate(source)
    except Exception as exc:
        return [f"invalid structured empirical evidence reference: {exc}"]
    artifact = reference.artifact
    path = (ROOT / artifact.path).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError:
        return [f"empirical evidence path escapes repository: {artifact.path}"]
    if not path.is_file():
        return [f"empirical evidence artifact is missing: {artifact.path}"]
    if path.stat().st_size != artifact.bytes or sha256_file(path) != artifact.sha256:
        errors.append(f"empirical evidence hash/size differs: {artifact.path}")
    if artifact.evidence_class != "measured" or artifact.claim_status != "current":
        errors.append(f"empirical evidence is not current measured evidence: {artifact.path}")
    try:
        payload = read_json(path)
    except Exception as exc:
        errors.append(f"empirical evidence is not parseable JSON: {artifact.path}: {exc}")
        return errors
    if reference.selector != "/decision":
        errors.append(f"unsupported empirical evidence selector: {reference.selector}")
    elif reference.expected_decision == "PASS":
        try:
            result = PrimaryGateResult.model_validate(payload)
            if result.decision != "PASS":
                errors.append(f"empirical success claim points to a non-PASS gate: {artifact.path}")
        except Exception as exc:
            errors.append(f"empirical success claim lacks a primary gate result: {exc}")
    return errors


def validate_paper(paper_dir: Path, submission: bool = False) -> tuple[list[str], list[str]]:
    paper_dir = Path(paper_dir).resolve()
    errors: list[str] = []
    notices: list[str] = []
    required = (
        "main.tex", "limitations.tex", "ethics.tex", "appendix.tex", "references.bib",
        "claim_map.json", "venue_requirements.json", "README.md", "style/README.md",
        "style/official_style_manifest.json",
    )
    for name in required:
        if not (paper_dir / name).is_file():
            errors.append(f"missing paper source: {name}")
    if errors:
        return errors, notices

    main = (paper_dir / "main.tex").read_text(encoding="utf-8")
    if r"\usepackage[review]{style/acl}" not in main:
        errors.append("main.tex does not use the official ACL review package path")
    if r"\IfFileExists{../../evaluation/releases/eacl_industry_v1/tables/summary.tex}" in main:
        errors.append("main.tex must not enable results by table filename existence")
    if r"\InputIfFileExists{build/release_table_gate.tex}" not in main:
        errors.append("main.tex lacks the validator-created release-table gate")
    if r"\author{Anonymous submission}" not in main:
        errors.append("review manuscript author block is not anonymous")
    markers = [
        r"\input{sections/deployment}",
        r"\section{Conclusion}",
        r"\input{limitations}",
        r"\input{ethics}",
        r"\bibliography{references}",
        r"\appendix",
        r"\input{appendix}",
    ]
    positions = [main.find(marker) for marker in markers]
    if any(position < 0 for position in positions) or positions != sorted(positions):
        errors.append("conclusion/limitations/ethics/references/appendix ordering violates venue structure")
    abstract_match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    if not abstract_match:
        errors.append("abstract is missing")
    else:
        abstract = re.sub(r"\\\w+|[{}]", " ", abstract_match.group(1))
        words = re.findall(r"\b[\w'-]+\b", abstract)
        if len(words) > 200:
            errors.append(f"abstract exceeds 200 words: {len(words)}")

    requirements = _load(paper_dir / "venue_requirements.json")
    expected = {
        "venue": "EACL 2027 Industry Track",
        "review_content_pages": 6,
        "double_blind": True,
        "limitations_required": True,
        "appendix_position": "after bibliography in the same PDF",
        "style_modification_allowed": False,
    }
    for key, value in expected.items():
        if requirements.get(key) != value:
            errors.append(f"venue requirement is missing or wrong: {key}")

    tex_paths = sorted(paper_dir.rglob("*.tex"))
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_TEXT:
            if token in text:
                errors.append(f"paper source references forbidden prior manuscript in {path.name}: {token}")
        for pattern in ANONYMITY:
            if pattern.search(text):
                errors.append(f"paper anonymity hygiene failed in {path.name}")

    claim_map = _load(paper_dir / "claim_map.json")
    claims = claim_map.get("claims", []) if isinstance(claim_map, dict) else []
    if not claims:
        errors.append("claim map has no claims")
    for claim in claims:
        status = claim.get("status")
        claim_type = claim.get("claim_type")
        sources = claim.get("sources") or []
        if status not in {"SUPPORTED", "PENDING"}:
            errors.append(f"claim {claim.get('claim_id')} has invalid status")
        if claim_type == "empirical" and status == "SUPPORTED":
            if not sources:
                errors.append(f"empirical claim {claim.get('claim_id')} lacks release evidence")
            for source in sources:
                for source_error in _validate_empirical_source(source):
                    errors.append(f"empirical claim {claim.get('claim_id')}: {source_error}")
        if claim_type != "empirical" and status == "SUPPORTED":
            for source in sources:
                if not (ROOT / source).exists():
                    errors.append(f"claim {claim.get('claim_id')} source is missing: {source}")

    release_dir = ROOT / "evaluation/releases/eacl_industry_v1"
    gates_path = release_dir / "gates.json"
    registry, gate_errors = validate_gate_registry(gates_path)
    if registry is None:
        errors.extend(gate_errors)
        gate_items = {}
        pending = sorted(REQUIRED_GATE_IDS)
    else:
        errors.extend(f"gate: {error}" for error in gate_errors)
        gate_items = {item.id: item for item in registry.gates}
        missing_gates = REQUIRED_GATE_IDS - set(gate_items)
        if missing_gates:
            errors.append(f"release gate registry is incomplete: {sorted(missing_gates)}")
        pending = sorted(
            gate_id for gate_id, item in gate_items.items() if item.status != "CLEARED"
        )
    if pending:
        notices.append("pending external/submission gates: " + ", ".join(pending))

    errors.extend(validate_seal(paper_dir, release_dir))

    if submission:
        if pending or registry is None or registry.status != "READY":
            errors.append("submission build blocked: required release/study gates are not cleared")
        pending_claims = [claim.get("claim_id") for claim in claims if claim.get("status") != "SUPPORTED"]
        if pending_claims:
            errors.append(f"submission build blocked by pending claims: {pending_claims}")
        errors.extend(validate_style_provenance(paper_dir))
        if not (paper_dir / "build/release_table_gate.json").is_file():
            errors.append("submission build missing checksum-bound release table seal")
        release_errors = validate_release_directory(release_dir)
        errors.extend(f"release: {error}" for error in release_errors)
        errors.extend(f"pdf: {error}" for error in validate_submission_pdf(paper_dir))
    return sorted(set(errors)), notices


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper/eacl_industry")
    parser.add_argument("--submission", action="store_true")
    args = parser.parse_args()
    errors, notices = validate_paper(args.paper_dir, args.submission)
    for notice in notices:
        print(f"NOTICE: {notice}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("paper source/provenance validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
