"""Validate EACL paper structure, claim provenance, anonymity, and submission gates."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.prepare_paper_tables import validate_seal  # noqa: E402
from evaluation.protocol_governance import validate_protocol  # noqa: E402
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
ANONYMITY = (
    re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/", re.I),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\", re.I),
)
MEASURED_FIGURE_GENERATORS = ("pareto_frontier", "risk_coverage", "ablation_study", "retrieval_recall")
CLAIM_FIGURE_STEMS = MEASURED_FIGURE_GENERATORS + (
    "multilevel_reasoning_radar",
    "pipeline_latency_breakdown",
    "verification_ladder_ablation",
)
OBSERVATION_NAMES = re.compile(
    r"(?:result|score|metric|latenc|recall|precision|accuracy|unsupported|value|point|system|variant)",
    re.I,
)
SYNTHETIC_CALLS = {"arange", "geomspace", "linspace", "logspace", "normal", "random", "uniform"}


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


def hard_coded_observation_errors(path: Path) -> list[str]:
    """Reject module-level empirical-looking arrays and generated synthetic curves."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"cannot inspect claim-bearing figure source {path.name}: {exc}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else ""
            )
            if name in SYNTHETIC_CALLS:
                errors.append(f"{path.name}:{node.lineno} generates synthetic numeric observations with {name}()")
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        if not any(OBSERVATION_NAMES.search(name) for name in names):
            continue
        if isinstance(value, (ast.List, ast.Tuple, ast.Set, ast.Dict)) and any(
            isinstance(item, (ast.Constant, ast.UnaryOp))
            and isinstance(getattr(item, "value", None), (int, float))
            for item in ast.walk(value)
        ):
            errors.append(
                f"{path.name}:{node.lineno} contains module-level hard-coded numeric observations"
            )
    return errors


def _validate_empirical_inclusion_contract(paper_dir: Path) -> list[str]:
    """Ensure claim-bearing prose, tables, and figures fail closed."""
    errors: list[str] = []
    results = (paper_dir / "sections/results.tex").read_text(encoding="utf-8")
    appendix = (paper_dir / "appendix.tex").read_text(encoding="utf-8")
    # The narrower retrospective paper reports only the audited Phase 2 machine
    # results. The separate human-evidence release remains fail-closed for any
    # claim of semantic support or user benefit.
    retrospective = r"\label{tab:retrieval}" in results and r"\label{tab:repair}" in results
    if retrospective:
        if not (ROOT / "evaluation/PHASE2_RESULTS.md").is_file():
            errors.append("retrospective paper lacks its tracked audited-results receipt")
        for phrase in ("Retrospective", "automatic proxies", "human", "paper-clustered"):
            if phrase.casefold() not in results.casefold():
                errors.append(f"retrospective results omit required scope qualifier: {phrase}")
    else:
        required_results = (
            r"\ifreleasetables",
            r"\input{../../evaluation/releases/eacl_industry_v1/tables/summary.tex}",
            r"\input{../../evaluation/releases/eacl_industry_v1/tables/results_narrative.tex}",
            r"\GatePending{",
        )
        for token in required_results:
            if token not in results:
                errors.append(f"results.tex lacks required fail-closed evidence contract: {token}")
    required_appendix = (
        r"\ifreleasetables",
        r"\input{../../evaluation/releases/eacl_industry_v1/tables/appendix_results.tex}",
    )
    for token in required_appendix:
        if token not in appendix:
            errors.append(f"appendix.tex lacks required fail-closed evidence contract: {token}")

    for path in (paper_dir / "main.tex", paper_dir / "sections/results.tex", paper_dir / "appendix.tex"):
        text = path.read_text(encoding="utf-8")
        for stem in CLAIM_FIGURE_STEMS:
            if re.search(rf"\\includegraphics(?:\[[^\]]*\])?\{{[^}}]*{re.escape(stem)}", text):
                errors.append(f"{path.name} directly includes claim-bearing figure {stem}")

    for stem in MEASURED_FIGURE_GENERATORS:
        script = paper_dir / "figs" / f"{stem}.py"
        if not script.is_file():
            errors.append(f"missing measured-data figure generator: figs/{stem}.py")
            continue
        source = script.read_text(encoding="utf-8")
        if "load_measured_figure_data" not in source:
            errors.append(f"figure generator is not release-bound: figs/{stem}.py")
        errors.extend(hard_coded_observation_errors(script))
    legacy_generator = paper_dir / "figures/generate_paper_figures.py"
    if (
        legacy_generator.is_file()
        and "LEGACY_FIGURE_ENTRYPOINT_RETIRED = True"
        not in legacy_generator.read_text(encoding="utf-8")
    ):
        errors.append("legacy figure generator is not retired")
    return errors


def validate_paper(paper_dir: Path, submission: bool = False) -> tuple[list[str], list[str]]:
    paper_dir = Path(paper_dir).resolve()
    errors: list[str] = []
    notices: list[str] = []
    required = (
        "main.tex", "limitations.tex", "ethics.tex", "appendix.tex", "references.bib",
        "sections/related_work.tex",
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
        r"\input{sections/introduction}",
        r"\input{sections/related_work}",
        r"\input{sections/system}",
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
        errors.append(
            "introduction/related-work/system/deployment/conclusion/limitations/ethics/"
            "references/appendix ordering violates venue structure"
        )
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

    protocol_path = ROOT / "evaluation/protocols/eacl_industry_v1_protocol.json"
    if not protocol_path.is_file():
        errors.append("EACL evaluation protocol is missing")
    else:
        protocol = _load(protocol_path)
        protocol_errors = validate_protocol(protocol, require_frozen=submission)
        errors.extend(f"protocol: {error}" for error in protocol_errors)
        if protocol.get("status") != "FROZEN":
            notices.append("EACL evaluation protocol remains DRAFT")

    tex_paths = sorted(paper_dir.rglob("*.tex"))
    for path in tex_paths:
        text = path.read_text(encoding="utf-8")
        if "??" in text:
            errors.append(f"unresolved reference marker in {path.relative_to(paper_dir)}")
        if re.search(r"\bNLI\b|natural[- ]language inference", text, re.I):
            errors.append(f"paper mislabels the lexical support checker as NLI in {path.name}")
        for token in FORBIDDEN_TEXT:
            if token in text:
                errors.append(f"paper source references forbidden prior manuscript in {path.name}: {token}")
        for pattern in ANONYMITY:
            if pattern.search(text):
                errors.append(f"paper anonymity hygiene failed in {path.name}")

    errors.extend(_validate_empirical_inclusion_contract(paper_dir))

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
        if claim_type == "empirical_retrospective" and status == "SUPPORTED":
            expected_source = "evaluation/PHASE2_AGGREGATE_PUBLIC.json"
            if sources != [expected_source]:
                errors.append(f"retrospective claim {claim.get('claim_id')} lacks the audited aggregate")
            else:
                public_path = ROOT / expected_source
                if not public_path.is_file():
                    errors.append("audited public aggregate is missing")
                else:
                    payload = read_json(public_path)
                    if payload.get("status") != "AUDITED_RETROSPECTIVE_MACHINE_RESULTS":
                        errors.append("public aggregate is not an audited retrospective machine result")
                    local_path = ROOT / "evaluation/results/final_machine/phase2_aggregate.json"
                    if local_path.is_file() and sha256_file(public_path) != sha256_file(local_path):
                        errors.append("public aggregate differs from local audited aggregate")
        elif claim_type == "empirical" and status == "SUPPORTED":
            if not sources:
                errors.append(f"empirical claim {claim.get('claim_id')} lacks release evidence")
            for source in sources:
                for source_error in _validate_empirical_source(source):
                    errors.append(f"empirical claim {claim.get('claim_id')}: {source_error}")
        if claim_type not in {"empirical", "empirical_retrospective"} and status == "SUPPORTED":
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

    seal_errors = validate_seal(paper_dir, release_dir)
    errors.extend(seal_errors)
    release_enabled = (paper_dir / "build/release_table_gate.json").is_file() and not seal_errors
    if not release_enabled:
        for stem in CLAIM_FIGURE_STEMS:
            for suffix in (".pdf", ".svg", ".png"):
                for directory in ("figs", "figures"):
                    artifact = paper_dir / directory / f"{stem}{suffix}"
                    if artifact.exists():
                        errors.append(
                            f"unsealed claim-bearing figure must not exist in the review tree: "
                            f"{artifact.relative_to(paper_dir)}"
                        )

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
