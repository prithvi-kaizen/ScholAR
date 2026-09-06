"""Typed, hash-bound release gate evidence validation."""

from __future__ import annotations

from pathlib import Path

from evaluation.release.io import ROOT, read_json, sha256_file
from evaluation.release.schemas import GateRegistry


def load_gate_registry(path: Path) -> GateRegistry:
    return GateRegistry.model_validate(read_json(path))


def validate_gate_registry(
    path: Path,
    *,
    required_phase: str | None = None,
    require_all_cleared: bool = False,
) -> tuple[GateRegistry | None, list[str]]:
    errors: list[str] = []
    try:
        registry = load_gate_registry(path)
    except Exception as exc:
        return None, [f"invalid gate registry: {exc}"]
    for gate in registry.gates:
        if required_phase is not None and gate.phase != required_phase:
            continue
        if gate.status != "CLEARED":
            if required_phase is not None or require_all_cleared:
                errors.append(f"gate {gate.id} is {gate.status}")
            continue
        for evidence in gate.evidence:
            candidate = (ROOT / evidence.path).resolve()
            try:
                candidate.relative_to(ROOT.resolve())
            except ValueError:
                errors.append(f"gate {gate.id} evidence escapes repository: {evidence.path}")
                continue
            if not candidate.is_file():
                errors.append(f"gate {gate.id} evidence is missing: {evidence.path}")
                continue
            if candidate.stat().st_size != evidence.bytes:
                errors.append(f"gate {gate.id} evidence byte count differs: {evidence.path}")
            if sha256_file(candidate) != evidence.sha256:
                errors.append(f"gate {gate.id} evidence hash differs: {evidence.path}")
            if evidence.claim_status != "current" or evidence.evidence_class == "toy":
                errors.append(f"gate {gate.id} uses non-current or toy evidence: {evidence.path}")
    return registry, sorted(set(errors))
