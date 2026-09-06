"""Ensure human/ethics scaffolds cannot be mistaken for completed study data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.schemas import StudyTemplateEnvelope

BASE = ROOT / "evaluation/human_eval"
TEMPLATE_DIRS = (
    "claim_annotation_v1",
    "judge_validation_v1",
    "researcher_pilot_v1",
    "ethics_v1",
)
BANNER = "STATUS: TEMPLATE ONLY — NO DATA COLLECTED"
ALLOWED_TEMPLATE_ENTRIES = frozenset({"README.md", "template.json"})


def validate_templates(base: Path = BASE) -> list[str]:
    errors: list[str] = []
    if base.is_symlink() or not base.is_dir():
        return ["human_eval: template root is missing or is a symlink"]
    try:
        unknown_template_dirs = sorted(
            path.name
            for path in base.iterdir()
            if path.name not in TEMPLATE_DIRS and (path / "template.json").exists()
        )
    except OSError as exc:
        return [f"human_eval: cannot inspect template root: {exc}"]
    if unknown_template_dirs:
        errors.append(f"human_eval: unrecognized template directories: {unknown_template_dirs}")
    for name in TEMPLATE_DIRS:
        directory = base / name
        if directory.is_symlink() or not directory.is_dir():
            errors.append(f"{name}: template directory is missing or is a symlink")
            continue

        try:
            entries = sorted(directory.rglob("*"), key=lambda path: path.as_posix())
        except OSError as exc:
            errors.append(f"{name}: cannot inspect template directory: {exc}")
            continue

        unexpected = [
            path.relative_to(directory).as_posix()
            for path in entries
            if path.relative_to(directory).as_posix() not in ALLOWED_TEMPLATE_ENTRIES
            or path.is_symlink()
            or not path.is_file()
        ]
        if unexpected:
            errors.append(f"{name}: unexpected or unsafe entries: {unexpected}")

        readme = directory / "README.md"
        template = directory / "template.json"
        try:
            readme_lines = (
                readme.read_text(encoding="utf-8").splitlines()
                if readme.is_file() and not readme.is_symlink()
                else []
            )
        except OSError as exc:
            errors.append(f"{name}: cannot read README.md: {exc}")
            readme_lines = []
        if not readme_lines or readme_lines[0] != BANNER:
            errors.append(f"{name}: missing exact template-only README banner")

        if template.is_symlink() or not template.is_file():
            errors.append(f"{name}: template.json is missing or is a symlink")
            continue
        try:
            envelope = StudyTemplateEnvelope.model_validate(
                json.loads(template.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            errors.append(f"{name}: invalid safe template envelope: {exc}")
            continue
        if envelope.records or envelope.contains_completed_data or envelope.study_status != "NOT_STARTED":
            errors.append(f"{name}: template claims completed study data")
    return errors


def main() -> int:
    errors = validate_templates()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("human/ethics template safety validation OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
