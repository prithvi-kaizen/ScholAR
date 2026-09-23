#!/usr/bin/env python3
"""Audit official style provenance and the compiled EACL review PDF."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.release.io import sha256_file, write_json  # noqa: E402
from evaluation.release.schemas import ArtifactHash  # noqa: E402


SUBMISSION_BLOCKING_PDF_PHRASES = (
    "evidence gate pending",
    "claim-bearing release does not yet exist",
    "the final version of this section will report",
)


class OfficialStyleManifest(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_type: Literal["official_acl_style_manifest"] = "official_acl_style_manifest"
    status: Literal["PENDING", "INSTALLED"]
    repository: Literal["https://github.com/acl-org/acl-style-files"]
    upstream_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    modified: bool | None = None
    files: list[ArtifactHash]

    model_config = ConfigDict(extra="forbid")


def validate_style_provenance(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    manifest_path = paper_dir / "style/official_style_manifest.json"
    try:
        manifest = OfficialStyleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"official style manifest is invalid: {exc}"]
    if manifest.status != "INSTALLED" or not manifest.upstream_commit or manifest.modified is not False:
        errors.append("official style manifest is not a pinned unmodified installation")
    required = {"style/acl.sty", "style/acl_natbib.bst"}
    records = {item.path: item for item in manifest.files}
    if set(records) != required:
        errors.append("official style manifest must contain exactly acl.sty and acl_natbib.bst")
    for name in sorted(required):
        path = paper_dir / name.removeprefix("style/") if paper_dir.name == "style" else paper_dir / name
        record = records.get(name)
        if record is None:
            continue
        if not path.is_file():
            errors.append(f"official style file is missing: {name}")
        elif path.stat().st_size != record.bytes or sha256_file(path) != record.sha256:
            errors.append(f"official style file differs from pinned upstream bytes: {name}")
    commit_path = paper_dir / "style/UPSTREAM_COMMIT.txt"
    if not commit_path.is_file() or commit_path.read_text(encoding="utf-8").strip() != manifest.upstream_commit:
        errors.append("UPSTREAM_COMMIT.txt differs from the pinned official-style commit")
    return errors


def _run_tool(command: list[str], cwd: Path) -> tuple[str, str | None]:
    if shutil.which(command[0]) is None:
        return "", f"required PDF audit tool is unavailable: {command[0]}"
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return completed.stdout, f"{' '.join(command)} failed: {detail}"
    return completed.stdout, None


def latex_reference_errors(log_text: str) -> list[str]:
    patterns = (
        r"LaTeX Warning: (?:Reference|Citation).+undefined",
        r"There were undefined (?:references|citations)",
        r"Please \(re\)run BibTeX",
    )
    return [
        "compiled LaTeX log contains unresolved references or citations"
        for pattern in patterns
        if re.search(pattern, log_text, re.I)
    ][:1]


def submission_draft_errors(pdf_text: str) -> list[str]:
    lowered_text = " ".join(pdf_text.casefold().split())
    return [
        f"compiled review PDF contains submission-blocking draft text: {phrase}"
        for phrase in SUBMISSION_BLOCKING_PDF_PHRASES
        if phrase in lowered_text
    ]


def validate_submission_pdf(paper_dir: Path, output: Path | None = None) -> list[str]:
    paper_dir = paper_dir.resolve()
    errors = validate_style_provenance(paper_dir)
    checks_performed = ["official_style_provenance"]
    pdf = paper_dir / "main.pdf"
    aux = paper_dir / "main.aux"
    log = paper_dir / "main.log"
    if not pdf.is_file():
        return errors + ["compiled submission PDF is missing: main.pdf"]

    if shutil.which("qpdf"):
        _stdout, error = _run_tool(["qpdf", "--check", "main.pdf"], paper_dir)
        if error:
            errors.append(error)
        else:
            checks_performed.append("qpdf")

    fonts_checked_with_pdffonts = False
    if shutil.which("pdffonts"):
        fonts, error = _run_tool(["pdffonts", "main.pdf"], paper_dir)
        if error:
            errors.append(error)
        else:
            fonts_checked_with_pdffonts = True
            checks_performed.append("embedded_fonts_pdffonts")
            for line in fonts.splitlines()[2:]:
                if not line.strip():
                    continue
                if "Type 3" in line:
                    errors.append("compiled PDF contains a forbidden Type 3 font")
                flags = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", line)
                if flags and flags.group(1) != "yes":
                    errors.append(f"compiled PDF contains an unembedded font: {line.strip()}")

    try:
        import fitz

        document = fitz.open(pdf)
        checks_performed.extend(["pdf_parse", "A4", "anonymous_review", "submission_placeholders"])
        if not document.page_count:
            errors.append("compiled PDF has no pages")
        for index, page in enumerate(document, start=1):
            width, height = page.rect.width, page.rect.height
            if abs(width - 595.276) > 2.0 or abs(height - 841.89) > 2.0:
                errors.append(f"PDF page {index} is not A4: {width:.2f}x{height:.2f} pt")
        text = "\n".join(page.get_text() for page in document)
        metadata = document.metadata or {}
        if not fonts_checked_with_pdffonts:
            checks_performed.append("embedded_fonts_pymupdf")
            font_records = {
                font
                for page in document
                for font in page.get_fonts(full=True)
            }
            for font in sorted(font_records):
                xref, _extension, font_type, base_name = font[:4]
                if "type3" in str(font_type).replace(" ", "").casefold():
                    errors.append(f"compiled PDF contains a forbidden Type 3 font: {base_name}")
                try:
                    embedded = document.extract_font(xref)[-1]
                except Exception as exc:
                    errors.append(f"compiled PDF font extraction failed for {base_name}: {exc}")
                    continue
                if not embedded:
                    errors.append(f"compiled PDF contains an unembedded font: {base_name}")
        document.close()
        if "Anonymous ACL submission" not in text:
            errors.append("compiled review PDF lacks the anonymous author marker")
        if "??" in text:
            errors.append("compiled review PDF contains an unresolved reference marker")
        errors.extend(submission_draft_errors(text))
        for value in metadata.values():
            if value and re.search(
                r"/(?:Users|home)/[A-Za-z0-9._-]+/|[A-Za-z]:\\Users\\[^\\\s]+\\",
                str(value),
                re.I,
            ):
                errors.append("compiled PDF metadata contains a deanonymizing token")
    except Exception as exc:
        errors.append(f"compiled PDF inspection failed: {exc}")

    if not aux.is_file():
        errors.append("LaTeX auxiliary file is missing; content-page limit cannot be checked")
    else:
        match = re.search(r"\\newlabel\{content:end\}\{\{.*?\}\{(\d+)\}", aux.read_text(encoding="utf-8", errors="replace"))
        if not match:
            errors.append("content:end label is missing from main.aux")
        elif int(match.group(1)) > 6:
            errors.append(f"review content exceeds six pages: {match.group(1)}")
        else:
            checks_performed.append("six_content_pages")
    if not log.is_file():
        errors.append("LaTeX log is missing; unresolved references cannot be audited")
    else:
        errors.extend(latex_reference_errors(log.read_text(encoding="utf-8", errors="replace")))

    if output is not None:
        write_json(output, {
            "schema_version": "1.0",
            "artifact_type": "compiled_pdf_compliance_report",
            "status": "PASS" if not errors else "FAIL",
            "pdf_path": "paper/eacl_industry/main.pdf",
            "pdf_sha256": sha256_file(pdf),
            "style_manifest_sha256": sha256_file(paper_dir / "style/official_style_manifest.json"),
            "checks": sorted(set(checks_performed)),
            "errors": sorted(set(errors)),
        })
    return sorted(set(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=ROOT / "paper/eacl_industry")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors = validate_submission_pdf(args.paper_dir, args.output)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("compiled submission PDF compliance OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
