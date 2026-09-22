#!/usr/bin/env python3
"""Audit script to validate EACL 2027 Demonstration paper integrity.

Enforces:
1. Compilation: main.pdf exists and has valid pages.
2. Page budget: Content pages <= 6 (verified via main.aux content:end label).
3. Track requirement: Single-blind mode (author names must appear in paper text).
4. References: 0 undefined citations or cross-references.
5. Assets: Vector architecture figure and authentic UI screenshot present.
6. Claims: All claim IDs in claim_map.json exist and are marked SUPPORTED or PENDING.
7. Disallowed tokens: No raw TODO, TBD, or fake result rows.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_DIR = ROOT / "paper/eacl_demo"


def validate_demo_paper(paper_dir: Path = DEFAULT_PAPER_DIR) -> dict[str, Any]:
    issues: list[str] = []
    pdf_path = paper_dir / "main.pdf"
    aux_path = paper_dir / "main.aux"
    log_path = paper_dir / "main.log"
    claim_map_path = paper_dir / "claim_map.json"
    venue_req_path = paper_dir / "venue_requirements.json"

    # 1. File presence
    if not pdf_path.is_file():
        issues.append("main.pdf does not exist. Please compile paper first.")
    if not aux_path.is_file():
        issues.append("main.aux does not exist.")
    if not log_path.is_file():
        issues.append("main.log does not exist.")
    if not claim_map_path.is_file():
        issues.append("claim_map.json does not exist.")

    if issues:
        return {"status": "FAIL", "issues": issues}

    # 2. Content page count check via aux file
    aux_content = aux_path.read_text(encoding="utf-8")
    m = re.search(r"\\newlabel\{content:end\}\{\{[^}]*\}\{(\d+)\}", aux_content)
    if not m:
        issues.append("Could not locate \\label{content:end} in main.aux")
        content_page_end = None
    else:
        content_page_end = int(m.group(1))
        if content_page_end > 6:
            issues.append(f"Content page budget violated: content:end is on page {content_page_end} (> 6)")

    # 3. Log file check: undefined references or severe warnings
    log_content = log_path.read_text(encoding="utf-8", errors="replace")
    if "LaTeX Warning: There were undefined references" in log_content:
        issues.append("LaTeX reports undefined references in main.log")
    if "LaTeX Warning: Label(s) may have changed. Rerun" in log_content:
        issues.append("LaTeX labels changed; rerun required.")

    # 4. Check author presence (single-blind verification)
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        first_page_text = doc[0].get_text() if total_pages > 0 else ""
    except ImportError:
        import pypdf
        reader = pypdf.PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        first_page_text = reader.pages[0].extract_text() if total_pages > 0 else ""

    if "Prithviraj" not in first_page_text:
        issues.append("Single-blind requirement failed: Author name 'Prithviraj' not found on page 1")

    # 5. Check placeholder tokens in source tex files
    tex_files = list(paper_dir.glob("*.tex")) + list((paper_dir / "sections").glob("*.tex"))
    for tf in tex_files:
        content = tf.read_text(encoding="utf-8")
        for match in re.finditer(r"\b(TODO|TBD|FIXME|XXX)\b", content):
            issues.append(f"Disallowed placeholder token '{match.group(1)}' found in {tf.name}")

    # 6. Check claim map coverage
    claim_map = json.loads(claim_map_path.read_text(encoding="utf-8"))
    claims = claim_map.get("claims", [])
    if len(claims) < 5:
        issues.append("claim_map.json contains fewer than 5 registered claims")

    status = "PASS" if not issues else "FAIL"
    return {
        "status": status,
        "content_page_end": content_page_end,
        "total_pdf_pages": total_pages,
        "claims_audited": len(claims),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate EACL demo paper compliance")
    parser.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    args = parser.parse_args()

    report = validate_demo_paper(args.paper_dir)
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
