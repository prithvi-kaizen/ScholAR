"""build_spiqa.py: Assemble or validate SPIQA multimodal scientific QA benchmark cases.

SPIQA (Pramanick et al., NeurIPS 2024) evaluates multimodal QA over scientific figures,
tables, plots, and schematics. This builder converts raw SPIQA downloads or validates
curated offline sample cases for strict-local evaluation.

Usage:
    python3 evaluation/spiqa/build_spiqa.py --selfcheck
    python3 evaluation/spiqa/build_spiqa.py --src /path/to/spiqa_raw --output evaluation/spiqa/spiqa_cases.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.spiqa import SPIQAAdapter, SPIQAVisualType

SAMPLE_CASES = HERE / "spiqa_cases_sample.json"


def validate_cases(cases_path: Path) -> bool:
    """Validate that cases file exists and satisfies required schema."""
    if not cases_path.exists():
        print(f"[ERROR] Cases file not found: {cases_path}", file=sys.stderr)
        return False

    try:
        data = json.loads(cases_path.read_text(encoding="utf-8"))
        cases = data.get("cases", []) if isinstance(data, dict) else data
        if not cases:
            print("[ERROR] No cases found in file", file=sys.stderr)
            return False

        valid_types = {t.value for t in SPIQAVisualType}
        for idx, c in enumerate(cases):
            assert "case_id" in c, f"Case {idx} missing 'case_id'"
            assert "paper_id" in c, f"Case {idx} missing 'paper_id'"
            assert "question" in c, f"Case {idx} missing 'question'"
            assert "answers" in c, f"Case {idx} missing 'answers'"
            assert "gold_pages" in c, f"Case {idx} missing 'gold_pages'"
            vtype = c.get("visual_type", "mixed")
            assert vtype in valid_types, f"Case {idx} invalid visual_type '{vtype}'"

        # Check adapter loading
        adapter = SPIQAAdapter(data_path=cases_path)
        examples = adapter.load_examples(split="all")
        assert len(examples) == len(cases), f"Adapter loaded {len(examples)} != {len(cases)}"
        print(f"[OK] Successfully validated {len(examples)} SPIQA cases from {cases_path.name}")
        return True
    except Exception as exc:
        print(f"[ERROR] Validation failed: {exc}", file=sys.stderr)
        return False


def convert_raw_spiqa(src_dir: Path, output_file: Path) -> None:
    """Convert raw SPIQA JSON splits into ScholAR's benchmark schema."""
    if not src_dir.exists():
        raise FileNotFoundError(f"Raw SPIQA directory not found: {src_dir}")

    cases: list[dict] = []
    for split in ("test", "val", "train"):
        split_file = src_dir / f"{split}.json"
        if not split_file.exists():
            continue

        raw = json.loads(split_file.read_text(encoding="utf-8"))
        items = raw if isinstance(raw, list) else raw.get("data", [])
        for item in items:
            case_id = item.get("id") or item.get("question_id") or f"spiqa_{len(cases)+1}"
            paper_id = item.get("paper_id") or item.get("doc_id", "")
            q = item.get("question", "")
            answers = item.get("answers") or ([item["answer"]] if "answer" in item else [])
            vtype = item.get("image_type") or item.get("figure_type", "mixed")
            if vtype not in {t.value for t in SPIQAVisualType}:
                vtype = SPIQAVisualType.MIXED.value

            cases.append({
                "case_id": case_id,
                "paper_id": paper_id,
                "paper_title": item.get("paper_title", ""),
                "question": q,
                "visual_type": vtype,
                "figure_label": item.get("figure_label", ""),
                "gold_pages": item.get("pages") or [1],
                "caption": item.get("caption", ""),
                "evidence": item.get("evidence", []),
                "answers": answers,
                "split": split,
                "answerable": True,
            })

    output_payload = {
        "benchmark": "SPIQA",
        "version": "1.0",
        "cases": cases,
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble or validate SPIQA cases.")
    parser.add_argument("--selfcheck", action="store_true", help="Validate default sample cases")
    parser.add_argument("--src", type=Path, help="Path to raw SPIQA directory")
    parser.add_argument("--output", type=Path, default=HERE / "spiqa_cases.json", help="Output JSON path")
    args = parser.parse_args()

    if args.selfcheck:
        success = validate_cases(SAMPLE_CASES)
        sys.exit(0 if success else 1)

    if args.src:
        convert_raw_spiqa(args.src, args.output)
        validate_cases(args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
