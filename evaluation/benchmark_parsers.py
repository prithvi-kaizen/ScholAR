"""Parser Ablation Benchmark Script (Phase 1).

Compares:
- P0: PyMuPDF + fixed 512 tokens
- P1: PyMuPDF + heuristic font AST
- P2: Docling + fixed 512 tokens
- P3: Docling + semantic chunks
- P4: ScholAR (Docling + provenance AST + structure-aware chunking)

Measures:
- Ingestion latency (ms / page)
- Section heading recall & hierarchy depth
- Structured table recovery count
- Visual figure linking rate
- Output EvidenceAST block distribution
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.schemas.evidence import EvidenceModality
from backend.services.ingestion_service import (
    DualEngineIngestionService,
    PARSER_ABLATIONS,
)
from backend.services.pdf_service import paper_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("parser_benchmark")

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_parser_on_paper(
    paper_id: str,
    config_id: str,
) -> dict[str, Any]:
    """Run a specific parser ablation configuration on a paper and extract metrics."""
    config = PARSER_ABLATIONS[config_id]
    p_dir = paper_dir(paper_id)
    pdf_path = p_dir / "paper.pdf"

    if not pdf_path.exists():
        logger.warning("PDF not found for paper [%s]", paper_id)
        return {"error": f"PDF not found at {pdf_path}"}

    start_time = time.perf_counter()
    ast = DualEngineIngestionService.ingest_paper(
        pdf_path=pdf_path,
        document_id=paper_id,
        config=config,
    )
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    num_pages = max(ast.page_count, 1)
    text_blocks = ast.get_text_blocks()
    table_blocks = ast.get_table_blocks()
    visual_blocks = ast.get_visual_blocks()

    ms_per_page = round(elapsed_ms / num_pages, 2)

    return {
        "config_id": config_id,
        "parser_name": config.parser_name,
        "chunking_strategy": config.chunking_strategy,
        "paper_id": paper_id,
        "page_count": num_pages,
        "total_latency_ms": round(elapsed_ms, 2),
        "ms_per_page": ms_per_page,
        "total_blocks": len(ast.blocks),
        "text_block_count": len(text_blocks),
        "table_block_count": len(table_blocks),
        "visual_block_count": len(visual_blocks),
        "section_node_count": len(ast.sections),
        "parser_engine": ast.parser_engine,
        "degraded_mode": ast.degraded_mode,
    }


def run_ablation_matrix(paper_ids: list[str]) -> dict[str, Any]:
    """Run all 5 parser configurations across a list of test papers."""
    results: dict[str, list[dict[str, Any]]] = {}

    for config_id in ["P0", "P1", "P2", "P3", "P4"]:
        logger.info(f"Evaluating Parser Configuration [{config_id}]: {PARSER_ABLATIONS[config_id].parser_name}...")
        results[config_id] = []
        for pid in paper_ids:
            res = evaluate_parser_on_paper(pid, config_id)
            results[config_id].append(res)

    # Compute aggregate statistics per configuration
    summary: dict[str, Any] = {}
    for config_id, run_list in results.items():
        valid_runs = [r for r in run_list if "error" not in r]
        if not valid_runs:
            continue
        avg_ms_page = sum(r["ms_per_page"] for r in valid_runs) / len(valid_runs)
        avg_tables = sum(r["table_block_count"] for r in valid_runs) / len(valid_runs)
        avg_sections = sum(r["section_node_count"] for r in valid_runs) / len(valid_runs)
        avg_blocks = sum(r["total_blocks"] for r in valid_runs) / len(valid_runs)

        summary[config_id] = {
            "config_id": config_id,
            "name": PARSER_ABLATIONS[config_id].parser_name,
            "mean_ms_per_page": round(avg_ms_page, 2),
            "mean_tables_extracted": round(avg_tables, 2),
            "mean_sections_extracted": round(avg_sections, 2),
            "mean_evidence_blocks": round(avg_blocks, 2),
        }

    output_payload = {
        "summary": summary,
        "detailed_runs": results,
        "evaluated_at": time.time(),
    }

    out_file = RESULTS_DIR / "parser_ablation_results.json"
    out_file.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    logger.info("Saved parser ablation results to %s", out_file)
    return output_payload


def main():
    parser = argparse.ArgumentParser(description="ScholAR Parser Ablation Benchmark")
    parser.add_argument(
        "--papers",
        nargs="+",
        default=["1406.2661", "1412.6980", "1706.03762"],
        help="List of paper IDs to benchmark",
    )
    args = parser.parse_args()

    print("\n" + "=" * 65)
    print("        ScholAR Parser Ablation Benchmark (P0 - P4)")
    print("=" * 65 + "\n")

    res = run_ablation_matrix(args.papers)

    print("\nSummary Results:")
    print("-" * 65)
    print(f"{'Config':<8} | {'Parser Name':<28} | {'ms/page':<8} | {'Tables':<6} | {'Sections':<8}")
    print("-" * 65)
    for cid, s in res["summary"].items():
        print(f"{cid:<8} | {s['name']:<28} | {s['mean_ms_per_page']:<8} | {s['mean_tables_extracted']:<6} | {s['mean_sections_extracted']:<8}")
    print("-" * 65 + "\n")


if __name__ == "__main__":
    main()
