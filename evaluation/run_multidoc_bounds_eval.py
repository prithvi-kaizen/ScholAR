"""
run_multidoc_bounds_eval.py
----------------------------
Computes two reference bounds for the multi-document locality benchmark
(see run_multidoc_eval.py), to give the R@1=0.10 headline number honest
context instead of leaving it unexplained:

- ORACLE: retrieval restricted to only the anchor paper + the single correct
  secondary paper's chunks (no other secondary papers in the pool). This
  isolates "can BM25 find the right chunk within the correct paper" from
  "can BM25 identify which paper, among many, is correct."
- RANDOM FLOOR: the expected recall/MRR of picking a uniformly random
  secondary paper (no retrieval signal at all) from however many secondary
  papers are actually loaded for that anchor.

Usage:
    python3 evaluation/run_multidoc_bounds_eval.py --no-ingest
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from backend.services.retrieval_service import retrieve_chunks  # noqa: E402
from run_multidoc_eval import (  # noqa: E402
    BENCHMARK_PATH,
    _build_secondary_pool,
    _load_or_resolve_refs,
    _load_paper_chunks,
    _precision_at_k,
    _reciprocal_rank,
    _src_matches,
)

RESULTS_DIR       = ROOT / "evaluation" / "results"
RESULTS_JSON      = RESULTS_DIR / "multidoc_bounds_results.json"
RESULTS_MD        = RESULTS_DIR / "multidoc_bounds_report.md"
MULTIDOC_EVAL_JSON = RESULTS_DIR / "multidoc_eval_results.json"


def _harmonic(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1))


async def evaluate(do_ingest: bool = False) -> None:
    cases: list[dict[str, Any]] = json.loads(BENCHMARK_PATH.read_text())
    locality_cases = [
        c for c in cases
        if c["question_type"] == "locality" and c.get("expected_secondary_arxiv_id")
    ]

    anchor_ids = list({c["anchor_paper_id"] for c in locality_cases})
    needed: dict[str, set[str]] = {}
    for c in locality_cases:
        needed.setdefault(c["anchor_paper_id"], set()).add(c["expected_secondary_arxiv_id"])

    anchor_chunks: dict[str, list[dict[str, Any]]] = {}
    secondary_pool: dict[str, list[dict[str, Any]]] = {}
    for aid in anchor_ids:
        anchor_chunks[aid] = _load_paper_chunks(aid)
        refs = await _load_or_resolve_refs(aid)
        secondary_pool[aid] = await _build_secondary_pool(aid, refs, needed.get(aid, set()), do_ingest)

    oracle_results: list[dict[str, Any]] = []
    floor_results: list[dict[str, Any]] = []

    for case in locality_cases:
        aid = case["anchor_paper_id"]
        exp_id = case["expected_secondary_arxiv_id"]
        exp_kw = case.get("expected_secondary_title_keywords", [])
        question = case["question"]

        full_secondary = secondary_pool.get(aid, [])
        distinct_papers = {c.get("source_paper_id") for c in full_secondary if c.get("source_paper_id")}
        n_papers = len(distinct_papers)

        # ORACLE: restrict the pool to anchor chunks + only the correct paper's chunks.
        oracle_secondary = [
            c for c in full_secondary if _src_matches(c.get("source_paper_id", ""), exp_id)
        ]
        if oracle_secondary:
            oracle_pool = anchor_chunks.get(aid, []) + oracle_secondary
            ranked = retrieve_chunks(question, oracle_pool, limit=10)
            r1 = int(_precision_at_k(ranked, exp_id, exp_kw, 1))
            r5 = int(_precision_at_k(ranked, exp_id, exp_kw, 5))
            mrr = _reciprocal_rank(ranked, exp_id, exp_kw)
        else:
            r1 = r5 = 0
            mrr = 0.0
        oracle_results.append({
            "case_id": case["case_id"], "recall@1": r1, "recall@5": r5,
            "mrr": round(mrr, 3), "n_papers": n_papers,
        })

        # RANDOM FLOOR: expected recall of guessing uniformly among n_papers, no signal.
        if n_papers > 0:
            floor_r1 = 1.0 / n_papers
            floor_r5 = min(5, n_papers) / n_papers
            floor_mrr = _harmonic(n_papers) / n_papers
        else:
            floor_r1 = floor_r5 = floor_mrr = 0.0
        floor_results.append({
            "case_id": case["case_id"], "recall@1": round(floor_r1, 3),
            "recall@5": round(floor_r5, 3), "mrr": round(floor_mrr, 3), "n_papers": n_papers,
        })

    def agg(rows: list[dict[str, Any]], key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 3) if rows else 0.0

    summary = {
        "n": len(locality_cases),
        "oracle": {
            "recall@1": agg(oracle_results, "recall@1"),
            "recall@5": agg(oracle_results, "recall@5"),
            "mrr": agg(oracle_results, "mrr"),
        },
        "random_floor": {
            "recall@1": agg(floor_results, "recall@1"),
            "recall@5": agg(floor_results, "recall@5"),
            "mrr": agg(floor_results, "mrr"),
        },
        "avg_candidate_papers": round(
            sum(r["n_papers"] for r in oracle_results) / len(oracle_results), 1
        ) if oracle_results else 0,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(
        {"summary": summary, "oracle_cases": oracle_results, "floor_cases": floor_results},
        indent=2,
    ))

    # Read ScholAR's actual flat-retrieval numbers from the current
    # multidoc_eval_results.json rather than hardcoding them, so this report
    # can't silently drift out of sync if that eval is re-run with different results.
    scholar_row = "| ScholAR (flat retrieval across all loaded papers) | n/a | n/a | n/a |"
    if MULTIDOC_EVAL_JSON.exists():
        locality_arxiv = json.loads(MULTIDOC_EVAL_JSON.read_text()).get("locality_arxiv")
        if locality_arxiv:
            scholar_row = (
                "| ScholAR (flat retrieval across all loaded papers) | "
                f"{locality_arxiv['recall@1']} | {locality_arxiv['recall@5']} | {locality_arxiv['mrr']} |"
            )

    md_lines = [
        "# Multi-Document Locality: Oracle and Random-Floor Bounds",
        "",
        f"Computed over the {len(locality_cases)} locality cases with a resolvable "
        "arXiv secondary paper (same subset as `locality_arxiv` in "
        "`multidoc_eval_report.md`).",
        "",
        "| Bound | R@1 | R@5 | MRR |",
        "|---|---:|---:|---:|",
        f"| Random floor (uniform guess, no retrieval signal) | {summary['random_floor']['recall@1']} | {summary['random_floor']['recall@5']} | {summary['random_floor']['mrr']} |",
        scholar_row,
        f"| Oracle (retrieval restricted to only the correct paper) | {summary['oracle']['recall@1']} | {summary['oracle']['recall@5']} | {summary['oracle']['mrr']} |",
        "",
        f"Average candidate secondary papers per case: {summary['avg_candidate_papers']}.",
    ]
    RESULTS_MD.write_text("\n".join(md_lines) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_JSON}\nWrote {RESULTS_MD}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-doc oracle / random-floor bounds")
    parser.add_argument("--no-ingest", action="store_true", help="Offline only, skip downloads")
    args = parser.parse_args()
    asyncio.run(evaluate(do_ingest=not args.no_ingest))
