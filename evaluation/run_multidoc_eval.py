"""
run_multidoc_eval.py
--------------------
Evaluates ScholAR's multi-document citation-chasing mode.

Metric: given an anchor paper and a question, does retrieving across
the anchor paper + its resolved secondary references surface a chunk
from the *correct* secondary paper in the top-k results?

Reports Recall@1/3/5 and MRR — same format as retrieval_eval_report.md
so numbers are directly comparable.

Usage:
    python3 evaluation/run_multidoc_eval.py            # ingest needed papers (default)
    python3 evaluation/run_multidoc_eval.py --no-ingest  # offline only, skip downloads

Prerequisites:
    - The anchor papers must already be prepared locally:
        POST /api/papers/prepare  (or use the frontend)
    - Network access to Semantic Scholar API (keyless) and arXiv PDF mirror
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

import httpx

# Make sure the project root is on sys.path when run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.chunking_service import chunk_pages
from backend.services.pdf_service import download_pdf, extract_pages, paper_dir, read_json, safe_paper_id, write_json
from backend.services.reference_service import load_references, resolve_references
from backend.services.retrieval_service import retrieve_chunks

BENCHMARK_PATH = ROOT / "evaluation" / "multidoc_benchmark.json"
RESULTS_DIR    = ROOT / "evaluation" / "results"
RESULTS_JSON   = RESULTS_DIR / "multidoc_eval_results.json"
RESULTS_MD     = RESULTS_DIR / "multidoc_eval_report.md"

ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
ARXIV_ABS_BASE = "https://arxiv.org/abs/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_paper_chunks(local_id: str) -> list[dict[str, Any]]:
    """Load chunks for a paper and ensure source_paper_id is set."""
    path = paper_dir(local_id) / "chunks.json"
    if not path.exists():
        return []
    chunks = read_json(path)
    return [{**c, "source_paper_id": local_id} if "source_paper_id" not in c else c
            for c in chunks]


def _normalise_id(arxiv_id: str) -> str:
    """Return the canonical local_id for an arXiv ID (dots preserved)."""
    return safe_paper_id(arxiv_id)


def _src_matches(src_local_id: str, expected_arxiv_id: str | None) -> bool:
    if not expected_arxiv_id:
        return False
    norm_expected = _normalise_id(expected_arxiv_id)
    return src_local_id == norm_expected or expected_arxiv_id in src_local_id


def _title_matches(local_id: str, keywords: list[str]) -> bool:
    if not keywords:
        return False
    meta_path = paper_dir(local_id) / "metadata.json"
    if not meta_path.exists():
        return False
    title = (read_json(meta_path).get("title") or "").lower()
    return all(kw.lower() in title for kw in keywords)


def _precision_at_k(
    ranked: list[dict[str, Any]],
    expected_arxiv_id: str | None,
    expected_keywords: list[str],
    k: int,
) -> bool:
    for chunk in ranked[:k]:
        src = chunk.get("source_paper_id", "")
        if _src_matches(src, expected_arxiv_id):
            return True
        if _title_matches(src, expected_keywords):
            return True
    return False


def _reciprocal_rank(
    ranked: list[dict[str, Any]],
    expected_arxiv_id: str | None,
    expected_keywords: list[str],
) -> float:
    for rank, chunk in enumerate(ranked, start=1):
        src = chunk.get("source_paper_id", "")
        if _src_matches(src, expected_arxiv_id):
            return 1.0 / rank
        if _title_matches(src, expected_keywords):
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Secondary paper ingestion
# ---------------------------------------------------------------------------

async def _get_best_pdf_url(arxiv_id: str, ref: dict[str, Any]) -> str:
    """
    Try to find the best open-access PDF URL for a paper.
    Priority:
      1. openAccessPdf.url from Semantic Scholar (already in ref if resolved via S2)
      2. Semantic Scholar API lookup for the paper
      3. arXiv PDF URL (may 403 under heavy rate limiting)
    """
    # Check if ref already has an open-access URL from S2 resolution
    oa_url = ref.get("openAccessPdf_url") or ""
    if oa_url and oa_url.startswith("http"):
        return oa_url

    # Try to fetch from Semantic Scholar
    s2_id = ref.get("s2_paper_id")
    if s2_id or arxiv_id:
        lookup_id = s2_id or f"arXiv:{arxiv_id}"
        s2_url = f"https://api.semanticscholar.org/graph/v1/paper/{lookup_id}?fields=openAccessPdf"
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                r = await client.get(s2_url)
                if r.status_code == 200:
                    data = r.json()
                    pdf_info = data.get("openAccessPdf") or {}
                    oa = pdf_info.get("url", "")
                    if oa and oa.startswith("http"):
                        return oa
        except Exception:  # noqa: BLE001
            pass

    return f"https://arxiv.org/pdf/{arxiv_id}"


async def _ingest_one(arxiv_id: str, ref: dict[str, Any]) -> list[dict[str, Any]]:
    """Download + chunk one secondary paper. Returns chunks or [] on failure."""
    local_id = _normalise_id(arxiv_id)
    existing = _load_paper_chunks(local_id)
    if existing:
        return existing  # already on disk

    sec_dir   = paper_dir(local_id)
    sec_pdf   = sec_dir / "paper.pdf"
    chunks_path = sec_dir / "chunks.json"
    pages_path  = sec_dir / "pages.json"
    meta_path   = sec_dir / "metadata.json"
    sec_dir.mkdir(parents=True, exist_ok=True)

    # Try to find a non-arXiv open-access URL first
    pdf_url = await _get_best_pdf_url(arxiv_id, ref)

    try:
        if not sec_pdf.exists():
            print(f"      Downloading {arxiv_id} from {pdf_url[:60]}…")
            await download_pdf(pdf_url, sec_pdf)

        pages = extract_pages(sec_pdf)
        chunks = chunk_pages(pages, source_paper_id=local_id)

        write_json(pages_path, pages)
        write_json(chunks_path, chunks)
        if not meta_path.exists():
            write_json(meta_path, {
                "id":        arxiv_id,
                "local_id":  local_id,
                "title":     ref.get("title", ""),
                "authors":   ref.get("authors", []),
                "year":      ref.get("year", ""),
                "summary":   ref.get("abstract", ""),
                "pdf_url":   pdf_url,
                "abs_url":   ref.get("abs_url") or f"{ARXIV_ABS_BASE}{arxiv_id}",
                "source":    "reference",
            })
        print(f"      OK: {len(chunks)} chunks from {arxiv_id}")
        return chunks
    except Exception as exc:  # noqa: BLE001
        print(f"      Could not ingest {arxiv_id}: {exc}")
        # Clean up partial download to avoid stale state
        if sec_pdf.exists() and sec_pdf.stat().st_size < 1024:
            sec_pdf.unlink(missing_ok=True)
        return []


async def _build_secondary_pool(
    anchor_id: str,
    refs: list[dict[str, Any]],
    needed_arxiv_ids: set[str],
    do_ingest: bool,
) -> list[dict[str, Any]]:
    """
    Build the secondary-paper chunk pool for one anchor.

    If do_ingest=True, download papers whose arXiv IDs appear in
    needed_arxiv_ids but are not yet on disk. All other resolved
    references with arXiv IDs are loaded if they happen to be local.
    """
    pool: list[dict[str, Any]] = []
    ref_by_arxiv: dict[str, dict[str, Any]] = {
        r["arxiv_id"]: r for r in refs if r.get("arxiv_id")
    }

    # Always try to load every resolved reference that's already local
    for arxiv_id, ref in ref_by_arxiv.items():
        local_id = _normalise_id(arxiv_id)
        existing = _load_paper_chunks(local_id)
        if existing:
            pool.extend(existing)
            continue
        # If needed by a benchmark case and ingest is allowed, download it
        if do_ingest and arxiv_id in needed_arxiv_ids:
            chunks = await _ingest_one(arxiv_id, ref)
            pool.extend(chunks)

    return pool


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

async def _load_or_resolve_refs(anchor_id: str) -> list[dict[str, Any]]:
    refs = load_references(anchor_id)
    if refs:
        return refs
    dir_ = paper_dir(anchor_id)
    meta_path  = dir_ / "metadata.json"
    pages_path = dir_ / "pages.json"
    if not meta_path.exists():
        print(f"  [SKIP] Anchor paper '{anchor_id}' not prepared locally.")
        return []
    meta  = read_json(meta_path)
    pages = read_json(pages_path) if pages_path.exists() else []
    print(f"  Resolving references for {anchor_id} via Semantic Scholar…")
    refs = await resolve_references(anchor_id, meta, pages)
    print(f"  Resolved {len(refs)} references.")
    return refs


async def evaluate(do_ingest: bool = True) -> None:
    cases: list[dict[str, Any]] = json.loads(BENCHMARK_PATH.read_text())

    # Collect the arXiv IDs actually needed per anchor
    needed: dict[str, set[str]] = {}
    for case in cases:
        aid = case["anchor_paper_id"]
        exp_id = case.get("expected_secondary_arxiv_id")
        needed.setdefault(aid, set())
        if exp_id:
            needed[aid].add(exp_id)

    anchor_ids = list({c["anchor_paper_id"] for c in cases})

    # Load anchor chunks
    anchor_chunks: dict[str, list[dict[str, Any]]] = {}
    for aid in anchor_ids:
        chunks = _load_paper_chunks(aid)
        if not chunks:
            print(f"WARNING: No chunks for anchor '{aid}'. Prepare it first with the backend.")
        anchor_chunks[aid] = chunks

    # Resolve + build secondary pool per anchor
    anchor_refs: dict[str, list[dict[str, Any]]] = {}
    secondary_pool: dict[str, list[dict[str, Any]]] = {}
    for aid in anchor_ids:
        refs = await _load_or_resolve_refs(aid)
        anchor_refs[aid] = refs
        print(f"  Building secondary pool for {aid} (ingest={'yes' if do_ingest else 'no'})…")
        pool = await _build_secondary_pool(aid, refs, needed.get(aid, set()), do_ingest)
        secondary_pool[aid] = pool
        print(f"    {len(pool)} secondary chunks loaded.")

    # Show which needed papers are covered
    for aid, needed_ids in needed.items():
        ref_by_arxiv = {r["arxiv_id"]: r for r in anchor_refs.get(aid, []) if r.get("arxiv_id")}
        for exp_id in sorted(needed_ids):
            local_id = _normalise_id(exp_id)
            chunks = _load_paper_chunks(local_id)
            status = f"{len(chunks)} chunks" if chunks else "NOT FOUND"
            resolved_title = (ref_by_arxiv.get(exp_id) or {}).get("title", "???")[:55]
            print(f"    [{status:>10}] {exp_id:15s} | {resolved_title}")

    # Run cases
    results: list[dict[str, Any]] = []

    def run_case(case: dict[str, Any]) -> dict[str, Any]:
        aid      = case["anchor_paper_id"]
        question = case["question"]
        all_chunks = anchor_chunks.get(aid, []) + secondary_pool.get(aid, [])
        if not all_chunks:
            return {**case, "recall@1": 0, "recall@3": 0, "recall@5": 0, "mrr": 0.0, "error": "no_chunks"}

        ranked = retrieve_chunks(question, all_chunks, limit=10)

        if case["question_type"] == "detail":
            # Detail: anchor-paper retrieval sanity check — passes if any chunk returned
            return {
                **case,
                "recall@1": 1 if ranked else 0,
                "recall@3": 1 if ranked else 0,
                "recall@5": 1 if ranked else 0,
                "mrr":      1.0 if ranked else 0.0,
                "note":     "detail — measures anchor retrieval, not secondary paper id",
            }

        exp_id = case.get("expected_secondary_arxiv_id")
        exp_kw = case.get("expected_secondary_title_keywords", [])
        r1  = int(_precision_at_k(ranked, exp_id, exp_kw, 1))
        r3  = int(_precision_at_k(ranked, exp_id, exp_kw, 3))
        r5  = int(_precision_at_k(ranked, exp_id, exp_kw, 5))
        mrr = _reciprocal_rank(ranked, exp_id, exp_kw)

        # Debug info for misses
        if r5 == 0 and exp_id:
            top_srcs = [c.get("source_paper_id", "?")[:30] for c in ranked[:5]]
            return {**case, "recall@1": r1, "recall@3": r3, "recall@5": r5,
                    "mrr": round(mrr, 3), "_debug_top_srcs": top_srcs}
        return {**case, "recall@1": r1, "recall@3": r3, "recall@5": r5, "mrr": round(mrr, 3)}

    print()
    for case in cases:
        print(f"  Evaluating {case['case_id']} ({case['question_type']}/{case['sub_type']})…")
        results.append(run_case(case))

    # Aggregate
    def aggregate(subset: list[dict[str, Any]], label: str) -> dict[str, Any]:
        n = len(subset)
        if n == 0:
            return {"label": label, "n": 0}
        return {
            "label":    label,
            "n":        n,
            "recall@1": round(sum(r["recall@1"] for r in subset) / n, 3),
            "recall@3": round(sum(r["recall@3"] for r in subset) / n, 3),
            "recall@5": round(sum(r["recall@5"] for r in subset) / n, 3),
            "mrr":      round(sum(r["mrr"] for r in subset) / n, 3),
        }

    locality_results = [r for r in results if r["question_type"] == "locality"]
    detail_results   = [r for r in results if r["question_type"] == "detail"]
    summary = {
        "all":      aggregate(results, "all"),
        "locality": aggregate(locality_results, "locality"),
        "detail":   aggregate(detail_results, "detail"),
        "cases":    results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {RESULTS_JSON}")

    # Markdown report
    lines = [
        "# ScholAR Multi-Document Evaluation Report",
        "",
        "## Summary",
        "",
        "| Subset | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for agg in [summary["all"], summary["locality"], summary["detail"]]:
        lines.append(
            f"| `{agg['label']}` | {agg['n']} |"
            f" {agg.get('recall@1', '-')} | {agg.get('recall@3', '-')} |"
            f" {agg.get('recall@5', '-')} | {agg.get('mrr', '-')} |"
        )
    lines += [
        "",
        "## Per-Case Results",
        "",
        "| ID | Type | Sub-type | R@1 | R@3 | R@5 | MRR | Expected |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for r in results:
        exp = r.get("expected_secondary_arxiv_id") or ", ".join(
            r.get("expected_secondary_title_keywords", [])[:2]
        )
        lines.append(
            f"| {r['case_id']} | {r['question_type']} | {r['sub_type']} |"
            f" {r['recall@1']} | {r['recall@3']} | {r['recall@5']} | {r['mrr']} | {exp or '-'} |"
        )
    lines += [
        "",
        "## How to Read",
        "",
        "- **Locality cases** test whether the correct *secondary paper* appears in the top-k retrieved chunks.",
        "- **Detail cases** test whether the anchor paper's own chunks are retrieved for direct factual questions.",
        "- Metrics are comparable to `retrieval_eval_report.md` (same Recall@k / MRR formulas).",
        "",
        "_Generated by `evaluation/run_multidoc_eval.py`_",
    ]
    RESULTS_MD.write_text("\n".join(lines))
    print(f"Wrote {RESULTS_MD}")

    # Terminal summary
    print()
    for agg in [summary["all"], summary["locality"], summary["detail"]]:
        print(
            f"{agg['label']:12s}  R@1={agg.get('recall@1', '-')}  "
            f"R@3={agg.get('recall@3', '-')}  R@5={agg.get('recall@5', '-')}  "
            f"MRR={agg.get('mrr', '-')}"
        )

    # Show debug info for missed locality cases
    misses = [r for r in results if r["question_type"] == "locality" and r["recall@5"] == 0 and r.get("expected_secondary_arxiv_id")]
    if misses:
        print(f"\n  {len(misses)} locality misses (expected paper not in top-5):")
        for r in misses:
            srcs = r.get("_debug_top_srcs", [])
            print(f"    {r['case_id']} | expected: {r.get('expected_secondary_arxiv_id')} | top-5 srcs: {srcs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScholAR multi-document evaluation")
    parser.add_argument(
        "--no-ingest",
        action="store_true",
        help="Skip downloading secondary papers (offline mode). Only evaluates what is already local.",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(do_ingest=not args.no_ingest))
