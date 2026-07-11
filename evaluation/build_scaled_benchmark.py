"""
build_scaled_benchmark.py
Auto-label the diverse 100-case benchmark (25 papers) with gold supporting chunks, to
scale the labeled retrieval + retrieval-support benchmarks from 3 papers to 25 WITHOUT
any model generation.

How the gold chunk is derived (and why it is trustworthy): mine_cases.py copied 2-4
EXACT substrings from the source passage into each case's `must_include`, and kept the
case only if those facts appear in the passage. So the gold chunk is simply the chunk
whose text contains those substrings. `answer_locus` (page) breaks ties toward the page
the fact was mined from.

Caveat (disclosed in the paper): auto-derived labels are weaker than hand labels, and
the questions were mined FROM passages, so they may favour lexical retrieval. The
3-paper hand-labeled set is kept as the higher-precision anchor.

Emits, matching the existing eval schemas:
  evaluation/benchmark_cases_scaled.json      (retrieval)
  evaluation/faithfulness_cases_scaled.json   (retrieval-support faithfulness)

Run from repo root:  python3 evaluation/build_scaled_benchmark.py [--selfcheck]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES = EVAL_DIR / "human_eval" / "cases.json"
OUT_RETRIEVAL = EVAL_DIR / "benchmark_cases_scaled.json"
OUT_FAITH = EVAL_DIR / "faithfulness_cases_scaled.json"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _page_of(locus) -> int | None:
    m = re.search(r"\d+", str(locus or ""))
    return int(m.group(0)) if m else None


def gold_chunks(case: dict, chunks: list[dict]) -> list[dict]:
    """Chunks that contain at least half of the case's must_include substrings.
    A chunk on the answer_locus page is preferred as the primary (first)."""
    must = case.get("must_include") or []
    need = max(1, (len(must) + 1) // 2)
    hits = []
    for ch in chunks:
        text = _norm(ch.get("text", ""))
        if must and sum(1 for m in must if _norm(m) in text) >= need:
            hits.append(ch)
    locus = _page_of(case.get("answer_locus"))
    hits.sort(key=lambda ch: (ch.get("page") != locus, str(ch.get("chunk_id"))))
    return hits


def build() -> tuple[list[dict], list[dict], dict]:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    retrieval, faith = [], []
    covered = 0
    for c in cases:
        chunks = json.loads((DATA_DIR / c["paper_id"] / "chunks.json").read_text())
        gold = gold_chunks(c, chunks)
        if not gold:
            continue  # (coverage is 100% in practice; guard anyway)
        covered += 1
        ids = [g["chunk_id"] for g in gold]
        pages = sorted({g.get("page") for g in gold if g.get("page") is not None})
        retrieval.append({
            "id": c["case_id"], "paper_id": c["paper_id"], "query": c["question"],
            "expected_pages": pages, "relevant_chunk_ids": ids,
            "reason": f"auto-labeled from mined must_include ({c['capability']})",
        })
        faith.append({
            "id": c["case_id"], "paper_id": c["paper_id"], "query": c["question"],
            "expected_claim": c.get("gold_answer") or c["question"],
            "supporting_chunk_id": ids[0], "claim_type": c["capability"],
        })
    stats = {"total": len(cases), "covered": covered,
             "retrieval_cases": len(retrieval), "faith_cases": len(faith)}
    return retrieval, faith, stats


def _selfcheck() -> None:
    case = {"must_include": ["BM25 k1 of 1.4", "b of 0.72"], "answer_locus": "page 2",
            "paper_id": "x", "question": "q", "gold_answer": "1.4", "capability": "single_doc_text",
            "case_id": "t1"}
    chunks = [{"chunk_id": "c1", "page": 1, "text": "unrelated text about training"},
              {"chunk_id": "c2", "page": 2, "text": "The ranker uses BM25 k1 of 1.4 and b of 0.72."}]
    g = gold_chunks(case, chunks)
    assert [x["chunk_id"] for x in g] == ["c2"], g
    # page tie-break: fact on two pages -> answer_locus page wins the primary slot
    chunks2 = [{"chunk_id": "a", "page": 5, "text": "BM25 k1 of 1.4 and b of 0.72 here"},
               {"chunk_id": "b", "page": 2, "text": "BM25 k1 of 1.4 and b of 0.72 too"}]
    g2 = gold_chunks({**case, "answer_locus": "page 2"}, chunks2)
    assert g2[0]["chunk_id"] == "b", g2
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
        sys.exit()
    retrieval, faith, stats = build()
    OUT_RETRIEVAL.write_text(json.dumps(retrieval, indent=2, ensure_ascii=False))
    OUT_FAITH.write_text(json.dumps(faith, indent=2, ensure_ascii=False))
    print(f"coverage: {stats['covered']}/{stats['total']} cases labeled")
    print(f"wrote {OUT_RETRIEVAL.name} ({stats['retrieval_cases']} cases)")
    print(f"wrote {OUT_FAITH.name} ({stats['faith_cases']} cases)")
    papers = len({r['paper_id'] for r in retrieval})
    print(f"papers covered: {papers}")
