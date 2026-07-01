#!/usr/bin/env python3
"""evaluation/run_visual_eval.py — Visual grounding evaluation for ScholAR.

Evaluates how well the figure/table extraction + vision-LLM pipeline handles
questions grounded in figures and tables, using the 18-case visual_benchmark.json.

Each case is evaluated on:
  (a) Retrieval grounding — did the correct figure appear in top-5 retrieved chunks?
  (b) Answer quality    — did the vision model answer correctly (judged by string match
                          + keyword overlap heuristic; no secondary LLM judge used)?

Failure modes (M3SciQA taxonomy):
  correct                         — retrieved correct figure AND answered correctly
  correct_reasoning_wrong_grounding — answered partially correctly but wrong figure retrieved
  wrong_grounding                 — wrong figure in top-5; answer based on wrong chunk
  vision_fallback                 — vision call failed, answer was caption-only
  extraction_missing              — figure not extracted from PDF at all

Usage:
    cd /path/to/ScholAR
    python3 evaluation/run_visual_eval.py [--no-llm] [--paper-id ARXIV_ID]

    --no-llm    Skip vision-LLM calls; test retrieval grounding only
    --paper-id  Run only cases for this anchor paper (e.g. 1706.03762)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.chunking_service import chunk_figures, chunk_pages
from backend.services.pdf_service import paper_dir, read_json
from backend.services.retrieval_service import retrieve_chunks
from backend.services.vision_service import answer_with_figure

BENCHMARK_PATH = ROOT / "evaluation" / "visual_benchmark.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_paper_chunks(paper_id: str) -> list[dict]:
    """Load text + figure chunks for a paper."""
    d = paper_dir(paper_id)
    text_chunks: list[dict] = []
    pages_path = d / "pages.json"
    chunks_path = d / "chunks.json"
    if chunks_path.exists():
        text_chunks = read_json(chunks_path)
    elif pages_path.exists():
        text_chunks = chunk_pages(read_json(pages_path))

    fig_chunks: list[dict] = []
    figures_path = d / "figures.json"
    if figures_path.exists():
        fig_chunks = chunk_figures(read_json(figures_path), source_paper_id=paper_id)

    return text_chunks + fig_chunks


def _top_figure_chunks(question: str, chunks: list[dict], k: int = 5) -> list[dict]:
    """Return top-k retrieved chunks, keeping only figure chunks."""
    selected = retrieve_chunks(question, chunks, limit=k)
    return [c for c in selected if c.get("is_figure_chunk")]


def _label_matches(chunk: dict, expected_label: str, expected_prefix: str) -> bool:
    """Check whether a retrieved figure chunk matches the expected figure."""
    label = (chunk.get("label") or "").lower()
    fig_id = (chunk.get("figure_id") or "")
    expected_label_lower = expected_label.lower()
    # Exact label match (e.g. "figure 1" in "figure 1: the transformer...")
    if expected_label_lower in label:
        return True
    # Prefix match on figure_id (e.g. "fig_03" prefix in "fig_03_005")
    if expected_prefix and fig_id.startswith(expected_prefix):
        return True
    return False


def _keyword_answer_score(answer: str, question: str) -> float:
    """Heuristic answer quality score: fraction of non-trivial question tokens in answer."""
    stop = {"what", "does", "show", "which", "how", "the", "a", "an", "is", "in", "on", "of", "to"}
    q_tokens = {t.lower() for t in re.findall(r"[a-z]+", question) if t.lower() not in stop and len(t) > 3}
    if not q_tokens:
        return 0.0
    a_lower = answer.lower()
    matched = sum(1 for t in q_tokens if t in a_lower)
    return matched / len(q_tokens)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

async def evaluate_case(
    case: dict,
    no_llm: bool,
) -> dict:
    paper_id = case["anchor_paper"]
    question = case["question"]
    expected_label = case.get("expected_figure_label", "")
    expected_prefix = case.get("expected_figure_id_prefix", "")
    reasoning_type = case.get("reasoning_type", "unknown")

    chunks = _load_paper_chunks(paper_id)
    fig_chunks_all = [c for c in chunks if c.get("is_figure_chunk")]

    # — Check extraction —
    if not fig_chunks_all:
        return {
            "case_id": case["case_id"],
            "reasoning_type": reasoning_type,
            "failure_mode": "extraction_missing",
            "retrieved_correct": False,
            "answer_score": 0.0,
            "note": "No figures extracted for this paper",
        }

    # — Retrieval grounding —
    top_figs = _top_figure_chunks(question, chunks, k=5)
    retrieved_correct = any(
        _label_matches(c, expected_label, expected_prefix)
        for c in top_figs
    )

    if no_llm:
        return {
            "case_id": case["case_id"],
            "reasoning_type": reasoning_type,
            "failure_mode": "correct" if retrieved_correct else "wrong_grounding",
            "retrieved_correct": retrieved_correct,
            "answer_score": None,
            "top_retrieved": [c.get("label") for c in top_figs],
            "note": "LLM call skipped (--no-llm)",
        }

    # — Vision answer —
    top_fig_chunk = top_figs[0] if top_figs else None
    if top_fig_chunk is None:
        return {
            "case_id": case["case_id"],
            "reasoning_type": reasoning_type,
            "failure_mode": "wrong_grounding",
            "retrieved_correct": False,
            "answer_score": 0.0,
            "note": "No figure chunk in top-5",
        }

    text_support = [c for c in retrieve_chunks(question, chunks, limit=8) if not c.get("is_figure_chunk")]
    result = await answer_with_figure(
        question=question,
        figure_chunk=top_fig_chunk,
        context_chunks=text_support,
        paper_id=paper_id,
        paper_metadata={"title": f"arXiv:{paper_id}"},
    )

    answer = result.get("answer", "")
    fallback = result.get("fallback", False)
    answer_score = _keyword_answer_score(answer, question)

    if fallback:
        failure_mode = "vision_fallback"
    elif retrieved_correct and answer_score >= 0.4:
        failure_mode = "correct"
    elif not retrieved_correct and answer_score >= 0.4:
        failure_mode = "correct_reasoning_wrong_grounding"
    else:
        failure_mode = "wrong_grounding"

    return {
        "case_id": case["case_id"],
        "reasoning_type": reasoning_type,
        "failure_mode": failure_mode,
        "retrieved_correct": retrieved_correct,
        "answer_score": round(answer_score, 3),
        "retrieved_label": top_fig_chunk.get("label"),
        "expected_label": expected_label,
        "answer_snippet": answer[:300],
        "vision_fallback": fallback,
    }


async def main(args: argparse.Namespace) -> None:
    cases = json.loads(BENCHMARK_PATH.read_text())
    if args.paper_id:
        cases = [c for c in cases if c["anchor_paper"] == args.paper_id]

    print(f"\nRunning visual evaluation on {len(cases)} cases (no-llm={args.no_llm})\n")

    results: list[dict] = []
    for case in cases:
        print(f"  [{case['case_id']}] ({case['reasoning_type']}) {case['question'][:70]}...")
        result = await evaluate_case(case, no_llm=args.no_llm)
        results.append(result)
        fm = result["failure_mode"]
        rc = "✓" if result["retrieved_correct"] else "✗"
        score = f"  ans={result['answer_score']:.3f}" if result["answer_score"] is not None else ""
        print(f"         {rc} retrieved  |  {fm}{score}")

    # — Aggregate metrics —
    print("\n" + "─" * 70)
    print("VISUAL GROUNDING EVALUATION SUMMARY")
    print("─" * 70)

    total = len(results)
    by_type: dict[str, list[dict]] = {}
    for r in results:
        by_type.setdefault(r["reasoning_type"], []).append(r)

    # Overall retrieval recall
    correct_retrieval = sum(1 for r in results if r["retrieved_correct"])
    print(f"\nRetrieval Grounding  R@5 = {correct_retrieval}/{total} = {correct_retrieval/total:.3f}")

    # Answer quality (when LLM was called)
    scored = [r for r in results if r.get("answer_score") is not None]
    if scored:
        avg_score = sum(r["answer_score"] for r in scored) / len(scored)
        print(f"Avg Answer Score     {avg_score:.3f}  (keyword overlap heuristic, n={len(scored)})")

    # Failure mode breakdown
    print("\nFailure Mode Breakdown:")
    modes = ["correct", "correct_reasoning_wrong_grounding", "wrong_grounding", "vision_fallback", "extraction_missing"]
    for mode in modes:
        n = sum(1 for r in results if r["failure_mode"] == mode)
        if n:
            print(f"  {mode:<42} {n:>3} ({n/total*100:.1f}%)")

    # Per-reasoning-type recall
    print("\nRetrieval by Reasoning Type:")
    for rtype, rs in sorted(by_type.items()):
        rc = sum(1 for r in rs if r["retrieved_correct"])
        print(f"  {rtype:<35} R@5={rc}/{len(rs)} = {rc/len(rs):.3f}")

    print("─" * 70)

    # Write results
    out_path = ROOT / "evaluation" / "visual_eval_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nDetailed results → {out_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ScholAR visual grounding evaluation")
    parser.add_argument("--no-llm", action="store_true", help="Test retrieval only, skip vision model calls")
    parser.add_argument("--paper-id", default="", help="Run only cases for this arXiv paper ID")
    asyncio.run(main(parser.parse_args()))
