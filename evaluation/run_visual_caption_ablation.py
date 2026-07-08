"""
run_visual_caption_ablation.py
-------------------------------
Quantifies the marginal contribution of actually reading the figure image,
as an alternative to a full ColPali-style visual-retrieval baseline (which
would require new infrastructure this benchmark isn't built to compare
against yet).

For each of the 18 visual grounding cases, builds the identical prompt
`answer_with_figure` would send to the vision model, but calls the local
Ollama model WITHOUT the image (text-only: caption + supporting context),
and scores the answer with the same keyword-overlap heuristic used in
run_visual_eval.py. Compares this "caption-only" condition against the
already-recorded full-vision answer scores in evaluation/visual_eval_results.json.

Usage:
    python3 evaluation/run_visual_caption_ablation.py
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from backend.services.ollama_service import OLLAMA_MODEL, generate  # noqa: E402
from backend.services.vision_service import _build_vision_prompt  # noqa: E402
from run_visual_eval import (  # noqa: E402
    BENCHMARK_PATH,
    _keyword_answer_score,
    _load_paper_chunks,
    _top_figure_chunks,
)
from backend.services.retrieval_service import retrieve_chunks  # noqa: E402

RESULTS_DIR  = ROOT / "evaluation" / "results"
RESULTS_JSON = RESULTS_DIR / "visual_caption_ablation_results.json"
RESULTS_MD   = RESULTS_DIR / "visual_caption_ablation_report.md"

VISION_RESULTS_PATH = ROOT / "evaluation" / "visual_eval_results.json"


async def _caption_only_answer(case: dict) -> dict:
    paper_id = case["anchor_paper"]
    question = case["question"]
    chunks = _load_paper_chunks(paper_id)

    top_figs = _top_figure_chunks(question, chunks, k=5)
    top_fig_chunk = top_figs[0] if top_figs else None
    if top_fig_chunk is None:
        return {"case_id": case["case_id"], "answer_score": 0.0, "note": "no figure in top-5"}

    label = top_fig_chunk.get("label", "")
    caption = top_fig_chunk.get("caption", "")
    text_support = [
        c for c in retrieve_chunks(question, chunks, limit=8)
        if not c.get("is_figure_chunk") and c.get("text", "").strip()
    ][:3]
    text_context = "\n\n".join(
        "[p. " + str(c.get("page")) + "] " + c.get("text", "")[:400] for c in text_support
    )

    # Identical prompt construction to the real vision path, minus the image.
    prompt = _build_vision_prompt(question, label, caption, text_context, f"arXiv:{paper_id}")
    answer = await generate(prompt, temperature=0.1)
    score = _keyword_answer_score(answer, question)
    return {
        "case_id": case["case_id"],
        "answer_score": round(score, 3),
        "answer_snippet": answer[:300],
    }


async def main() -> None:
    cases = json.loads(BENCHMARK_PATH.read_text())
    vision_results = {
        r["case_id"]: r for r in json.loads(VISION_RESULTS_PATH.read_text())
    } if VISION_RESULTS_PATH.exists() else {}

    caption_results = []
    for case in cases:
        print(f"  {case['case_id']} ({case.get('reasoning_type')})…")
        caption_results.append(await _caption_only_answer(case))

    # Paired comparison: both means must be computed over the SAME set of cases,
    # namely those that have a real (non-None) vision answer_score. Averaging
    # caption over all 18 while averaging vision over a subset is not a paired
    # "does the image help" delta (the C8 fix). If the vision run had no scored
    # cases (e.g. produced with --no-llm), the comparison is unavailable.
    paired = [
        r for r in caption_results
        if r["case_id"] in vision_results
        and vision_results[r["case_id"]].get("answer_score") is not None
    ]
    caption_scores = [r["answer_score"] for r in paired]
    vision_scores = [vision_results[r["case_id"]]["answer_score"] for r in paired]

    summary = {
        "model": OLLAMA_MODEL,
        "n_total_cases": len(caption_results),
        "n_paired": len(paired),
        "mean_caption_only_score": round(sum(caption_scores) / len(caption_scores), 3) if caption_scores else None,
        "mean_vision_score": round(sum(vision_scores) / len(vision_scores), 3) if vision_scores else None,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps({"summary": summary, "cases": caption_results}, indent=2))

    if summary["n_paired"] == 0:
        md = [
            "# Visual Grounding: Caption-Only vs. Full-Vision Ablation",
            "",
            f"Model: `{OLLAMA_MODEL}`.",
            "",
            "Paired comparison unavailable: no case has a scored full-vision answer "
            "(the vision run may have been produced with `--no-llm`). Re-run the visual "
            "eval with answer scoring before running this ablation.",
        ]
    else:
        md = [
            "# Visual Grounding: Caption-Only vs. Full-Vision Ablation",
            "",
            f"Model: `{OLLAMA_MODEL}`. Paired over the {summary['n_paired']} of "
            f"{summary['n_total_cases']} cases that have a scored full-vision answer, "
            "same prompt template and retrieved figure, differing only in whether the "
            "figure image is sent to the model.",
            "",
            "| Condition | Mean Answer Score |",
            "|---|---:|",
            f"| Caption + context only (no image) | {summary['mean_caption_only_score']} |",
            f"| Full vision (image + caption + context) | {summary['mean_vision_score']} |",
        ]
    RESULTS_MD.write_text("\n".join(md) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {RESULTS_JSON}\nWrote {RESULTS_MD}")


if __name__ == "__main__":
    asyncio.run(main())
