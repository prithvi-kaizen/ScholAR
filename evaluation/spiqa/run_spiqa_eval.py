"""run_spiqa_eval.py: Evaluate ScholAR on the SPIQA multimodal scientific QA benchmark.

SPIQA (NeurIPS 2024 Datasets and Benchmarks Track) benchmarks question answering over
complex scientific figures, plots, charts, tables, and schematic diagrams.

Supported evaluation tiers:
  1. --tier retrieval: Tests ScholAR's hybrid retriever (BM25 + Dense + Crop CLIP + ColQwen2)
     for retrieving the target visual unit and gold page. (Deterministic, no LLM required).
  2. --tier generation: Evaluates end-to-end multimodal answer generation with an installed
     local model via Ollama.
  3. --tier all: Executes both retrieval and generation passes.

Usage:
    python3 evaluation/spiqa/run_spiqa_eval.py --tier retrieval
    python3 evaluation/spiqa/run_spiqa_eval.py --tier generation --model qwen3.5:9b
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL))

from backend.services.pdf_service import paper_dir
from evaluation.benchmarks.spiqa import SPIQAAdapter, _compute_token_f1, _normalize_text

DEFAULT_CASES = HERE / "spiqa_cases_sample.json"
DEFAULT_OUTPUT = EVAL / "results" / "spiqa_results.json"


def evaluate_retrieval(adapter: SPIQAAdapter, split: str = "test") -> dict[str, Any]:
    """Evaluate multimodal retrieval hit rate and MRR against gold pages/figures."""
    from backend.services.retrieval_service import retrieve_chunks

    examples = adapter.load_examples(split=split)
    predictions: list[dict[str, Any]] = []

    for ex in examples:
        gold_pages = {ev.page for ev in ex.gold_evidence if ev.page is not None}
        p_dir = paper_dir(ex.document_id)

        if not p_dir.exists() or not (p_dir / "chunks.json").exists():
            # Paper not seeded locally; record unseeded placeholder
            predictions.append({
                "example_id": ex.example_id,
                "document_id": ex.document_id,
                "gold_pages": list(gold_pages),
                "retrieved_pages": [],
                "visual_rank": None,
                "status": "unseeded_paper",
                "figure_found": False,
            })
            continue

        try:
            hits = retrieve_chunks(
                paper_id=ex.document_id,
                query=ex.question,
                limit=10,
            )

            retrieved_pages = [int(h.get("page", 0)) for h in hits if "page" in h]

            # Find highest rank of any gold page in the retrieved set
            gold_rank: int | None = None
            for idx, p in enumerate(retrieved_pages, start=1):
                if p in gold_pages:
                    gold_rank = idx
                    break

            # Check if any retrieved chunk is a figure/table matching gold section
            figure_found = any(
                h.get("is_figure_chunk") or h.get("is_table_chunk")
                for h in hits[:3]
            )

            predictions.append({
                "example_id": ex.example_id,
                "document_id": ex.document_id,
                "visual_type": ex.metadata.get("visual_type"),
                "gold_pages": list(gold_pages),
                "retrieved_pages": retrieved_pages[:5],
                "visual_rank": gold_rank,
                "figure_found": figure_found,
                "status": "evaluated",
            })
        except Exception as exc:
            predictions.append({
                "example_id": ex.example_id,
                "document_id": ex.document_id,
                "error": str(exc),
                "visual_rank": None,
                "status": "error",
            })

    metrics = adapter.compute_metrics(predictions)
    return {
        "tier": "retrieval",
        "split": split,
        "metrics": metrics,
        "predictions": predictions,
    }


async def evaluate_generation(
    adapter: SPIQAAdapter,
    model: str,
    split: str = "test",
    backend_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """Evaluate end-to-end multimodal answer synthesis with a local model."""
    from scholar_runner import run_scholar_http

    examples = adapter.load_examples(split=split)
    predictions: list[dict[str, Any]] = []

    for ex in examples:
        p_dir = paper_dir(ex.document_id)
        if not p_dir.exists():
            predictions.append({
                "example_id": ex.example_id,
                "gold_answers": ex.gold_answers,
                "prediction": "",
                "status": "unseeded_paper",
            })
            continue

        try:
            res = run_scholar_http(
                paper_id=ex.document_id,
                query=ex.question,
                model=model,
                backend_url=backend_url,
            )
            pred_text = res.get("final_answer", "")
            verification = res.get("verification_report", {})

            predictions.append({
                "example_id": ex.example_id,
                "document_id": ex.document_id,
                "gold_answers": ex.gold_answers,
                "prediction": pred_text,
                "verified": verification.get("overall_supported", False),
                "figure_found": bool(res.get("response_metadata", {}).get("vision")),
                "status": "ok",
            })
        except Exception as exc:
            predictions.append({
                "example_id": ex.example_id,
                "document_id": ex.document_id,
                "gold_answers": ex.gold_answers,
                "prediction": "",
                "error": str(exc),
                "status": "error",
            })

    metrics = adapter.compute_metrics(predictions)
    return {
        "tier": "generation",
        "model": model,
        "split": split,
        "metrics": metrics,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SPIQA multimodal scientific QA benchmark evaluation.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="Path to SPIQA cases JSON")
    parser.add_argument("--tier", choices=["retrieval", "generation", "all"], default="retrieval", help="Evaluation tier")
    parser.add_argument("--split", choices=["test", "dev", "all"], default="test", help="Dataset split")
    parser.add_argument("--model", type=str, default="qwen3.5:9b", help="Model tag for generation tier")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to write results JSON")
    parser.add_argument("--backend-url", type=str, default="http://localhost:8000", help="FastAPI backend URL")
    args = parser.parse_args()

    adapter = SPIQAAdapter(data_path=args.cases)
    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "SPIQA",
        "cases_file": str(args.cases),
        "tiers": {},
    }

    print(f"=== Running SPIQA Benchmark Evaluation [{args.tier.upper()}] ===")
    print(f"Cases: {args.cases} | Split: {args.split}")

    if args.tier in ("retrieval", "all"):
        print("\n--> Evaluating Tier: Multimodal Retrieval & Grounding...")
        ret_res = evaluate_retrieval(adapter, split=args.split)
        results["tiers"]["retrieval"] = ret_res
        print(f"Hit@1: {ret_res['metrics']['visual_hit_rate_at_1']:.3f} | Hit@3: {ret_res['metrics']['visual_hit_rate_at_3']:.3f} | MRR: {ret_res['metrics']['visual_mrr']:.3f}")

    if args.tier in ("generation", "all"):
        print(f"\n--> Evaluating Tier: Generation (Model: {args.model})...")
        gen_res = asyncio.run(
            evaluate_generation(adapter, model=args.model, split=args.split, backend_url=args.backend_url)
        )
        results["tiers"]["generation"] = gen_res
        print(f"Token F1: {gen_res['metrics']['mean_token_f1']:.3f} | Exact Match: {gen_res['metrics']['exact_match']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[OK] Evaluation report saved to: {args.output}")


if __name__ == "__main__":
    main()
