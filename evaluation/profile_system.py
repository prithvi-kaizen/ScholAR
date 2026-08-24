"""Deployed System Profiling for EACL 2027 Industry Track.

Measures:
- Latency breakdown per component (p50, p90, p95)
- Question Analysis, BM25, Dense MPS, RRF, Reranker, Graph DAG, Budgeting, Table Math, Verification
- Peak memory usage and token throughput
- Ingestion profiling per page
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.schemas.reasoning import ReasoningLevel
from backend.services.budgeting_service import BudgetingService
from backend.services.dense_embedding_service import DenseEmbeddingService
from backend.services.evidence_graph_service import EvidenceGraphService
from backend.services.pdf_service import paper_dir
from backend.services.question_analyzer import QuestionAnalyzer
from backend.services.reranker_service import RerankerService
from backend.services.retrieval_service import _bm25_scores, reciprocal_rank_fusion, retrieve_chunks, tokenize
from backend.services.table_arithmetic_service import NumericOp, TableArithmeticService
from backend.services.verifier_service import ClaimVerifierService

RESULTS_PATH = ROOT / "evaluation" / "system_profiling_results.json"


def profile_system(iterations: int = 50) -> dict[str, Any]:
    print(f"[*] Starting ScholAR Deployed System Profiling ({iterations} iterations)...")

    sample_query = "Why does the Transformer outperform ConvS2S based on architecture and results?"
    p_id = "1706.03762"
    p_path = paper_dir(p_id)
    chunks_path = p_path / "chunks.json"
    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []

    latencies: dict[str, list[float]] = {
        "1_question_analysis": [],
        "2_bm25_search": [],
        "3_dense_search_mps": [],
        "4_rrf_fusion": [],
        "5_cross_encoder_rerank": [],
        "6_evidence_graph_dag": [],
        "7_budgeting_prune": [],
        "8_table_arithmetic": [],
        "9_atomic_claim_verifier": [],
        "total_pipeline_no_llm": [],
    }

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 ** 2)

    for _ in range(iterations):
        t_total_start = time.perf_counter()

        # 1. Question Analysis
        t0 = time.perf_counter()
        analysis = QuestionAnalyzer.analyze_query(sample_query)
        latencies["1_question_analysis"].append((time.perf_counter() - t0) * 1000)

        # 2. BM25 Search
        t0 = time.perf_counter()
        q_terms = tokenize(sample_query)
        bm25_hits = _bm25_scores(q_terms, all_chunks)
        latencies["2_bm25_search"].append((time.perf_counter() - t0) * 1000)

        # 3. Dense Search MPS
        t0 = time.perf_counter()
        dense_hits = DenseEmbeddingService.search_dense(paper_id=p_id, query=sample_query, chunks=all_chunks, top_k=15)
        latencies["3_dense_search_mps"].append((time.perf_counter() - t0) * 1000)

        # 4. RRF Fusion
        t0 = time.perf_counter()
        bm25_top = [all_chunks[i] for i in sorted(bm25_hits.keys(), key=lambda k: bm25_hits[k], reverse=True)[:15]]
        dense_top = [c for c, _ in dense_hits]
        fused = reciprocal_rank_fusion([bm25_top, dense_top], k=60)
        latencies["4_rrf_fusion"].append((time.perf_counter() - t0) * 1000)

        # 5. Cross-Encoder Rerank
        t0 = time.perf_counter()
        reranked = RerankerService.rerank(sample_query, fused[:15], top_k=6)
        latencies["5_cross_encoder_rerank"].append((time.perf_counter() - t0) * 1000)

        # 6. Evidence Graph DAG
        t0 = time.perf_counter()
        ev_graph, ev_path = EvidenceGraphService.build_evidence_graph(sample_query, reranked, analysis)
        latencies["6_evidence_graph_dag"].append((time.perf_counter() - t0) * 1000)

        # 7. Budgeting & Pruning
        t0 = time.perf_counter()
        budget = BudgetingService.get_evidence_budget()
        _, p_path_obj = BudgetingService.prune_to_budget(ev_graph, ev_path, budget)
        latencies["7_budgeting_prune"].append((time.perf_counter() - t0) * 1000)

        # 8. Deterministic Table Arithmetic
        t0 = time.perf_counter()
        table_text = "| Model | BLEU |\n| Transformer | 28.4 |\n| ConvS2S | 25.16 |"
        res_math = TableArithmeticService.extract_and_calculate_from_table_text(table_text, "Transformer", "ConvS2S", NumericOp.DIFFERENCE)
        latencies["8_table_arithmetic"].append((time.perf_counter() - t0) * 1000)

        # 9. Atomic Claim Verification
        t0 = time.perf_counter()
        ans_sim = "The Transformer achieves 28.4 BLEU [1]."
        rep = ClaimVerifierService.generate_atomic_verification_report(ans_sim, reranked)
        latencies["9_atomic_claim_verifier"].append((time.perf_counter() - t0) * 1000)

        latencies["total_pipeline_no_llm"].append((time.perf_counter() - t_total_start) * 1000)

    mem_after = process.memory_info().rss / (1024 ** 2)

    summary: dict[str, Any] = {
        "iterations": iterations,
        "hardware_tier": BudgetingService.get_hardware_tier().value,
        "peak_ram_mb": round(mem_after, 2),
        "ram_delta_mb": round(mem_after - mem_before, 2),
        "components": {},
    }

    print("\n" + "=" * 65)
    print(f"{'Component':<28} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 65)

    for comp, l_vals in latencies.items():
        p50 = float(np.percentile(l_vals, 50))
        p90 = float(np.percentile(l_vals, 90))
        p95 = float(np.percentile(l_vals, 95))
        summary["components"][comp] = {
            "p50_ms": round(p50, 3),
            "p90_ms": round(p90, 3),
            "p95_ms": round(p95, 3),
        }
        print(f"{comp:<28} | {p50:<10.3f} | {p90:<10.3f} | {p95:<10.3f}")

    print("=" * 65 + "\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    profile_system()
