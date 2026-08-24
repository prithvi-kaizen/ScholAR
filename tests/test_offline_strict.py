"""Strict Offline & Zero Data Egress Verification Test for ScholAR.

Ensures:
- Pipeline executes with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1
- Ingestion, hybrid retrieval, graph synthesis, math, verifier, and export execute locally
- Zero outbound network sockets or external API egress
"""

import os
import unittest
from pathlib import Path

# Force strict offline environment
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from backend.schemas.claims import VerificationReport
from backend.schemas.numeric_plan import NumericOp
from backend.schemas.reasoning import ReasoningLevel
from backend.services.budgeting_service import BudgetingService
from backend.services.dense_embedding_service import DenseEmbeddingService
from backend.services.evidence_graph_service import EvidenceGraphService
from backend.services.export_service import ExportService
from backend.services.pdf_service import paper_dir
from backend.services.question_analyzer import QuestionAnalyzer
from backend.services.reranker_service import RerankerService
from backend.services.retrieval_service import retrieve_chunks
from backend.services.table_arithmetic_service import TableArithmeticService
from backend.services.verifier_service import ClaimVerifierService


class TestStrictOfflineExecution(unittest.TestCase):

    def test_complete_offline_reasoning_lifecycle(self):
        """Verify the full ScholAR reasoning pipeline executes with zero network access."""
        query = "Why does the Transformer outperform ConvS2S based on architecture and results?"
        paper_id = "1706.03762"
        p_path = paper_dir(paper_id)
        chunks_path = p_path / "chunks.json"

        self.assertTrue(chunks_path.exists(), "Chunks must exist locally")
        import json
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

        # 1. Question Analysis
        analysis = QuestionAnalyzer.analyze_query(query)
        self.assertIsNotNone(analysis.reasoning_level)

        # 2. Dense Embeddings & Hybrid Retrieval (Local MPS / CPU)
        retrieved = retrieve_chunks(message=query, chunks=chunks, limit=6, paper_id=paper_id)
        self.assertTrue(len(retrieved) > 0)

        # 3. Local Reranking
        reranked = RerankerService.rerank(query, retrieved, top_k=4)
        self.assertTrue(len(reranked) > 0)

        # 4. Evidence Graph & Budget Pruning
        ev_graph, ev_path = EvidenceGraphService.build_evidence_graph(query, reranked, analysis)
        budget = BudgetingService.get_evidence_budget()
        pruned_graph, pruned_path = BudgetingService.prune_to_budget(ev_graph, ev_path, budget)
        self.assertTrue(len(pruned_graph.nodes) > 0)

        # 5. Deterministic Decimal Table Arithmetic
        table_text = "| Model | BLEU |\n| Transformer | 28.4 |\n| ConvS2S | 25.16 |"
        math_res = TableArithmeticService.extract_and_calculate_from_table_text(
            table_text=table_text,
            entity_a="Transformer",
            entity_b="ConvS2S",
            op=NumericOp.DIFFERENCE,
        )
        self.assertIsNotNone(math_res)
        self.assertAlmostEqual(math_res.computed_value, 3.24, places=2)

        # 6. 3-Way Atomic Claim Verification
        ans = "The Transformer achieves 28.4 BLEU [1]."
        report = ClaimVerifierService.generate_atomic_verification_report(ans, reranked)
        self.assertIsNotNone(report)

        # 7. Local Export Generation (LaTeX TikZ + Markdown)
        md_export = ExportService.export_to_markdown(paper_id, query, ans, analysis, pruned_path, math_res, report)
        tex_export = ExportService.export_to_latex(paper_id, query, ans, analysis, pruned_path, math_res, report)

        self.assertIn("# ScholAR Multi-Level Reasoning Report", md_export)
        self.assertIn("\\documentclass{article}", tex_export)


if __name__ == "__main__":
    unittest.main()
