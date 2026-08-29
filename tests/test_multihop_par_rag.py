"""Unit and integration test suite for SOTA Multi-Level Reasoning enhancements (PAR-RAG, PaperQA2, MultiHop-RAG, Citation F1)."""

import unittest
from pathlib import Path
from backend.schemas.reasoning import (
    QuestionAnalysis,
    ReasoningLevel,
    SubQuery,
    TargetModality,
)
from backend.services.multi_hop_service import MultiHopRetrievalService
from backend.services.reference_service import traverse_citation_graph, save_references
from backend.services.verifier_service import ClaimVerifierService, ClaimVerificationResult, VerificationLabel
from evaluation.benchmarks.multihop_rag import MultiHopRAGAdapter


class TestMultiHopPARRAG(unittest.TestCase):

    def test_subquery_intermediate_gating_and_error_arrest(self):
        """Test PAR-RAG subquery sufficiency evaluation and error-arresting."""
        chunks = [
            {
                "chunk_id": "c1",
                "evidence_id": "E_001",
                "text": "The Transformer uses multi-head attention to attend to different representation subspaces.",
                "is_table_chunk": False,
                "is_figure_chunk": False,
            },
            {
                "chunk_id": "c2",
                "evidence_id": "E_002",
                "text": "Table 2 shows WMT 2014 English-to-German BLEU score of 28.4 for the big model.",
                "is_table_chunk": True,
                "is_figure_chunk": False,
            }
        ]

        analysis = QuestionAnalysis(
            original_query="Why does Transformer outperform ConvS2S based on multi-head attention and translation results?",
            reasoning_level=ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
            subqueries=[
                SubQuery(subquery_id="SQ1", query_text="multi-head attention mechanism architecture"),
                SubQuery(subquery_id="SQ2", query_text="quantum entanglement teleportation protocol across black holes"),  # Noise / ungrounded
            ]
        )

        retrieved, updated_analysis = MultiHopRetrievalService.execute_multi_hop_retrieval(
            query=analysis.original_query,
            chunks=chunks,
            limit=4,
            analysis=analysis,
        )

        # SQ1 should be grounded and have evidence
        sq1 = updated_analysis.subqueries[0]
        self.assertTrue(sq1.is_grounded)
        self.assertGreater(sq1.sufficiency_score, 0.1)
        self.assertTrue(len(sq1.retrieved_evidence_ids) > 0)

        # SQ2 (ungrounded query) should be arrested / low sufficiency
        sq2 = updated_analysis.subqueries[1]
        self.assertFalse(sq2.is_grounded)
        self.assertLess(sq2.sufficiency_score, 0.15)

    def test_autonomous_citation_traversal(self):
        """Test PaperQA2 citation graph traversal and anchor text extraction."""
        sample_refs = [
            {
                "title": "Adam: A Method for Stochastic Optimization",
                "authors": ["Diederik P. Kingma", "Jimmy Ba"],
                "year": 2014,
                "arxiv_id": "1412.6980",
            },
            {
                "title": "Convolutional Sequence to Sequence Learning",
                "authors": ["Jonas Gehring", "Michael Auli"],
                "year": 2017,
                "arxiv_id": "1705.03122",
            }
        ]
        save_references("test_paper_1706", sample_refs)

        sample_chunks = [
            {
                "chunk_id": "chunk_opt_1",
                "page": 7,
                "section": "Optimizer",
                "text": "We used the Adam optimizer (Kingma and Ba, 2014) with beta1=0.9, beta2=0.98 and eps=1e-9 [1].",
            },
            {
                "chunk_id": "chunk_res_1",
                "page": 8,
                "section": "Results",
                "text": "On English-to-German, our model outperforms ConvS2S (Gehring et al., 2017) by 2.0 BLEU.",
            }
        ]

        traversal = traverse_citation_graph(
            paper_id="test_paper_1706",
            query_entities=["Adam", "Kingma", "optimization"],
            chunks=sample_chunks,
        )

        self.assertGreater(len(traversal), 0)
        self.assertEqual(traversal[0]["target_title"], "Adam: A Method for Stochastic Optimization")
        self.assertGreater(len(traversal[0]["citing_anchors"]), 0)
        self.assertEqual(traversal[0]["citing_anchors"][0]["chunk_id"], "chunk_opt_1")

    def test_multihop_rag_adapter_and_metrics(self):
        """Test MultiHop-RAG benchmark adapter and evaluation metric computation."""
        adapter = MultiHopRAGAdapter()
        examples = adapter.load_examples()
        self.assertGreater(len(examples), 0)

        sample_predictions = [
            {
                "example_id": "multihop_01",
                "prediction": "The Transformer was trained using the Adam optimizer with beta1=0.9 and dynamic warmup.",
                "gold_answers": ["The Transformer was trained using the Adam optimizer with beta1=0.9."],
                "gold_evidence": [{"source_id": "1706.03762_p7"}, {"source_id": "1412.6980_p2"}],
                "retrieved_sources": ["1706.03762_p7", "1412.6980_p2"],
                "answerable": True,
            },
            {
                "example_id": "multihop_03",
                "prediction": "Insufficient evidence in document to determine adversarial loss comparison.",
                "gold_answers": [],
                "gold_evidence": [],
                "retrieved_sources": [],
                "answerable": False,
                "abstained": True,
            }
        ]

        metrics = adapter.compute_metrics(sample_predictions)
        self.assertIn("multihop_exact_match", metrics)
        self.assertIn("multihop_f1", metrics)
        self.assertIn("evidence_path_recall", metrics)
        self.assertEqual(metrics["evidence_path_recall"], 1.0)
        self.assertEqual(metrics["abstention_accuracy"], 1.0)

    def test_citation_f1_and_unsupported_claim_rate(self):
        """Test calculation of formal Industry Track Citation F1 and Unsupported Claim Rate (UCR)."""
        verified_claims = [
            ClaimVerificationResult(
                claim_id="c1",
                claim_text="Transformer achieves 28.4 BLEU.",
                cited_evidence_ids=["E_001"],
                label=VerificationLabel.SUPPORTED,
                confidence=1.0,
            ),
            ClaimVerificationResult(
                claim_id="c2",
                claim_text="Residual dropout rate was set to 0.1.",
                cited_evidence_ids=["E_002"],
                label=VerificationLabel.SUPPORTED,
                confidence=0.95,
            ),
            ClaimVerificationResult(
                claim_id="c3",
                claim_text="The model was evaluated on the ImageNet visual challenge.",
                cited_evidence_ids=[],
                label=VerificationLabel.UNSUPPORTED,
                confidence=0.0,
            ),
        ]

        gold_citations = [{"source_id": "E_001"}, {"source_id": "E_002"}]
        metrics = ClaimVerifierService.compute_citation_metrics(
            verified_claims=verified_claims,
            gold_citations=gold_citations,
        )

        self.assertAlmostEqual(metrics["citation_precision"], 2 / 3, places=2)
        self.assertEqual(metrics["citation_recall"], 1.0)
        self.assertGreater(metrics["citation_f1"], 0.75)
        self.assertAlmostEqual(metrics["unsupported_claim_rate"], 1 / 3, places=2)
        self.assertEqual(metrics["total_claims"], 3)


if __name__ == "__main__":
    unittest.main()
