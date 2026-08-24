"""Unit tests for Phase 3: Multi-Level Question Classifier (L1-L5) & Bounded Decomposer."""

import unittest
from backend.schemas.reasoning import (
    QuestionAnalysis,
    ReasoningLevel,
    TargetModality,
)
from backend.services.multi_hop_service import MultiHopRetrievalService
from backend.services.question_analyzer import QuestionAnalyzer


class TestQuestionAnalyzer(unittest.TestCase):

    def test_l1_direct_lookup(self):
        """Verify L1 classification on factual / hyperparameter questions."""
        q = "What learning rate was used in the experiments?"
        analysis = QuestionAnalyzer.analyze_query(q)
        self.assertEqual(analysis.reasoning_level, ReasoningLevel.L1_DIRECT_LOOKUP)
        self.assertEqual(len(analysis.subqueries), 1)

    def test_l2_same_section_reasoning(self):
        """Verify L2 classification on explanatory questions."""
        q = "Why was the residual connection architecture chosen?"
        analysis = QuestionAnalyzer.analyze_query(q)
        self.assertIn(analysis.reasoning_level, (ReasoningLevel.L2_SAME_SECTION, ReasoningLevel.L5_MULTI_HOP_SYNTHESIS))

    def test_l3_cross_section_reasoning(self):
        """Verify L3 classification on cross-section methodology <-> results questions."""
        q = "How do the experiments compare the proposed method with the baseline?"
        analysis = QuestionAnalyzer.analyze_query(q)
        self.assertIn(analysis.reasoning_level, (ReasoningLevel.L3_CROSS_SECTION, ReasoningLevel.L5_MULTI_HOP_SYNTHESIS))
        self.assertTrue(len(analysis.subqueries) >= 2)

    def test_l4_cross_modal_reasoning(self):
        """Verify L4 classification on table / figure questions."""
        q = "What is the BLEU score reported in Table 2 for the base model?"
        analysis = QuestionAnalyzer.analyze_query(q)
        self.assertEqual(analysis.reasoning_level, ReasoningLevel.L4_CROSS_MODAL)
        self.assertTrue(analysis.requires_arithmetic or TargetModality.TABLE in analysis.target_modalities)

    def test_l5_multi_hop_synthesis_bounded(self):
        """Verify L5 classification and bounded subquery decomposition (<= 3 subqueries)."""
        q = "Why does Model A outperform Model B based on the architecture, ablation study, and Table 2 results?"
        analysis = QuestionAnalyzer.analyze_query(q)
        self.assertEqual(analysis.reasoning_level, ReasoningLevel.L5_MULTI_HOP_SYNTHESIS)
        self.assertLessEqual(len(analysis.subqueries), 3)
        self.assertEqual(len(analysis.subqueries), 3)
        subquery_ids = [sq.subquery_id for sq in analysis.subqueries]
        self.assertEqual(subquery_ids, ["SQ1", "SQ2", "SQ3"])

    def test_multi_hop_retrieval_execution(self):
        """Verify MultiHopRetrievalService retrieves and attaches reasoning roles."""
        sample_chunks = [
            {
                "chunk_id": "c1",
                "evidence_id": "E_001",
                "text": "Our model architecture introduces a sparse attention module.",
                "section": "Methodology",
                "is_table_chunk": False,
                "is_figure_chunk": False,
            },
            {
                "chunk_id": "c2",
                "evidence_id": "E_002",
                "text": "Ablation study: removing sparse attention drops accuracy by 3.2%.",
                "section": "Ablation",
                "is_table_chunk": False,
                "is_figure_chunk": False,
            },
            {
                "chunk_id": "c3",
                "evidence_id": "E_TAB_01",
                "text": "| Model | Accuracy |\n| Model A | 88.4 |\n| Model B | 85.2 |",
                "section": "Results",
                "is_table_chunk": True,
                "is_figure_chunk": False,
            },
        ]

        q = "Why does Model A outperform Model B based on the architecture and ablation?"
        results, analysis = MultiHopRetrievalService.execute_multi_hop_retrieval(
            query=q,
            chunks=sample_chunks,
            limit=4,
            paper_id="test_paper",
        )

        self.assertTrue(len(results) > 0)
        self.assertIn("reasoning_role", results[0])
        self.assertIn("subquery_id", results[0])


if __name__ == "__main__":
    unittest.main()
