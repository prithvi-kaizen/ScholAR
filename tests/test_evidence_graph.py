"""Unit tests for Phase 4: Evidence Graph & Structured Reasoning Path Builder."""

import unittest
from backend.schemas.evidence_graph import (
    EvidenceGraph,
    EvidenceRelation,
    ReasoningPath,
)
from backend.schemas.reasoning import (
    QuestionAnalysis,
    ReasoningLevel,
    SubQuery,
    TargetModality,
)
from backend.services.evidence_graph_service import EvidenceGraphService


class TestEvidenceGraph(unittest.TestCase):

    def setUp(self):
        self.sample_retrieved = [
            {
                "chunk_id": "c1",
                "evidence_id": "E_001",
                "document_id": "transformer_paper",
                "page": 3,
                "section": "3. Model Architecture",
                "text": "The Transformer uses multi-head attention instead of recurrent layers.",
                "reasoning_role": "method_definition",
                "is_table_chunk": False,
                "is_figure_chunk": False,
                "rerank_score": 0.95,
            },
            {
                "chunk_id": "c2",
                "evidence_id": "E_002",
                "document_id": "transformer_paper",
                "page": 7,
                "section": "5.4 Ablation Studies",
                "text": "Table 3 shows that reducing attention heads drops BLEU score by 1.1 points.",
                "reasoning_role": "ablation_support",
                "is_table_chunk": False,
                "is_figure_chunk": False,
                "rerank_score": 0.88,
            },
            {
                "chunk_id": "c3",
                "evidence_id": "E_TAB_01",
                "document_id": "transformer_paper",
                "page": 8,
                "section": "5.1 Machine Translation",
                "text": "| Model | BLEU (EN-DE) |\n| Transformer (big) | 28.4 |\n| ConvS2S | 25.16 |",
                "reasoning_role": "final_result",
                "is_table_chunk": True,
                "is_figure_chunk": False,
                "rerank_score": 0.92,
            },
        ]

        self.analysis = QuestionAnalysis(
            original_query="Why does the Transformer outperform ConvS2S based on architecture, ablations, and results?",
            reasoning_level=ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
            target_modalities=[TargetModality.TEXT, TargetModality.TABLE],
            requires_arithmetic=True,
            requires_visual=False,
            subqueries=[
                SubQuery(subquery_id="SQ1", query_text="Architecture", target_sections=["Model Architecture"]),
                SubQuery(subquery_id="SQ2", query_text="Ablation", target_sections=["Ablation Studies"]),
                SubQuery(subquery_id="SQ3", query_text="Results", target_sections=["Machine Translation"]),
            ],
        )

    def test_evidence_graph_nodes_and_edges(self):
        """Verify nodes and directed semantic edges in EvidenceGraph."""
        graph, path = EvidenceGraphService.build_evidence_graph(
            query=self.analysis.original_query,
            retrieved_chunks=self.sample_retrieved,
            analysis=self.analysis,
        )

        self.assertEqual(len(graph.nodes), 3)
        self.assertTrue(len(graph.edges) >= 2)

        # Check edge relations
        relations = [e.relation for e in graph.edges]
        self.assertIn(EvidenceRelation.ABLATION_EVIDENCE, relations)
        self.assertIn(EvidenceRelation.EXPLAINS_RESULT, relations)

    def test_reasoning_path_ordering(self):
        """Verify that ReasoningPath steps are ordered logically (method -> ablation -> result)."""
        graph, path = EvidenceGraphService.build_evidence_graph(
            query=self.analysis.original_query,
            retrieved_chunks=self.sample_retrieved,
            analysis=self.analysis,
        )

        self.assertEqual(len(path.steps), 3)
        self.assertEqual(path.steps[0].role, "method_definition")
        self.assertEqual(path.steps[1].role, "ablation_support")
        self.assertEqual(path.steps[2].role, "final_result")

        # Check evidence IDs match
        self.assertEqual(path.steps[0].evidence_id, "E_001")
        self.assertEqual(path.steps[1].evidence_id, "E_002")
        self.assertEqual(path.steps[2].evidence_id, "E_TAB_01")


if __name__ == "__main__":
    unittest.main()
