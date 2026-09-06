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
    def test_duplicate_local_ids_are_source_scoped_across_documents(self):
        duplicated = [
            {**self.sample_retrieved[0], "source_paper_id": "paper_a", "document_id": "paper_a"},
            {
                **self.sample_retrieved[0],
                "source_paper_id": "paper_b",
                "document_id": "paper_b",
                "text": "A distinct paper reuses the same local evidence identifier.",
            },
        ]
        graph, path = EvidenceGraphService.build_evidence_graph(
            query=self.analysis.original_query,
            retrieved_chunks=duplicated,
            analysis=self.analysis,
        )
        node_ids = [node.node_id for node in graph.nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        self.assertTrue(any(node_id.startswith("paper_a::") for node_id in node_ids))
        self.assertTrue(any(node_id.startswith("paper_b::") for node_id in node_ids))
        self.assertEqual({step.evidence_id for step in path.steps}, set(node_ids))

    def test_mlr_reasoning_modes_and_subgoals(self):
        """Verify MLR reasoning modes and concise subgoals are assigned properly."""
        from backend.schemas.evidence_graph import MLRReasoningMode

        graph, path = EvidenceGraphService.build_evidence_graph(
            query=self.analysis.original_query,
            retrieved_chunks=self.sample_retrieved,
            analysis=self.analysis,
        )

        self.assertEqual(len(path.steps), 3)

        # Step 1: method_definition on step 1 -> ProblemUnderstanding
        self.assertEqual(path.steps[0].reasoning_mode, MLRReasoningMode.PROBLEM_UNDERSTANDING)
        self.assertTrue(path.steps[0].subgoal)
        self.assertLessEqual(len(path.steps[0].subgoal.split()), 30)

        # Step 2: ablation_support -> CaseAnalysis
        self.assertEqual(path.steps[1].reasoning_mode, MLRReasoningMode.CASE_ANALYSIS)
        self.assertTrue(path.steps[1].subgoal)
        self.assertLessEqual(len(path.steps[1].subgoal.split()), 30)

        # Step 3: final_result on terminal step -> Synthesis
        self.assertEqual(path.steps[2].reasoning_mode, MLRReasoningMode.SYNTHESIS)
        self.assertTrue(path.steps[2].subgoal)
        self.assertLessEqual(len(path.steps[2].subgoal.split()), 30)

        # Nodes also reflect the assigned reasoning mode
        for node, step in zip(graph.nodes, path.steps):
            self.assertEqual(node.reasoning_mode, step.reasoning_mode)

    def test_mlr_calculation_and_visual_verification(self):
        """Verify Calculation mode for arithmetic tables and Verification mode for figures."""
        from backend.schemas.evidence_graph import MLRReasoningMode

        mixed_chunks = [
            {
                "chunk_id": "c_tab",
                "evidence_id": "E_TAB_01",
                "page": 5,
                "section": "Experiments",
                "text": "| A | 10 |\n| B | 20 |",
                "reasoning_role": "primary_evidence",
                "is_table_chunk": True,
                "is_figure_chunk": False,
            },
            {
                "chunk_id": "c_fig",
                "evidence_id": "VIS_01",
                "page": 6,
                "section": "Visualization",
                "text": "Figure 3: Attention heatmaps show localized focus.",
                "reasoning_role": "primary_evidence",
                "is_table_chunk": False,
                "is_figure_chunk": True,
            },
        ]

        analysis_with_arith = QuestionAnalysis(
            original_query="What is the difference between A and B in Figure 3?",
            reasoning_level=ReasoningLevel.L4_CROSS_MODAL,
            requires_arithmetic=True,
            requires_visual=True,
        )

        _, path = EvidenceGraphService.build_evidence_graph(
            query=analysis_with_arith.original_query,
            retrieved_chunks=mixed_chunks,
            analysis=analysis_with_arith,
        )

        self.assertEqual(len(path.steps), 2)
        # Table with arithmetic query -> Calculation
        self.assertEqual(path.steps[0].reasoning_mode, MLRReasoningMode.CALCULATION)
        self.assertTrue("table" in path.steps[0].subgoal.lower())
        # Visual figure chunk -> Verification
        self.assertEqual(path.steps[1].reasoning_mode, MLRReasoningMode.VERIFICATION)
        self.assertTrue("visual" in path.steps[1].subgoal.lower() or "figure" in path.steps[1].subgoal.lower())


if __name__ == "__main__":
    unittest.main()
