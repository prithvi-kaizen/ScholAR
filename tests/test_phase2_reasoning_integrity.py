"""Unit tests for Phase 2: Reasoning Pipeline & Evidence Selection Integrity."""

import unittest
from unittest.mock import patch, MagicMock
from backend.schemas.capabilities import EvidenceBudget, HardwareTier
from backend.schemas.evidence_graph import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRelation,
    ReasoningPath,
    ReasoningPathStep,
)
from backend.schemas.numeric_plan import (
    CellOperand,
    NumericExecutionResult,
    NumericOp,
    NumericPlan,
)
from backend.schemas.reasoning_plan import (
    AnswerDraft,
    EvidenceBundle,
    EvidenceRequirement,
    NodeOperation,
    NodeResult,
    NodeStatus,
    PlanNode,
    ReasoningPlan,
)
from backend.services.budgeting_service import BudgetingService
from backend.services.cross_document_reasoning_service import CrossDocumentReasoningService
from backend.services.multi_hop_service import MultiHopRetrievalService
from backend.services.answer_pipeline import _build_prompt, _extractive_tutor_answer, AnswerPipelineRequest


class TestPhase2ReasoningIntegrity(unittest.TestCase):

    def test_reasoning_plan_schema_dag_validation(self):
        """ReasoningPlan must validate DAG acyclicity and unique IDs."""
        n1 = PlanNode(node_id="N1", operation=NodeOperation.RETRIEVE_TEXT)
        n2 = PlanNode(node_id="N2", operation=NodeOperation.CALCULATE, depends_on=["N1"])
        plan = ReasoningPlan(plan_id="P1", query="Query", nodes=[n1, n2])
        self.assertEqual(len(plan.nodes), 2)

        # Cyclic dependency should raise ValueError
        n1_cycle = PlanNode(node_id="N1", operation=NodeOperation.RETRIEVE_TEXT, depends_on=["N2"])
        n2_cycle = PlanNode(node_id="N2", operation=NodeOperation.CALCULATE, depends_on=["N1"])
        with self.assertRaises(ValueError):
            ReasoningPlan(plan_id="P_cycle", query="Query", nodes=[n1_cycle, n2_cycle])

    def test_answer_draft_schema(self):
        """AnswerDraft preserves exact claim-evidence bindings and generation mode."""
        draft = AnswerDraft(
            draft_id="D1",
            query="Test query",
            text="Draft answer text",
            generation_mode="extractive_fallback",
            support_ids=["E1", "E2"],
            is_fallback=True,
        )
        self.assertEqual(draft.generation_mode, "extractive_fallback")
        self.assertTrue(draft.is_fallback)
        self.assertEqual(draft.support_ids, ["E1", "E2"])

    def test_budgeting_service_enforces_table_and_visual_sublimits_when_total_within_budget(self):
        """Sublimits must be enforced even if len(nodes) <= max_evidence_blocks."""
        # Create 3 nodes: 3 tables. max_evidence_blocks=4, but max_table_blocks=1
        nodes = [
            EvidenceNode(node_id="TAB_1", document_id="doc", page=1, modality="table", score=0.9),
            EvidenceNode(node_id="TAB_2", document_id="doc", page=2, modality="table", score=0.8),
            EvidenceNode(node_id="TAB_3", document_id="doc", page=3, modality="table", score=0.7),
        ]
        graph = EvidenceGraph(nodes=nodes, edges=[], query="Table question")
        path = ReasoningPath(query="Table question", reasoning_level="L4", steps=[], graph=graph)
        budget = EvidenceBudget(
            hardware_tier=HardwareTier.TIER_8GB,
            max_evidence_blocks=4,
            max_table_blocks=1,
            max_visual_crops=0,
        )
        pruned_graph, _ = BudgetingService.prune_to_budget(graph, path, budget)
        # Should have pruned from 3 tables down to 1 table
        self.assertEqual(len(pruned_graph.nodes), 1)
        self.assertEqual(pruned_graph.nodes[0].node_id, "TAB_1")

    def test_cross_document_edges_require_semantic_concept_overlap(self):
        """Cross-document edges must not be created between unrelated nodes."""
        node_unrelated_1 = EvidenceNode(
            node_id="N_DOC1",
            document_id="doc1",
            page=1,
            text_preview="Photosynthesis in C3 and C4 plants under high temperatures.",
        )
        node_unrelated_2 = EvidenceNode(
            node_id="N_DOC2",
            document_id="doc2",
            page=1,
            text_preview="Quantum chromodynamics and gluon scattering amplitudes.",
        )
        node_related_3 = EvidenceNode(
            node_id="N_DOC3",
            document_id="doc2",
            page=2,
            text_preview="Photosynthesis rate comparison across varying light temperatures.",
        )

        dummy_chunk = {"chunk_id": "c1", "text": "Photosynthesis rate comparison across varying light temperatures."}
        with patch("backend.services.multi_hop_service.MultiHopRetrievalService.execute_multi_hop_retrieval") as mock_mh, \
             patch("backend.services.evidence_graph_service.EvidenceGraphService.build_evidence_graph") as mock_beg, \
             patch("backend.services.cross_document_reasoning_service.read_json", return_value=[dummy_chunk]), \
             patch("backend.services.cross_document_reasoning_service.paper_dir") as mock_pdir:
            mock_pdir.return_value.__truediv__.return_value.exists.return_value = True
            mock_mh.return_value = ([], MagicMock())
            mock_graph = EvidenceGraph(
                nodes=[node_unrelated_1, node_unrelated_2, node_related_3],
                edges=[],
                query="Compare photosynthesis mechanisms",
            )
            mock_beg.return_value = (mock_graph, MagicMock())

            graph, _, _ = CrossDocumentReasoningService.synthesize_cross_document_reasoning(
                query="Compare photosynthesis mechanisms",
                primary_paper_id="doc1",
                secondary_paper_ids=["doc2"],
            )

            # There should be an edge between N_DOC1 and N_DOC3 (shared: photosynthesis, temperatures)
            # but NO edge between N_DOC1 and N_DOC2 (completely unrelated)
            edge_pairs = {(e.source_id, e.target_id) for e in graph.edges}
            self.assertNotIn(("N_DOC1", "N_DOC2"), edge_pairs)
            self.assertIn(("N_DOC1", "N_DOC3"), edge_pairs)

    def test_multi_hop_retrieval_filters_ungrounded_subqueries(self):
        """Ungrounded subquery evidence must be filtered out, not appended."""
        chunks = [
            {"chunk_id": "c1", "text": "Transformer self attention mechanism is described.", "evidence_id": "E1"},
        ]
        # Query asking for completely unrelated topic in SQ2
        with patch("backend.services.question_analyzer.QuestionAnalyzer.analyze_query") as mock_analyze:
            mock_analysis = MagicMock()
            mock_analysis.reasoning_level.value = "L4_CROSS_MODAL"
            from backend.schemas.reasoning import ReasoningLevel, SubQuery, TargetModality
            mock_analysis.reasoning_level = ReasoningLevel.L4_CROSS_MODAL
            sq1 = SubQuery(subquery_id="SQ1", query_text="Transformer self attention", target_modality=TargetModality.TEXT)
            sq2 = SubQuery(subquery_id="SQ2", query_text="Unrelated astrophysical supernova explosions", target_modality=TargetModality.TEXT)
            mock_analysis.subqueries = [sq1, sq2]

            with patch("backend.services.multi_hop_service.retrieve_chunks") as mock_ret:
                # SQ1 returns c1 (relevant)
                # SQ2 returns c1 (irrelevant, overlap = 0.0)
                mock_ret.side_effect = [[chunks[0]], [chunks[0]]]

                retrieved, updated_analysis = MultiHopRetrievalService.execute_multi_hop_retrieval(
                    query="Transformer and supernova",
                    chunks=chunks,
                    limit=4,
                    analysis=mock_analysis,
                )

                # SQ1 is grounded
                self.assertTrue(updated_analysis.subqueries[0].is_grounded)
                # SQ2 is NOT grounded
                self.assertFalse(updated_analysis.subqueries[1].is_grounded)
                # Only SQ1 chunk was collected; SQ2 chunks were filtered out!
                self.assertEqual(len(retrieved), 1)
                self.assertEqual(retrieved[0]["subquery_id"], "SQ1")

    def test_numeric_result_injected_into_prompt(self):
        """Precomputed deterministic numeric result must be injected into the LLM prompt."""
        from backend.services.routing_service import QuestionRouteType
        request = AnswerPipelineRequest(paper_id="test", query="What is the difference?")
        numeric_res = NumericExecutionResult(
            operation=NumericOp.DIFFERENCE,
            computed_value=3.24,
            formatted_value="+3.24",
            formatted_statement="The difference in performance is +3.24 (A: 28.4 vs B: 25.16).",
            is_exact=True,
        )
        prompt = _build_prompt(
            request=request,
            metadata={"title": "Test Paper"},
            evidence_items=[{"evidence_id": "E1", "quote": "Test quote"}],
            secondary_meta={},
            route_type=QuestionRouteType.TABLE_NUMERIC,
            numeric_result=numeric_res,
        )
        self.assertIn("Deterministic Calculation Result", prompt)
        self.assertIn("+3.24", prompt)
        self.assertIn("The difference in performance is +3.24", prompt)


if __name__ == "__main__":
    unittest.main()
