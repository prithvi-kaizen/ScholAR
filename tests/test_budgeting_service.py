"""Unit tests for Phase 5: Capability-Adaptive Evidence Budgeting."""

import unittest
from backend.schemas.capabilities import EvidenceBudget, HardwareTier
from backend.schemas.evidence_graph import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    ReasoningPath,
    ReasoningPathStep,
)
from backend.services.budgeting_service import BudgetingService


class TestBudgetingService(unittest.TestCase):

    def test_budget_tiers(self):
        """Verify budget parameters across 8GB, 16GB, and 32GB+ tiers."""
        budget_8gb = EvidenceBudget(hardware_tier=HardwareTier.TIER_8GB, max_evidence_blocks=4, max_table_blocks=1)
        self.assertEqual(budget_8gb.max_evidence_blocks, 4)
        self.assertEqual(budget_8gb.max_table_blocks, 1)

        budget_16gb = EvidenceBudget(hardware_tier=HardwareTier.TIER_16GB, max_evidence_blocks=6, max_table_blocks=2)
        self.assertEqual(budget_16gb.max_evidence_blocks, 6)

        budget_32gb = EvidenceBudget(hardware_tier=HardwareTier.TIER_32GB_PLUS, max_evidence_blocks=10)
        self.assertEqual(budget_32gb.max_evidence_blocks, 10)

    def test_prune_graph_preserves_bridge_nodes(self):
        """Verify that pruning a graph preserves high-priority reasoning bridge nodes."""
        nodes = [
            EvidenceNode(node_id=f"E_PERIPH_{i}", document_id="doc", page=i, reasoning_role="primary_evidence")
            for i in range(1, 8)
        ]
        # Add critical bridge nodes
        nodes.append(EvidenceNode(node_id="E_METHOD", document_id="doc", page=3, reasoning_role="method_definition"))
        nodes.append(EvidenceNode(node_id="E_RESULT", document_id="doc", page=8, reasoning_role="final_result"))

        steps = [
            ReasoningPathStep(step_index=idx, evidence_id=n.node_id, role=n.reasoning_role)
            for idx, n in enumerate(nodes, start=1)
        ]

        graph = EvidenceGraph(nodes=nodes, edges=[], query="Test Query")
        path = ReasoningPath(query="Test Query", reasoning_level="L5", steps=steps, graph=graph)

        budget = EvidenceBudget(hardware_tier=HardwareTier.TIER_8GB, max_evidence_blocks=4)
        pruned_graph, pruned_path = BudgetingService.prune_to_budget(graph, path, budget)

        self.assertLessEqual(len(pruned_graph.nodes), 4)
        pruned_node_ids = {n.node_id for n in pruned_graph.nodes}
        self.assertIn("E_METHOD", pruned_node_ids)
        self.assertIn("E_RESULT", pruned_node_ids)


if __name__ == "__main__":
    unittest.main()
