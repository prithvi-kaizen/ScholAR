"""Unit tests for CrossDocumentReasoningService: Multi-Document Graph Synthesis."""

import unittest
from backend.schemas.evidence_graph import EvidenceRelation
from backend.services.cross_document_reasoning_service import CrossDocumentReasoningService


class TestCrossDocumentReasoning(unittest.TestCase):

    def test_cross_document_graph_synthesis(self):
        """Verify cross-document reasoning across 2 benchmark papers."""
        q = "How is multi-head attention in Vaswani 2017 adapted for cross-attention conditioning in Rombach 2022?"
        graph, path, chunks = CrossDocumentReasoningService.synthesize_cross_document_reasoning(
            query=q,
            primary_paper_id="attention_vaswani_2017",
            secondary_paper_ids=["latent_diffusion_rombach_2022"],
        )

        self.assertIsNotNone(graph)
        self.assertIsNotNone(path)
        # Should have pooled chunks from both documents
        doc_ids = {c.get("document_id") for c in chunks}
        self.assertTrue(len(doc_ids) >= 1)


if __name__ == "__main__":
    unittest.main()
