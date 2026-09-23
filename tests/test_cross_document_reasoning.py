"""Unit tests for CrossDocumentReasoningService: Multi-Document Graph Synthesis."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.schemas.evidence_graph import EvidenceGraph, ReasoningPath
from backend.services.cross_document_reasoning_service import CrossDocumentReasoningService


class TestCrossDocumentReasoning(unittest.TestCase):

    def test_cross_document_graph_synthesis(self):
        """Verify cross-document pooling without private/local paper caches."""
        query = "How is attention adapted across these two systems?"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for paper_id, text in (
                ("1706.03762", "Multi-head attention relates tokens within a sequence."),
                ("2112.10752", "Cross-attention conditions image generation on text tokens."),
            ):
                directory = root / paper_id
                directory.mkdir()
                (directory / "chunks.json").write_text(
                    json.dumps([{"chunk_id": f"{paper_id}:1", "text": text, "page": 1}]),
                    encoding="utf-8",
                )

            graph = EvidenceGraph(query=query)
            path = ReasoningPath(
                query=query,
                reasoning_level="L5_MULTI_HOP_SYNTHESIS",
                graph=graph,
            )

            def retrieve(*, chunks, analysis, **_kwargs):
                return chunks, analysis

            with patch(
                "backend.services.cross_document_reasoning_service.paper_dir",
                side_effect=lambda paper_id: root / paper_id,
            ), patch(
                "backend.services.cross_document_reasoning_service.MultiHopRetrievalService.execute_multi_hop_retrieval",
                side_effect=retrieve,
            ), patch(
                "backend.services.cross_document_reasoning_service.EvidenceGraphService.build_evidence_graph",
                return_value=(graph, path),
            ), patch(
                "backend.services.reference_service.traverse_citation_graph",
                return_value=[],
            ):
                graph, path, chunks = CrossDocumentReasoningService.synthesize_cross_document_reasoning(
                    query=query,
                    primary_paper_id="attention_vaswani_2017",
                    secondary_paper_ids=["latent_diffusion_rombach_2022"],
                )

        self.assertIsNotNone(graph)
        self.assertIsNotNone(path)
        doc_ids = {c.get("document_id") for c in chunks}
        self.assertEqual(doc_ids, {"1706.03762", "2112.10752"})


if __name__ == "__main__":
    unittest.main()
