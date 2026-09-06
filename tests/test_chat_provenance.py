"""Regression tests for source-paper provenance through prompt evidence and citations."""

import unittest

from backend.services.answer_pipeline import (
    _build_answer_citations,
    _build_evidence_items,
    _format_evidence_context,
    _normalize_evidence_citations,
)


class TestChatProvenance(unittest.TestCase):
    def setUp(self) -> None:
        self.anchor = {
            "source_paper_id": "anchor",
            "document_id": "anchor",
            "chunk_id": "chunk_001",
            "evidence_id": "A_001",
            "page": 2,
            "section_title": "Method",
            "chunk_type": "method",
            "text": "We propose an attention retrieval method that improves scientific search accuracy.",
        }
        self.reference = {
            "source_paper_id": "reference",
            "document_id": "reference",
            "chunk_id": "chunk_001",
            "evidence_id": "R_001",
            "page": 7,
            "section_title": "Results",
            "chunk_type": "result",
            "text": "The reference experiment reports improved retrieval accuracy on the benchmark.",
        }

    def test_evidence_items_keep_same_local_id_from_distinct_sources(self) -> None:
        items = _build_evidence_items(
            "What improves retrieval accuracy?", [self.anchor, self.reference], limit=4
        )
        self.assertEqual(len(items), 2)
        self.assertEqual({item["source_paper_id"] for item in items}, {"anchor", "reference"})
        self.assertEqual({item["chunk_id"] for item in items}, {"chunk_001"})
        self.assertEqual({item["source_evidence_id"] for item in items}, {"A_001", "R_001"})

        context = _format_evidence_context(
            items, secondary_meta={"reference": {"title": "Reference Paper"}}
        )
        self.assertIn("anchor", context)
        self.assertIn("ref:Reference Paper", context)

    def test_normalized_citation_preserves_source_identity(self) -> None:
        items = _build_evidence_items(
            "What improves retrieval accuracy?", [self.reference], limit=1
        )
        answer, citations = _normalize_evidence_citations("Accuracy improves [E1].", items)
        self.assertEqual(answer, "Accuracy improves [1].")
        self.assertEqual(citations[0]["source_paper_id"], "reference")
        self.assertEqual(citations[0]["document_id"], "reference")
        self.assertEqual(citations[0]["source_evidence_id"], "R_001")

    def test_extractively_built_citation_preserves_source_identity(self) -> None:
        citations = _build_answer_citations(
            "The reference experiment improves retrieval accuracy.",
            "What improves retrieval accuracy?",
            [self.reference],
            limit=1,
        )
        self.assertEqual(citations[0]["source_paper_id"], "reference")
        self.assertEqual(citations[0]["source_evidence_id"], "R_001")


if __name__ == "__main__":
    unittest.main()
