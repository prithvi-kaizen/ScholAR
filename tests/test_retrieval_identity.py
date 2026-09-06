"""Regression tests for deterministic, source-scoped retrieval identities."""

import unittest
from unittest.mock import patch

from backend.schemas.reasoning import QuestionAnalysis, ReasoningLevel, SubQuery
from backend.services.multi_hop_service import MultiHopRetrievalService
from backend.services.retrieval_service import (
    evidence_identity,
    evidence_key,
    reciprocal_rank_fusion,
    retrieve_chunks,
)


class TestEvidenceIdentity(unittest.TestCase):
    def test_identity_uses_source_precedence_and_content_fallback(self):
        source_chunk = {
            "source_paper_id": "paper-source",
            "document_id": "paper-document",
            "chunk_id": "chunk_001",
        }
        document_chunk = {
            "document_id": "paper-document",
            "chunk_id": "chunk_001",
        }
        caller_chunk = {"chunk_id": "chunk_001"}

        source_key = evidence_key(source_chunk, paper_id="paper-caller")
        document_key = evidence_key(document_chunk, paper_id="paper-caller")
        caller_key = evidence_key(caller_chunk, paper_id="paper-caller")

        self.assertEqual(source_key, ("paper-source", "chunk_id", "chunk_001"))
        self.assertEqual(document_key, ("paper-document", "chunk_id", "chunk_001"))
        self.assertEqual(caller_key, ("paper-caller", "chunk_id", "chunk_001"))
        self.assertEqual(len({source_key, document_key, caller_key}), 3)
        self.assertEqual(caller_chunk, {"chunk_id": "chunk_001"})

        no_id = {
            "page": 4,
            "section": "Results",
            "text": "A deterministic evidence passage.",
            "is_figure_chunk": False,
        }
        same_content = dict(no_id)
        changed_content = {**no_id, "text": "Different evidence content."}

        self.assertEqual(
            evidence_identity(no_id, paper_id="paper-caller"),
            evidence_identity(same_content, paper_id="paper-caller"),
        )
        self.assertNotEqual(
            evidence_identity(no_id, paper_id="paper-caller"),
            evidence_identity(changed_content, paper_id="paper-caller"),
        )

    def test_rrf_keeps_cross_paper_local_ids_and_accumulates_true_copies(self):
        paper_a = {
            "source_paper_id": "paper-a",
            "chunk_id": "chunk_001",
            "text": "Shared retrieval evidence from paper A.",
        }
        paper_b = {
            "source_paper_id": "paper-b",
            "chunk_id": "chunk_001",
            "text": "Shared retrieval evidence from paper B.",
        }

        fused = reciprocal_rank_fusion(
            [[paper_a, paper_b], [dict(paper_a)]],
            k=60,
            paper_id="anchor-paper",
        )

        self.assertEqual(len(fused), 2)
        by_source = {chunk["source_paper_id"]: chunk for chunk in fused}
        self.assertEqual(set(by_source), {"paper-a", "paper-b"})
        self.assertEqual(by_source["paper-a"]["rrf_score"], round(2.0 / 61, 6))
        self.assertEqual(by_source["paper-b"]["rrf_score"], round(1.0 / 62, 6))

    def test_retrieve_passes_caller_fallback_without_overriding_chunk_sources(self):
        paper_a = {
            "source_paper_id": "paper-a",
            "chunk_id": "chunk_001",
            "text": "Identity regression evidence from alpha paper.",
            "is_figure_chunk": False,
            "is_table_chunk": False,
        }
        paper_b = {
            "document_id": "paper-b",
            "chunk_id": "chunk_001",
            "text": "Identity regression evidence from beta paper.",
            "is_figure_chunk": False,
            "is_table_chunk": False,
        }

        def keep_order(_query, candidates, top_k):
            return candidates[:top_k]

        with (
            patch(
                "backend.services.retrieval_service.DenseEmbeddingService.search_dense",
                return_value=[(dict(paper_a), 0.9), (dict(paper_b), 0.8)],
            ),
            patch(
                "backend.services.retrieval_service.RerankerService.rerank",
                side_effect=keep_order,
            ),
            patch(
                "backend.services.retrieval_service.reciprocal_rank_fusion",
                wraps=reciprocal_rank_fusion,
            ) as fusion,
        ):
            results = retrieve_chunks(
                message="identity regression evidence",
                chunks=[paper_a, paper_b],
                limit=2,
                paper_id="anchor-paper",
            )

        self.assertEqual(fusion.call_args.kwargs["paper_id"], "anchor-paper")
        self.assertEqual(len(results), 2)
        self.assertEqual(
            {evidence_key(chunk, paper_id="anchor-paper") for chunk in results},
            {
                ("paper-a", "chunk_id", "chunk_001"),
                ("paper-b", "chunk_id", "chunk_001"),
            },
        )

    def test_pinned_exclusion_only_removes_same_global_evidence(self):
        paper_a_figure = {
            "source_paper_id": "paper-a",
            "chunk_id": "chunk_001",
            "text": "Figure 1 architecture diagram for paper A.",
            "label": "Figure 1",
            "is_figure_chunk": True,
            "is_table_chunk": False,
        }
        paper_b_figure = {
            "source_paper_id": "paper-b",
            "chunk_id": "chunk_001",
            "text": "Figure 1 architecture diagram for paper B.",
            "label": "Figure 1",
            "is_figure_chunk": True,
            "is_table_chunk": False,
        }

        def keep_order(_query, candidates, top_k):
            return candidates[:top_k]

        with (
            patch(
                "backend.services.retrieval_service.DenseEmbeddingService.search_dense",
                return_value=[(dict(paper_a_figure), 0.9), (dict(paper_b_figure), 0.8)],
            ),
            patch(
                "backend.services.retrieval_service.RerankerService.rerank",
                side_effect=keep_order,
            ),
        ):
            results = retrieve_chunks(
                message="Show Figure 1 architecture diagram",
                chunks=[paper_a_figure, paper_b_figure],
                limit=2,
                paper_id="anchor-paper",
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source_paper_id"], "paper-a")
        self.assertEqual(results[1]["source_paper_id"], "paper-b")

    def test_multi_hop_dedup_keeps_same_local_id_from_different_papers(self):
        paper_a = {
            "source_paper_id": "paper-a",
            "chunk_id": "chunk_001",
            "text": "Alpha evidence explains the method.",
            "is_figure_chunk": False,
            "is_table_chunk": False,
        }
        paper_b = {
            "source_paper_id": "paper-b",
            "chunk_id": "chunk_001",
            "text": "Beta evidence reports the result.",
            "is_figure_chunk": False,
            "is_table_chunk": False,
        }
        analysis = QuestionAnalysis(
            original_query="Combine alpha method and beta result evidence",
            reasoning_level=ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
            subqueries=[
                SubQuery(subquery_id="SQ1", query_text="alpha evidence method"),
                SubQuery(subquery_id="SQ2", query_text="beta evidence result"),
            ],
        )

        with patch(
            "backend.services.multi_hop_service.retrieve_chunks",
            side_effect=[[paper_a], [paper_b]],
        ) as retrieve_mock:
            results, _ = MultiHopRetrievalService.execute_multi_hop_retrieval(
                query=analysis.original_query,
                chunks=[paper_a, paper_b],
                limit=4,
                paper_id="anchor-paper",
                analysis=analysis,
            )

        self.assertEqual(len(results), 2)
        self.assertEqual({chunk["source_paper_id"] for chunk in results}, {"paper-a", "paper-b"})
        self.assertTrue(all(chunk["chunk_id"] == "chunk_001" for chunk in results))
        self.assertTrue(
            all(call.kwargs["paper_id"] == "anchor-paper" for call in retrieve_mock.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
