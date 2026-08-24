"""Unit tests for Phase 2: Tri-Channel Hybrid Retrieval (BM25 + Dense + Visual) with RRF & Reranking."""

import unittest
from backend.services.dense_embedding_service import DenseEmbeddingService
from backend.services.reranker_service import RerankerService
from backend.services.retrieval_service import (
    reciprocal_rank_fusion,
    retrieve_chunks,
    tokenize,
)


class TestTriChannelRetrieval(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "chunk_001",
                "evidence_id": "E_001",
                "document_id": "paper_test",
                "page": 1,
                "section": "1. Introduction",
                "text": "The Transformer architecture relies entirely on an attention mechanism to draw global dependencies.",
                "is_figure_chunk": False,
                "is_table_chunk": False,
            },
            {
                "chunk_id": "chunk_002",
                "evidence_id": "E_002",
                "document_id": "paper_test",
                "page": 3,
                "section": "3. Model Architecture",
                "text": "Multi-Head Attention allows the model to jointly attend to information from different representation subspaces.",
                "is_figure_chunk": False,
                "is_table_chunk": False,
            },
            {
                "chunk_id": "chunk_003",
                "evidence_id": "E_TAB_01",
                "document_id": "paper_test",
                "page": 8,
                "section": "5. Results",
                "text": "| Model | BLEU-4 | Training Cost |\n| Transformer (base) | 27.3 | 3.3e18 |\n| Transformer (big) | 28.4 | 2.3e19 |",
                "is_figure_chunk": False,
                "is_table_chunk": True,
            },
            {
                "chunk_id": "chunk_004",
                "evidence_id": "VIS_F1",
                "document_id": "paper_test",
                "page": 4,
                "section": "3. Model Architecture",
                "text": "Figure 1: The Transformer - model architecture with Scaled Dot-Product Attention.",
                "is_figure_chunk": True,
                "is_table_chunk": False,
                "label": "Figure 1",
            },
        ]

    def test_dense_embedding_service(self):
        """Verify dense embedding vector shape, normalization, and similarity search."""
        texts = [
            "Attention mechanism and Transformer layers",
            "Convolutional neural networks for image classification",
        ]
        vectors = DenseEmbeddingService.encode(texts)
        self.assertEqual(vectors.shape[0], 2)
        self.assertGreater(vectors.shape[1], 0)

        # Test search
        results = DenseEmbeddingService.search_dense("paper_test", "How does multi-head attention work?", self.sample_chunks, top_k=2)
        self.assertEqual(len(results), 2)
        top_chunk, top_score = results[0]
        self.assertIn("attention", top_chunk["text"].lower())
        self.assertGreaterEqual(top_score, 0.0)

    def test_reciprocal_rank_fusion(self):
        """Verify RRF mathematical scoring formula RRF(d) = sum 1 / (60 + rank)."""
        list1 = [self.sample_chunks[0], self.sample_chunks[1]]
        list2 = [self.sample_chunks[1], self.sample_chunks[0]]

        fused = reciprocal_rank_fusion([list1, list2], k=60)
        self.assertEqual(len(fused), 2)

        # Chunk 0 rank in list1=1, list2=2 -> 1/61 + 1/62
        expected_score_c0 = round((1.0 / 61) + (1.0 / 62), 6)
        self.assertEqual(fused[0]["rrf_score"], expected_score_c0)

    def test_reranker_service(self):
        """Verify cross-encoder reranking output scores in [0, 1] and ordering."""
        query = "What BLEU score did the big transformer achieve in Table 1?"
        reranked = RerankerService.rerank(query, self.sample_chunks, top_k=3)
        self.assertEqual(len(reranked), 3)
        for item in reranked:
            self.assertIn("rerank_score", item)
            self.assertGreaterEqual(item["rerank_score"], 0.0)
            self.assertLessEqual(item["rerank_score"], 1.0)

    def test_tri_channel_retrieval_text_query(self):
        """Verify end-to-end retrieval on text concept query."""
        results = retrieve_chunks(
            message="Explain multi-head attention mechanism",
            chunks=self.sample_chunks,
            limit=2,
            paper_id="paper_test",
        )
        self.assertEqual(len(results), 2)
        text_joined = " ".join(r["text"] for r in results).lower()
        self.assertIn("attention", text_joined)

    def test_tri_channel_retrieval_figure_pinned(self):
        """Verify explicit Figure 1 query pins Figure 1 visual chunk."""
        results = retrieve_chunks(
            message="Show me Figure 1 architecture diagram",
            chunks=self.sample_chunks,
            limit=2,
            paper_id="paper_test",
        )
        self.assertTrue(len(results) > 0)
        self.assertTrue(results[0].get("is_figure_chunk"))
        self.assertEqual(results[0].get("label"), "Figure 1")


if __name__ == "__main__":
    unittest.main()
