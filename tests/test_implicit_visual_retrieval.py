"""Unit tests for Implicit Visual & Tabular Context Retrieval."""

import unittest
from backend.schemas.capabilities import ModelCapabilities
from backend.services.chunking_service import chunk_figures
from backend.services.question_analyzer import QuestionAnalyzer
from backend.services.retrieval_service import (
    _is_implicit_visual_or_tabular_query,
    extract_figure_refs,
    retrieve_chunks,
    tokenize,
)
from backend.services.routing_service import QuestionRouter, QuestionRouteType


class TestImplicitVisualRetrieval(unittest.TestCase):

    def test_chunk_figures_body_text_and_table_flag(self):
        """Verify that chunk_figures incorporates body_text and sets is_table_chunk."""
        figures = [
            {
                "figure_id": "01_001",
                "figure_type": "table",
                "label": "Table 2",
                "caption": "Comparison of translation performance.",
                "body_text": "Transformer (base) 27.3 BLEU 38.1 BLEU. Transformer (big) 28.4 BLEU 41.0 BLEU 2.3e19 FLOPs.",
                "page": 8,
                "image_file": "fig_01_001.png",
            },
            {
                "figure_id": "01_002",
                "figure_type": "figure",
                "label": "Figure 1",
                "caption": "The Transformer model architecture.",
                "body_text": "Scaled Dot-Product Attention Multi-Head Attention Feed Forward Positional Encoding",
                "page": 3,
                "image_file": "fig_01_002.png",
            },
        ]

        chunks = chunk_figures(figures, source_paper_id="paper_123")
        self.assertEqual(len(chunks), 2)

        # Table 2 chunk
        t_chunk = chunks[0]
        self.assertTrue(t_chunk["is_table_chunk"])
        self.assertTrue(t_chunk["is_figure_chunk"])
        self.assertIn("28.4 BLEU", t_chunk["text"])
        self.assertIn("FLOPs", t_chunk["retrieval_text"])
        self.assertEqual(t_chunk["source_paper_id"], "paper_123")

        # Figure 1 chunk
        f_chunk = chunks[1]
        self.assertFalse(f_chunk["is_table_chunk"])
        self.assertTrue(f_chunk["is_figure_chunk"])
        self.assertIn("Multi-Head Attention", f_chunk["text"])

    def test_implicit_intent_detection(self):
        """Verify detection of implicit visual and tabular query intent without keywords."""
        # Metric query (implicit tabular)
        terms = tokenize("What is the BLEU score on WMT 2014 English-to-German?")
        vis, tab = _is_implicit_visual_or_tabular_query(
            "What is the BLEU score on WMT 2014 English-to-German?", terms
        )
        self.assertTrue(tab)

        # Hyperparameter query (implicit tabular)
        terms2 = tokenize("What learning rate warmup steps and dropout rate were used?")
        vis2, tab2 = _is_implicit_visual_or_tabular_query(
            "What learning rate warmup steps and dropout rate were used?", terms2
        )
        self.assertTrue(tab2)

        # Visual architecture query (implicit visual)
        terms3 = tokenize("Explain the attention distribution and workflow")
        vis3, tab3 = _is_implicit_visual_or_tabular_query(
            "Explain the attention distribution and workflow", terms3
        )
        self.assertTrue(vis3)

    def test_implicit_metric_retrieval(self):
        """Verify that an implicit query retrieves a table chunk containing the metric in its body."""
        chunks = [
            {
                "chunk_id": "chunk_001",
                "page": 1,
                "text": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks.",
                "retrieval_text": "Introduction > The dominant sequence transduction models...",
                "is_figure_chunk": False,
                "is_table_chunk": False,
                "chunk_type": "introduction",
            },
            {
                "chunk_id": "fig_08_001",
                "page": 8,
                "text": "Table 2: Comparison on translation benchmarks.\nModel EN-DE BLEU EN-FR BLEU Training FLOPs\nTransformer (base) 27.3 38.1 3.3e18\nTransformer (big) 28.4 41.0 2.3e19",
                "retrieval_text": "Table > Table 2: Comparison on translation benchmarks.\nModel EN-DE BLEU EN-FR BLEU Training FLOPs\nTransformer (base) 27.3 38.1 3.3e18\nTransformer (big) 28.4 41.0 2.3e19",
                "is_figure_chunk": True,
                "is_table_chunk": True,
                "chunk_type": "table",
                "label": "Table 2",
            },
            {
                "chunk_id": "chunk_002",
                "page": 5,
                "text": "We use residual connections around each of the two sub-layers, followed by layer normalization.",
                "retrieval_text": "Architecture > We use residual connections...",
                "is_figure_chunk": False,
                "is_table_chunk": False,
                "chunk_type": "method",
            },
        ]

        # Query has NO explicit words "table" or "figure"
        query = "What is the English-to-German BLEU score for Transformer (big)?"
        retrieved = retrieve_chunks(query, chunks, limit=2)

        # The table chunk should be retrieved as top result
        self.assertTrue(any(c.get("chunk_id") == "fig_08_001" for c in retrieved))
        self.assertEqual(retrieved[0]["chunk_id"], "fig_08_001")

    def test_cross_modal_reference_bridging(self):
        """Verify that citations in top text chunks bridge and promote the referenced visual chunk."""
        chunks = [
            {
                "chunk_id": "chunk_010",
                "page": 7,
                "text": "On the WMT 2014 English-to-German translation task, the big transformer model outperforms the best previously reported models as presented in Table 2.",
                "retrieval_text": "Results > On the WMT 2014 English-to-German translation task...",
                "is_figure_chunk": False,
                "is_table_chunk": False,
                "chunk_type": "result",
            },
            {
                "chunk_id": "fig_08_001",
                "page": 8,
                "text": "Table 2: English-to-German and English-to-French translation results.",
                "retrieval_text": "Table > Table 2: Results",
                "is_figure_chunk": True,
                "is_table_chunk": True,
                "chunk_type": "table",
                "label": "Table 2",
            },
            {
                "chunk_id": "chunk_011",
                "page": 9,
                "text": "We described the Transformer, the first sequence transduction model based entirely on attention.",
                "retrieval_text": "Conclusion > We described the Transformer...",
                "is_figure_chunk": False,
                "is_table_chunk": False,
                "chunk_type": "conclusion",
            },
        ]

        query = "How does the big transformer model compare to previous translation baselines?"
        retrieved = retrieve_chunks(query, chunks, limit=2)

        # Both the text chunk and the bridged Table 2 chunk should be in the retrieved pool
        chunk_ids = [c["chunk_id"] for c in retrieved]
        self.assertIn("chunk_010", chunk_ids)
        self.assertIn("fig_08_001", chunk_ids)

    def test_question_router_implicit_classification(self):
        """Verify that QuestionRouter correctly allocates visual budget for implicit queries."""
        vlm = ModelCapabilities(model_id="gemma4:12b", display_name="Gemma 4 12B", supports_vision=True)

        # Query with BLEU metric
        budget = QuestionRouter.route(
            "What was the BLEU score achieved on English-to-German?", capabilities=vlm
        )
        self.assertEqual(budget.route_type, QuestionRouteType.TABLE_NUMERIC)
        self.assertGreaterEqual(budget.visual_items, 1)

        # Query with FLOPs & latency
        budget2 = QuestionRouter.route(
            "Compare the training FLOPs and parameter count of the base and big models",
            capabilities=vlm,
        )
        self.assertEqual(budget2.route_type, QuestionRouteType.TABLE_NUMERIC)
        self.assertGreaterEqual(budget2.visual_items, 1)

    def test_question_analyzer_implicit_modalities(self):
        """Verify that QuestionAnalyzer activates table/visual modalities for implicit queries."""
        from backend.schemas.reasoning import TargetModality
        analysis = QuestionAnalyzer.analyze_query("What is the BLEU score on WMT 2014 English-to-German?")
        self.assertTrue(
            any(
                sq.target_modality in (TargetModality.TABLE, TargetModality.MULTIMODAL, TargetModality.FIGURE)
                for sq in analysis.subqueries
            )
        )


if __name__ == "__main__":
    unittest.main()
