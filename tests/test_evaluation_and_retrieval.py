import unittest
from backend.services.retrieval_multimodal import MultimodalHybridRetriever
from evaluation.benchmarks.qasper import QASPERAdapter
from evaluation.benchmarks.peerqa import PeerQAAdapter
from evaluation.benchmarks.scivqa import SciVQAAdapter
from evaluation.interventions.perturbation import EvidencePerturbationRunner


class TestEvaluationAndRetrieval(unittest.TestCase):

    def test_multimodal_hybrid_retriever(self):
        chunks = [
            {
                "chunk_id": "chunk_01",
                "text": "The Transformer relies entirely on self-attention mechanisms without recurrence.",
                "retrieval_text": "Introduction. The Transformer relies entirely on self-attention mechanisms without recurrence.",
                "page": 1,
                "section_title": "Introduction",
                "chunk_type": "paragraph",
                "is_figure_chunk": False,
            },
            {
                "chunk_id": "fig_01",
                "text": "Figure 1: The Transformer model architecture with encoder and decoder stacks.",
                "retrieval_text": "Figure > Figure 1: The Transformer model architecture with encoder and decoder stacks.",
                "page": 3,
                "section_title": "Figure",
                "chunk_type": "figure",
                "is_figure_chunk": True,
                "figure_id": "fig_03_001",
            },
        ]
        retriever = MultimodalHybridRetriever(rrf_k=60)
        results = retriever.retrieve_hybrid("What is the architecture shown in Figure 1?", chunks, text_limit=1, visual_limit=1)
        self.assertEqual(len(results), 2)
        # Verify RRF score and rank are calculated
        self.assertIn("rrf_score", results[0])
        self.assertIn("dense_score", results[0])
        self.assertIn("bm25_score", results[0])

    def test_qasper_adapter(self):
        adapter = QASPERAdapter()
        examples = adapter.load_examples("dev")
        self.assertGreater(len(examples), 0)
        self.assertEqual(examples[0].dataset, "QASPER")

        metrics = adapter.compute_metrics([
            {
                "gold_answer": "28.4 BLEU",
                "prediction": "The model achieved 28.4 BLEU score on English to German.",
                "gold_page_found": True,
            }
        ])
        self.assertGreater(metrics["answer_f1"], 0.0)
        self.assertEqual(metrics["evidence_recall"], 1.0)

    def test_peerqa_adapter(self):
        adapter = PeerQAAdapter()
        examples = adapter.load_examples("dev")
        self.assertGreater(len(examples), 0)
        self.assertEqual(examples[0].dataset, "PeerQA")

        metrics = adapter.compute_metrics([
            {"gold_answerable": False, "abstained": True},
            {"gold_answerable": True, "abstained": False},
        ])
        self.assertEqual(metrics["abstention_precision"], 1.0)
        self.assertEqual(metrics["abstention_recall"], 1.0)
        self.assertEqual(metrics["abstention_f1"], 1.0)

    def test_scivqa_adapter(self):
        adapter = SciVQAAdapter()
        examples = adapter.load_examples("dev")
        self.assertGreater(len(examples), 0)
        self.assertEqual(examples[0].dataset, "SciVQA")

        metrics = adapter.compute_metrics([
            {
                "gold_answer": "Multi-Head Attention",
                "prediction": "The encoder sub-layers are Multi-Head Attention and feed-forward network.",
                "figure_found": True,
            }
        ])
        self.assertEqual(metrics["visual_qa_accuracy"], 1.0)
        self.assertEqual(metrics["visual_hit_rate"], 1.0)

    def test_evidence_perturbation_runner(self):
        def mock_generate(q, context):
            return f"Based on the context, the learning rate was 7e-5."

        result = EvidencePerturbationRunner.test_text_sensitivity(
            original_chunk_text="We trained using AdamW with learning rate 2e-5 for 10 epochs.",
            question="What was the learning rate?",
            target_entity="learning_rate",
            original_value="2e-5",
            perturbed_value="7e-5",
            generate_fn=mock_generate,
        )
        self.assertTrue(result.followed_intervention)
        self.assertEqual(EvidencePerturbationRunner.compute_tesr([result]), 1.0)


if __name__ == "__main__":
    unittest.main()
