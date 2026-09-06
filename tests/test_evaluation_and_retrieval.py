import unittest
from unittest.mock import patch

from evaluation.benchmarks.qasper import QASPERAdapter
from evaluation.benchmarks.peerqa import PeerQAAdapter
from evaluation.benchmarks.scivqa import SciVQAAdapter
from evaluation.interventions.perturbation import EvidencePerturbationRunner
from evaluation.run_retrieval_eval import RETRIEVERS, evaluate_case, summarize


class TestEvaluationAndRetrieval(unittest.TestCase):

    def test_four_channel_rows_record_case_specific_visual_participation(self):
        retriever_name = "four_channel_image_rrf_v1_with_page_hints"
        hit = {
            "chunk_id": "fig_1",
            "page": 2,
            "bm25_rank": 3,
            "dense_rank": 2,
            "image_embedding_rank": 1,
            "image_embedding_eligible": True,
        }
        case = {
            "id": "visual-case",
            "paper_id": "paper",
            "query": "What relation is shown?",
            "relevant_chunk_ids": ["fig_1"],
        }
        with (
            patch("evaluation.run_retrieval_eval.load_chunks", return_value=[hit]),
            patch.dict(RETRIEVERS, {retriever_name: lambda *_args, **_kwargs: [hit]}),
            patch(
                "evaluation.run_retrieval_eval.VisualEmbeddingService.status",
                return_value={
                    "model_loaded": True,
                    "active": True,
                    "encoder_fingerprint": "a" * 64,
                    "last_request_attempted": True,
                    "last_request_succeeded": True,
                    "last_request_hit_count": 1,
                },
            ),
        ):
            row = evaluate_case(case, retriever_name)

        self.assertIn("image_embedding", row["active_channels"])
        self.assertEqual(row["eligible_image_hits_in_top_k"], 1)
        self.assertTrue(row["visual_embedding"]["condition_enabled"])
        self.assertTrue(row["condition_eligible"])
        self.assertTrue(row["visual_embedding"]["last_request_succeeded"])
        self.assertEqual(row["visual_embedding"]["encoder_fingerprint"], "a" * 64)

    def test_four_channel_summary_excludes_image_ineligible_rows(self):
        name = "four_channel_image_rrf_v1_with_page_hints"
        base = {
            "retriever": name,
            "recall_at_1": 1,
            "recall_at_3": 1,
            "recall_at_5": 1,
            "reciprocal_rank": 1.0,
            "ndcg_at_5": 1.0,
        }
        metrics = summarize([
            {**base, "condition_eligible": True},
            {
                **base,
                "condition_eligible": False,
                "recall_at_1": 0,
                "recall_at_3": 0,
                "recall_at_5": 0,
                "reciprocal_rank": 0.0,
                "ndcg_at_5": 0.0,
            },
        ])[name]
        self.assertEqual(metrics["cases"], 1)
        self.assertEqual(metrics["total_cases"], 2)
        self.assertEqual(metrics["excluded_no_image"], 1)
        self.assertEqual(metrics["recall_at_1"], 1.0)

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
