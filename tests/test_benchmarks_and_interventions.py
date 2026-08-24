import unittest
from evaluation.benchmarks.qasper import QASPERAdapter
from evaluation.benchmarks.peerqa import PeerQAAdapter
from evaluation.benchmarks.scivqa import SciVQAAdapter
from evaluation.interventions.perturbation import EvidencePerturbationRunner, InterventionResult
from evaluation.run_comprehensive_eval import run_model_matrix


class TestBenchmarksAndInterventions(unittest.TestCase):

    def test_qasper_adapter(self):
        adapter = QASPERAdapter()
        examples = adapter.load_examples()
        self.assertGreaterEqual(len(examples), 2)
        self.assertEqual(examples[0].dataset, "QASPER")

        # Test metrics
        preds = [
            {"gold_answer": "28.4 BLEU", "prediction": "28.4 BLEU score achieved", "gold_page_found": True},
            {"gold_answer": "", "prediction": "", "abstained": True, "gold_page_found": False},
        ]
        metrics = adapter.compute_metrics(preds)
        self.assertIn("answer_f1", metrics)
        self.assertIn("evidence_recall", metrics)
        self.assertGreater(metrics["answer_f1"], 0.5)

    def test_peerqa_adapter(self):
        adapter = PeerQAAdapter()
        examples = adapter.load_examples()
        self.assertGreaterEqual(len(examples), 2)
        self.assertEqual(examples[0].dataset, "PeerQA")

        preds = [
            {"gold_answerable": False, "abstained": True},
            {"gold_answerable": True, "abstained": False},
        ]
        metrics = adapter.compute_metrics(preds)
        self.assertEqual(metrics["abstention_f1"], 1.0)

    def test_scivqa_adapter(self):
        adapter = SciVQAAdapter()
        examples = adapter.load_examples()
        self.assertGreaterEqual(len(examples), 2)
        self.assertEqual(examples[0].dataset, "SciVQA")

        preds = [
            {"gold_answer": "Multi-Head Attention", "prediction": "It contains Multi-Head Attention layers", "figure_found": True},
        ]
        metrics = adapter.compute_metrics(preds)
        self.assertEqual(metrics["visual_qa_accuracy"], 1.0)
        self.assertEqual(metrics["visual_hit_rate"], 1.0)

    def test_evidence_perturbation_runner(self):
        # TESR
        def mock_gen(q, ctx):
            return f"The answer based on text is {ctx}"

        res_tesr = EvidencePerturbationRunner.test_text_sensitivity(
            original_chunk_text="Metric is 10.0.",
            question="What is the metric?",
            target_entity="metric",
            original_value="10.0",
            perturbed_value="99.9",
            generate_fn=mock_gen,
        )
        self.assertTrue(res_tesr.followed_intervention)
        tesr = EvidencePerturbationRunner.compute_tesr([res_tesr])
        self.assertEqual(tesr, 1.0)

        # VESR
        res_vesr = EvidencePerturbationRunner.test_visual_sensitivity(
            figure_label="Figure 2",
            original_caption="The loss dropped to 0.15.",
            question="What was the final loss?",
            target_entity="loss",
            original_value="0.15",
            perturbed_value="0.01",
            generate_fn=mock_gen,
        )
        self.assertTrue(res_vesr.followed_intervention)
        vesr = EvidencePerturbationRunner.compute_vesr([res_vesr])
        self.assertEqual(vesr, 1.0)

    def test_model_capability_matrix(self):
        matrix = run_model_matrix()
        self.assertIn("qwen3.5:9b", matrix)
        self.assertIn("gemma4:12b", matrix)
        self.assertIn("llama3.1:8b", matrix)
        # Vision models have visual budget in AUTO mode
        self.assertGreater(matrix["qwen3.5:9b"]["auto_visual_budget"], 0)
        # In TEXT_ONLY mode, visual budget is 0
        self.assertEqual(matrix["qwen3.5:9b"]["text_only_mode_visual_budget"], 0)


if __name__ == "__main__":
    unittest.main()
