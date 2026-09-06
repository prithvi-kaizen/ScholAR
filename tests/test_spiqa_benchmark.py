"""Unit tests for SPIQA multimodal scientific benchmark integration."""

import unittest
from pathlib import Path

from evaluation.benchmarks.spiqa import (
    SPIQAAdapter,
    SPIQAVisualType,
    _compute_token_f1,
    _normalize_text,
)
from evaluation.spiqa.build_spiqa import validate_cases


class TestSPIQABenchmark(unittest.TestCase):

    def setUp(self):
        self.sample_path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "spiqa"
            / "spiqa_cases_sample.json"
        )
        self.adapter = SPIQAAdapter(data_path=self.sample_path)

    def test_sample_cases_validation(self):
        """Verify the offline sample fixture satisfies schema constraints."""
        self.assertTrue(self.sample_path.exists())
        is_valid = validate_cases(self.sample_path)
        self.assertTrue(is_valid)

    def test_adapter_load_examples(self):
        """Verify adapter loads QAExamples with proper visual metadata."""
        examples = self.adapter.load_examples(split="all")
        self.assertGreaterEqual(len(examples), 4)

        # Test case properties
        ex = examples[0]
        self.assertEqual(ex.dataset, "SPIQA")
        self.assertTrue(ex.question)
        self.assertTrue(len(ex.gold_answers) >= 1)
        self.assertTrue(len(ex.gold_evidence) >= 1)

        # Verify visual element categories
        valid_types = {t.value for t in SPIQAVisualType}
        for e in examples:
            self.assertIn(e.metadata["visual_type"], valid_types)

    def test_token_f1_computation(self):
        """Verify token F1 overlap computation."""
        # Exact match
        f1_exact = _compute_token_f1("Multi-Head Attention", "multi-head attention")
        self.assertAlmostEqual(f1_exact, 1.0)

        # Partial overlap
        f1_partial = _compute_token_f1("Feed Forward Network", "Feed Forward")
        self.assertGreater(f1_partial, 0.5)
        self.assertLess(f1_partial, 1.0)

        # Disjoint
        f1_disjoint = _compute_token_f1("Convolutional Layer", "Recurrent Unit")
        self.assertEqual(f1_disjoint, 0.0)

    def test_adapter_compute_metrics(self):
        """Verify metric aggregation (MRR, hit rates, F1)."""
        mock_predictions = [
            {
                "visual_rank": 1,
                "prediction": "Multi-Head Attention and feed-forward network",
                "gold_answers": ["Multi-Head Attention and feed-forward network"],
            },
            {
                "visual_rank": 2,
                "prediction": "28.4",
                "gold_answers": ["28.4 BLEU"],
            },
            {
                "visual_rank": None,
                "prediction": "Unknown answer",
                "gold_answers": ["Specific model"],
            },
        ]

        metrics = self.adapter.compute_metrics(mock_predictions)
        self.assertEqual(metrics["n_cases"], 3.0)
        self.assertAlmostEqual(metrics["visual_hit_rate_at_1"], round(1 / 3, 4))
        self.assertAlmostEqual(metrics["visual_hit_rate_at_3"], round(2 / 3, 4))
        self.assertAlmostEqual(metrics["visual_hit_rate_at_5"], round(2 / 3, 4))
        # MRR = (1/1 + 1/2 + 0) / 3 = 1.5 / 3 = 0.5
        self.assertAlmostEqual(metrics["visual_mrr"], 0.5)
        self.assertGreater(metrics["mean_token_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
