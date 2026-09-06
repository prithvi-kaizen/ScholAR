"""Unit tests for Phase 1: Benchmark Metric Remediations."""

import unittest
from evaluation.benchmarks.multihop_rag import MultiHopRAGAdapter, normalize_answer, compute_exact_match, compute_token_f1
from evaluation.benchmarks.spiqa import _compute_token_f1 as spiqa_token_f1
from evaluation.benchmarks.scivqa import SciVQAAdapter


class TestBenchmarkMetricsRemediation(unittest.TestCase):

    def test_multihop_rag_empty_prediction_em(self):
        """Empty prediction on nonempty gold must return EM=0.0."""
        adapter = MultiHopRAGAdapter()
        predictions = [
            {
                "prediction": "",
                "gold_answers": ["the correct answer"],
                "answerable": True,
            }
        ]
        metrics = adapter.compute_metrics(predictions)
        self.assertEqual(metrics["multihop_exact_match"], 0.0)

    def test_multihop_rag_exact_match_not_substring(self):
        """Substring containment must not count as Exact Match."""
        adapter = MultiHopRAGAdapter()
        # "a" is a substring of "a long gold answer", but not exact match
        predictions = [
            {
                "prediction": "a",
                "gold_answers": ["a long gold answer"],
                "answerable": True,
            }
        ]
        metrics = adapter.compute_metrics(predictions)
        self.assertEqual(metrics["multihop_exact_match"], 0.0)

        # Exact match (case and punctuation insensitive)
        predictions_exact = [
            {
                "prediction": "A Long Gold Answer!",
                "gold_answers": ["a long gold answer"],
                "answerable": True,
            }
        ]
        metrics_exact = adapter.compute_metrics(predictions_exact)
        self.assertEqual(metrics_exact["multihop_exact_match"], 1.0)

    def test_multihop_rag_missing_gold_evidence_epr(self):
        """Missing gold evidence must be excluded from Evidence Path Recall denominator, not scored as 1.0."""
        adapter = MultiHopRAGAdapter()
        predictions = [
            {
                "prediction": "some answer",
                "gold_answers": ["some answer"],
                "gold_evidence": [],  # Missing supervision
                "retrieved_sources": ["p1", "p2"],
                "answerable": True,
            }
        ]
        metrics = adapter.compute_metrics(predictions)
        # Because no gold evidence was provided, there are 0 supervised cases
        self.assertEqual(metrics["evidence_path_recall"], 0.0)
        self.assertTrue(predictions[0].get("missing_supervision"))

    def test_spiqa_multiset_token_f1(self):
        """Repeated identical tokens 'a a b' vs 'a a b' must yield F1 = 1.0."""
        f1 = spiqa_token_f1("a a b", "a a b")
        self.assertEqual(f1, 1.0)

        # Partial repetition
        f1_partial = spiqa_token_f1("a b", "a a b")
        # common = {'a': 1, 'b': 1}, len(pred)=2, len(gold)=3 -> p=1.0, r=2/3 -> f1 = 4/5 = 0.8
        self.assertAlmostEqual(f1_partial, 0.8, places=2)

        # Empty vs nonempty
        self.assertEqual(spiqa_token_f1("", "a b"), 0.0)
        self.assertEqual(spiqa_token_f1("a b", ""), 0.0)

    def test_scivqa_remediation_no_arbitrary_word_proxy(self):
        """Unrelated prediction sharing only an arbitrary single word must NOT score 1.0."""
        adapter = SciVQAAdapter()
        # Gold answer is "Multi-Head Attention", prediction has "Attention" in unrelated context
        predictions_unrelated = [
            {
                "gold_answer": "Multi-Head Attention",
                "prediction": "The authors pay great attention to detail.",
                "figure_found": True,
            }
        ]
        metrics = adapter.compute_metrics(predictions_unrelated)
        self.assertEqual(metrics["visual_qa_accuracy"], 0.0)
        self.assertEqual(metrics["exact_match"], 0.0)

        # Full phrase containment succeeds
        predictions_containment = [
            {
                "gold_answer": "Multi-Head Attention",
                "prediction": "The encoder sub-layers are Multi-Head Attention and feed-forward network.",
                "figure_found": True,
            }
        ]
        metrics_cont = adapter.compute_metrics(predictions_containment)
        self.assertEqual(metrics_cont["visual_qa_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
