"""Tests for the implicit-visual retrieval evaluation contract."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from evaluation.run_visual_retrieval_eval import (
    VisualRetrievalBenchmark,
    _iou,
    clustered_interval,
)


def _case(case_id: str, paper_id: str, split: str) -> dict:
    return {
        "case_id": case_id,
        "pair_id": "pair_1",
        "paper_id": paper_id,
        "query": "Which system peaks after the crossover?",
        "formulation": "implicit",
        "visual_type": "plot",
        "visual_necessity": "visual_only",
        "gold_pages": [2],
        "gold_regions": [{"page": 2, "bbox_norm": [0.1, 0.2, 0.8, 0.9]}],
        "answerable": True,
        "split": split,
    }


class TestVisualRetrievalEvaluation(unittest.TestCase):
    def test_disjoint_benchmark_rejects_paper_overlap(self) -> None:
        with self.assertRaises(ValidationError):
            VisualRetrievalBenchmark.model_validate({
                "name": "test",
                "version": "1",
                "paper_disjoint_from_development": True,
                "cases": [
                    _case("dev", "paper", "development"),
                    _case("test", "paper", "test"),
                ],
            })

    def test_clustered_interval_uses_papers_as_units(self) -> None:
        rows = [
            {"paper_id": "a", "recall_at_5": 1.0},
            {"paper_id": "a", "recall_at_5": 1.0},
            {"paper_id": "b", "recall_at_5": 0.0},
        ]
        result = clustered_interval(rows, "recall_at_5", samples=500, seed=7)
        self.assertEqual(result["papers"], 2)
        self.assertEqual(result["mean"], 0.5)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])

    def test_region_iou_is_exact_for_identical_boxes(self) -> None:
        box = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(_iou(box, box), 1.0)
        self.assertEqual(_iou(box, [0.81, 0.2, 0.9, 0.9]), 0.0)


if __name__ == "__main__":
    unittest.main()
