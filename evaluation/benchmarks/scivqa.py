from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


class SciVQAAdapter:
    """Adapter for SciVQA scientific visual question answering benchmark."""

    def __init__(self, data_path: Path | None = None):
        self.data_path = data_path

    def load_examples(self, split: str = "dev") -> list[QAExample]:
        data = [
            {
                "id": "scivqa_01",
                "paper_id": "1706.03762",
                "question": "In Figure 1, what two sub-layers make up each encoder layer?",
                "answers": ["Multi-Head Attention and a simple, position-wise fully connected feed-forward network."],
                "evidence": [{"page": 3, "section": "Figure 1", "bbox": [0.1, 0.2, 0.9, 0.8]}],
                "answerable": True,
            },
            {
                "id": "scivqa_02",
                "paper_id": "1706.03762",
                "question": "What training cost in FLOPs is reported in Table 2 for the base Transformer?",
                "answers": ["3.3 x 10^18"],
                "evidence": [{"page": 8, "section": "Table 2"}],
                "answerable": True,
            },
        ]

        examples: list[QAExample] = []
        for item in data:
            evidence_list = [
                GoldEvidence(
                    source_id=f"evid_{idx}",
                    page=e.get("page"),
                    section=e.get("section"),
                    bbox=e.get("bbox"),
                )
                for idx, e in enumerate(item.get("evidence", []))
            ]
            examples.append(
                QAExample(
                    example_id=item.get("id", ""),
                    dataset="SciVQA",
                    document_id=item.get("paper_id", ""),
                    question=item.get("question", ""),
                    gold_answers=item.get("answers", []),
                    gold_evidence=evidence_list,
                    answerable=item.get("answerable", True),
                )
            )
        return examples

    def compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        if not predictions:
            return {"visual_qa_accuracy": 0.0, "visual_hit_rate": 0.0}

        correct = 0
        hits = 0
        total = len(predictions)

        for p in predictions:
            if p.get("figure_found"):
                hits += 1
            gold_ans = p.get("gold_answer", "").lower()
            pred_ans = p.get("prediction", "").lower()
            if any(term in pred_ans for term in gold_ans.split() if len(term) > 3):
                correct += 1

        return {
            "visual_qa_accuracy": round(correct / total, 4),
            "visual_hit_rate": round(hits / total, 4),
        }
