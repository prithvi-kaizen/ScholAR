from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


class PeerQAAdapter:
    """Adapter for PeerQA reviewer-style QA and unanswerability benchmark."""

    def __init__(self, data_path: Path | None = None):
        self.data_path = data_path

    def load_examples(self, split: str = "dev") -> list[QAExample]:
        data = [
            {
                "id": "peerqa_01",
                "paper_id": "1706.03762",
                "question": "How do the authors justify removing recurrence entirely from the architecture?",
                "answers": ["Recurrence precludes parallelization within training examples, which becomes critical at longer sequence lengths."],
                "evidence": [{"page": 2, "section": "Background"}],
                "answerable": True,
            },
            {
                "id": "peerqa_02",
                "paper_id": "1706.03762",
                "question": "Does the paper evaluate on image classification benchmarks like ImageNet?",
                "answers": [],
                "evidence": [],
                "answerable": False,
            },
        ]

        examples: list[QAExample] = []
        for item in data:
            evidence_list = [
                GoldEvidence(
                    source_id=f"evid_{idx}",
                    page=e.get("page"),
                    section=e.get("section"),
                )
                for idx, e in enumerate(item.get("evidence", []))
            ]
            examples.append(
                QAExample(
                    example_id=item.get("id", ""),
                    dataset="PeerQA",
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
            return {"abstention_precision": 0.0, "abstention_recall": 0.0, "abstention_f1": 0.0}

        tp = fp = fn = tn = 0
        for p in predictions:
            gold_unanswerable = not p.get("gold_answerable", True)
            pred_abstained = bool(p.get("abstained", False))
            if pred_abstained and gold_unanswerable:
                tp += 1
            elif pred_abstained and not gold_unanswerable:
                fp += 1
            elif not pred_abstained and gold_unanswerable:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "abstention_precision": round(precision, 4),
            "abstention_recall": round(recall, 4),
            "abstention_f1": round(f1, 4),
        }
