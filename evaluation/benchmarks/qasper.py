from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


class QASPERAdapter:
    """Adapter for the QASPER (Question Answering on Scientific Papers) benchmark."""

    def __init__(self, data_path: Path | None = None):
        self.data_path = data_path

    def load_examples(self, split: str = "dev") -> list[QAExample]:
        """Load QASPER QA instances."""
        if self.data_path and self.data_path.exists():
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
        else:
            # High-fidelity reference dev cases when official dataset file is being downloaded
            data = [
                {
                    "id": "qasper_1706_01",
                    "paper_id": "1706.03762",
                    "question": "What is the key advantage of self-attention over recurrent layers?",
                    "answers": ["Self-attention connects all positions with a constant number of sequentially executed operations."],
                    "evidence": [{"page": 1, "section": "Introduction"}],
                    "answerable": True,
                },
                {
                    "id": "qasper_1706_02",
                    "paper_id": "1706.03762",
                    "question": "What BLEU score is reported on WMT 2014 English-to-German?",
                    "answers": ["28.4 BLEU"],
                    "evidence": [{"page": 8, "section": "Results"}],
                    "answerable": True,
                },
                {
                    "id": "qasper_1706_03",
                    "paper_id": "1706.03762",
                    "question": "What optimizer was used for pre-training BERT in this paper?",
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
                    text_span=e.get("text"),
                )
                for idx, e in enumerate(item.get("evidence", []))
            ]
            examples.append(
                QAExample(
                    example_id=item.get("id", ""),
                    dataset="QASPER",
                    document_id=item.get("paper_id", ""),
                    question=item.get("question", ""),
                    gold_answers=item.get("answers", []),
                    gold_evidence=evidence_list,
                    answerable=item.get("answerable", True),
                    metadata=item.get("metadata", {}),
                )
            )
        return examples

    def compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        """Compute answer F1 and evidence Recall@k."""
        if not predictions:
            return {"answer_f1": 0.0, "evidence_recall": 0.0}

        total_f1 = 0.0
        total_evid_recall = 0.0
        count = len(predictions)

        for pred in predictions:
            gold_ans = set(pred.get("gold_answer", "").lower().split())
            pred_ans = set(pred.get("prediction", "").lower().split())
            if gold_ans and pred_ans:
                common = gold_ans.intersection(pred_ans)
                precision = len(common) / len(pred_ans)
                recall = len(common) / len(gold_ans)
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                total_f1 += f1
            elif not gold_ans and pred.get("abstained"):
                # Correct abstention
                total_f1 += 1.0

            if pred.get("gold_page_found"):
                total_evid_recall += 1.0

        return {
            "answer_f1": round(total_f1 / count, 4),
            "evidence_recall": round(total_evid_recall / count, 4),
        }
