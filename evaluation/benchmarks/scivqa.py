from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


def _normalize_text(text: str) -> str:
    """Normalize text for token-level scoring."""
    lower = str(text or "").lower()
    cleaned = re.sub(r"[^\w\s]", " ", lower)
    return " ".join(cleaned.split())


def _compute_token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 overlap between prediction and reference answer."""
    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2.0 * (precision * recall) / (precision + recall)


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
            return {"visual_qa_accuracy": 0.0, "exact_match": 0.0, "mean_token_f1": 0.0, "visual_hit_rate": 0.0}

        correct = 0
        hits = 0
        total = len(predictions)
        f1_sum = 0.0
        em_sum = 0

        for p in predictions:
            if p.get("figure_found"):
                hits += 1

            gold_raw = p.get("gold_answer") or p.get("gold_answers", [])
            if isinstance(gold_raw, str):
                golds = [gold_raw] if gold_raw else []
            else:
                golds = list(gold_raw)

            pred_ans = str(p.get("prediction", "")).strip()
            norm_pred = _normalize_text(pred_ans)

            is_correct = False
            best_f1 = 0.0
            is_em = False

            for g in golds:
                norm_gold = _normalize_text(g)
                if not norm_gold:
                    continue
                if norm_pred == norm_gold:
                    is_em = True
                # Containment of the complete normalized gold phrase (not arbitrary single words)
                if norm_gold in norm_pred:
                    is_correct = True
                f1 = _compute_token_f1(pred_ans, g)
                if f1 > best_f1:
                    best_f1 = f1

            if is_correct:
                correct += 1
            if is_em:
                em_sum += 1
            f1_sum += best_f1

        return {
            "visual_qa_accuracy": round(correct / total, 4),
            "exact_match": round(em_sum / total, 4),
            "mean_token_f1": round(f1_sum / total, 4),
            "visual_hit_rate": round(hits / total, 4),
        }
