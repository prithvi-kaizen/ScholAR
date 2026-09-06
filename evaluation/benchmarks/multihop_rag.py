from __future__ import annotations

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s or "")))))


def compute_exact_match(prediction: str, gold: str) -> float:
    """Compute normalized SQuAD-standard Exact Match."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(gold)
    if not norm_pred or not norm_gold:
        return 1.0 if norm_pred == norm_gold else 0.0
    return 1.0 if norm_pred == norm_gold else 0.0


def compute_token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 with multiset Counter overlap."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return 1.0 if pred_tokens == gold_tokens else 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


class MultiHopRAGAdapter:
    """Adapter for the MultiHop-RAG benchmark for inter-document scientific reasoning."""

    def __init__(self, data_path: Path | None = None):
        self.data_path = data_path or (Path(__file__).resolve().parents[1] / "multihop_rag_cases.json")

    def load_examples(self, split: str = "test") -> list[QAExample]:
        """Load MultiHop-RAG QA instances."""
        if self.data_path and self.data_path.exists():
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
        else:
            data = []

        examples: list[QAExample] = []
        for item in data:
            evidence_list = [
                GoldEvidence(
                    source_id=f"{e.get('paper_id', 'doc')}_p{e.get('page', 1)}",
                    page=e.get("page"),
                    section=e.get("section"),
                    text_span=e.get("text"),
                )
                for idx, e in enumerate(item.get("evidence", []))
            ]
            examples.append(
                QAExample(
                    example_id=item.get("id", ""),
                    dataset="MultiHop-RAG",
                    document_id=item.get("paper_id", ""),
                    question=item.get("question", ""),
                    gold_answers=item.get("answers", []),
                    gold_evidence=evidence_list,
                    answerable=item.get("answerable", True),
                    metadata={
                        "secondary_paper_ids": item.get("secondary_paper_ids", []),
                        "reasoning_level": item.get("reasoning_level", "L5_MULTI_HOP_SYNTHESIS"),
                    },
                )
            )
        return examples

    def compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        """Compute Exact Match, Token F1, and Evidence Path Recall (EPR) for MultiHop-RAG."""
        if not predictions:
            return {
                "multihop_exact_match": 0.0,
                "multihop_f1": 0.0,
                "evidence_path_recall": 0.0,
                "abstention_accuracy": 0.0,
            }

        em_total = 0.0
        f1_total = 0.0
        epr_total = 0.0
        epr_supervised_count = 0
        abstention_correct = 0
        abstention_total = 0

        for pred in predictions:
            gold_answers = pred.get("gold_answers", [])
            pred_answer = str(pred.get("prediction", "")).strip()
            answerable = pred.get("answerable", True)

            if not answerable:
                abstention_total += 1
                if pred.get("abstained", False) or "insufficient" in pred_answer.lower() or not pred_answer:
                    abstention_correct += 1
                continue

            if not gold_answers:
                continue

            # Exact Match (normalized SQuAD standard)
            em = max(compute_exact_match(pred_answer, g) for g in gold_answers)
            em_total += em

            # Token F1 (multiset Counter overlap)
            f1 = max(compute_token_f1(pred_answer, g) for g in gold_answers)
            f1_total += f1

            # Evidence Path Recall (EPR): checks if evidence from all required papers was retrieved
            gold_sources = {
                e.get("source_id", "") for e in pred.get("gold_evidence", []) if e.get("source_id")
            }
            retrieved_sources = {
                s for s in pred.get("retrieved_sources", []) if s
            }
            if gold_sources:
                epr = len(gold_sources.intersection(retrieved_sources)) / len(gold_sources)
                epr_total += epr
                epr_supervised_count += 1
            else:
                pred["missing_supervision"] = True

        n_ans = max(1, len(predictions) - abstention_total)
        return {
            "multihop_exact_match": round(em_total / n_ans, 4),
            "multihop_f1": round(f1_total / n_ans, 4),
            "evidence_path_recall": round(epr_total / max(1, epr_supervised_count), 4) if epr_supervised_count > 0 else 0.0,
            "abstention_accuracy": round(abstention_correct / max(1, abstention_total), 4) if abstention_total > 0 else 1.0,
        }

