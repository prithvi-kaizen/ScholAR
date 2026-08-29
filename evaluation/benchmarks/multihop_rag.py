from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from evaluation.benchmarks.base import GoldEvidence, QAExample


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
        abstention_correct = 0
        abstention_total = 0

        for pred in predictions:
            gold_answers = pred.get("gold_answers", [])
            pred_answer = pred.get("prediction", "").strip().lower()
            answerable = pred.get("answerable", True)

            if not answerable:
                abstention_total += 1
                if pred.get("abstained", False) or "insufficient" in pred_answer or not pred_answer:
                    abstention_correct += 1
                continue

            if not gold_answers:
                continue

            # Exact Match
            em = max(float(g.lower() in pred_answer or pred_answer in g.lower()) for g in gold_answers)
            em_total += em

            # Token F1
            pred_toks = set(pred_answer.split())
            best_f1 = 0.0
            for g in gold_answers:
                gold_toks = set(g.lower().split())
                common = pred_toks.intersection(gold_toks)
                if not common:
                    continue
                p = len(common) / len(pred_toks)
                r = len(common) / len(gold_toks)
                f1 = 2 * p * r / (p + r)
                best_f1 = max(best_f1, f1)
            f1_total += best_f1

            # Evidence Path Recall (EPR): checks if evidence from all required papers was retrieved
            gold_sources = {e.get("source_id", "") for e in pred.get("gold_evidence", [])}
            retrieved_sources = set(pred.get("retrieved_sources", []))
            if gold_sources:
                epr = len(gold_sources.intersection(retrieved_sources)) / len(gold_sources)
            else:
                epr = 1.0
            epr_total += epr

        n_ans = max(1, len(predictions) - abstention_total)
        return {
            "multihop_exact_match": round(em_total / n_ans, 4),
            "multihop_f1": round(f1_total / n_ans, 4),
            "evidence_path_recall": round(epr_total / n_ans, 4),
            "abstention_accuracy": round(abstention_correct / max(1, abstention_total), 4) if abstention_total > 0 else 1.0,
        }
