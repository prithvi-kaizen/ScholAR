"""SPIQA: Scientific Paper Image Question Answering Benchmark Adapter (NeurIPS 2024).

Implements BenchmarkAdapter protocol for SPIQA (Pramanick et al., 2024).
Evaluates multimodal question answering across complex scientific figures, plots,
charts, schematic diagrams, and benchmark tables.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Any

from evaluation.benchmarks.base import GoldEvidence, QAExample


class SPIQAVisualType(str, Enum):
    """Visual category taxonomy established in the SPIQA benchmark."""
    PLOT = "plot"
    CHART = "chart"
    TABLE = "table"
    SCHEMATIC_DIAGRAM = "schematic_diagram"
    RESULT_VISUALIZATION = "result_visualization"
    MIXED = "mixed"


def _normalize_text(text: str) -> str:
    """Normalize text for token-level scoring."""
    lower = str(text or "").lower()
    cleaned = re.sub(r"[^\w\s]", " ", lower)
    return " ".join(cleaned.split())


def _compute_token_f1(prediction: str, gold: str) -> float:
    """Compute token-level F1 overlap between prediction and reference answer using multiset overlap."""
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


class SPIQAAdapter:
    """Adapter for SPIQA scientific paper multimodal QA benchmark."""

    def __init__(self, data_path: Path | None = None):
        self.data_path = (
            Path(data_path)
            if data_path
            else Path(__file__).resolve().parents[1] / "spiqa" / "spiqa_cases_sample.json"
        )

    def load_examples(self, split: str = "test") -> list[QAExample]:
        """Load benchmark cases filtered by split."""
        if not self.data_path.exists():
            return []

        raw_data = json.loads(self.data_path.read_text(encoding="utf-8"))
        items = raw_data.get("cases", []) if isinstance(raw_data, dict) else raw_data

        examples: list[QAExample] = []
        for item in items:
            item_split = item.get("split", "test")
            if split and item_split != split and split != "all":
                continue

            evidence_list: list[GoldEvidence] = []
            for idx, ev in enumerate(item.get("evidence", [])):
                evidence_list.append(
                    GoldEvidence(
                        source_id=str(ev.get("source_id") or f"evid_{idx+1}"),
                        page=ev.get("page"),
                        section=ev.get("section") or ev.get("figure_label"),
                        text_span=ev.get("caption") or ev.get("text_span"),
                        bbox=ev.get("bbox") or ev.get("bbox_norm"),
                    )
                )

            # Fallback gold evidence from top-level fields if evidence list is empty
            if not evidence_list and item.get("gold_pages"):
                for idx, p in enumerate(item["gold_pages"]):
                    evidence_list.append(
                        GoldEvidence(
                            source_id=f"page_{p}_{idx}",
                            page=p,
                            section=item.get("figure_label") or item.get("visual_type"),
                        )
                    )

            examples.append(
                QAExample(
                    example_id=str(item.get("case_id") or item.get("id", "")),
                    dataset="SPIQA",
                    document_id=str(item.get("paper_id", "")),
                    question=str(item.get("question", "")),
                    gold_answers=list(item.get("answers") or ([item["answer"]] if "answer" in item else [])),
                    gold_evidence=evidence_list,
                    answerable=item.get("answerable", True),
                    metadata={
                        "visual_type": item.get("visual_type", SPIQAVisualType.MIXED.value),
                        "figure_id": item.get("figure_id"),
                        "figure_label": item.get("figure_label"),
                        "split": item_split,
                        "paper_title": item.get("paper_title"),
                        "caption": item.get("caption"),
                    },
                )
            )

        return examples

    def compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        """Compute multimodal retrieval and answering metrics."""
        if not predictions:
            return {
                "n_cases": 0.0,
                "visual_hit_rate_at_1": 0.0,
                "visual_hit_rate_at_3": 0.0,
                "visual_hit_rate_at_5": 0.0,
                "visual_mrr": 0.0,
                "mean_token_f1": 0.0,
                "exact_match": 0.0,
            }

        total = len(predictions)
        hits_1 = 0
        hits_3 = 0
        hits_5 = 0
        rr_sum = 0.0
        f1_sum = 0.0
        em_sum = 0

        for p in predictions:
            # Retrieval rank assessment (if available)
            rank = p.get("visual_rank") or p.get("gold_page_rank")
            if rank is not None and isinstance(rank, int) and rank > 0:
                if rank <= 1:
                    hits_1 += 1
                if rank <= 3:
                    hits_3 += 1
                if rank <= 5:
                    hits_5 += 1
                rr_sum += 1.0 / rank
            elif p.get("figure_found"):
                hits_1 += 1
                hits_3 += 1
                hits_5 += 1
                rr_sum += 1.0

            # Answer quality assessment
            pred_ans = str(p.get("prediction", "")).strip()
            gold_answers = p.get("gold_answers", [])
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            best_f1 = 0.0
            is_em = False
            for gold in gold_answers:
                f1 = _compute_token_f1(pred_ans, gold)
                if f1 > best_f1:
                    best_f1 = f1
                if _normalize_text(pred_ans) == _normalize_text(gold):
                    is_em = True

            f1_sum += best_f1
            if is_em:
                em_sum += 1

        return {
            "n_cases": float(total),
            "visual_hit_rate_at_1": round(hits_1 / total, 4),
            "visual_hit_rate_at_3": round(hits_3 / total, 4),
            "visual_hit_rate_at_5": round(hits_5 / total, 4),
            "visual_mrr": round(rr_sum / total, 4),
            "mean_token_f1": round(f1_sum / total, 4),
            "exact_match": round(em_sum / total, 4),
        }
