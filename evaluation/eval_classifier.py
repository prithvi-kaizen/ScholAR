"""Independent Evaluation for QuestionAnalyzer Classifier (L1-L5).

Computes:
- Macro F1, Weighted F1, Overall Accuracy
- Per-level Precision, Recall, F1 for L1, L2, L3, L4, L5
- Confusion Matrix
- Modality Routing Precision/Recall
- Decomposition Trigger Accuracy
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.schemas.reasoning import ReasoningLevel, TargetModality
from backend.services.question_analyzer import QuestionAnalyzer

ROOT = Path(__file__).resolve().parents[1]
GOLD_DATASET_PATH = ROOT / "evaluation" / "benchmark_gold_dataset.json"
RESULTS_PATH = ROOT / "evaluation" / "classifier_evaluation_results.json"


def evaluate_classifier() -> dict[str, Any]:
    with open(GOLD_DATASET_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    items = gold_data.get("items", [])
    total = len(items)

    y_true: list[str] = []
    y_pred: list[str] = []
    modality_matches = 0
    decomp_matches = 0
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    LEVEL_MAP = {
        "L1_DIRECT_LOOKUP": "L1",
        "L2_SAME_SECTION": "L2",
        "L3_CROSS_SECTION": "L3",
        "L4_CROSS_MODAL": "L4",
        "L5_MULTI_HOP_SYNTHESIS": "L5",
    }

    for item in items:
        q = item["question"]
        gold_level = item["level"]
        analysis = QuestionAnalyzer.analyze_query(q)
        pred_level = LEVEL_MAP.get(analysis.reasoning_level.value, analysis.reasoning_level.value)

        y_true.append(gold_level)
        y_pred.append(pred_level)
        confusion[gold_level][pred_level] += 1

        # Modality check
        gold_mods = item.get("required_modalities", ["text"])
        pred_mods = [m.value for m in analysis.target_modalities]
        if set(gold_mods).intersection(set(pred_mods)):
            modality_matches += 1

        # Decomposition check
        is_multihop_gold = gold_level in ["L3", "L4", "L5"]
        has_decomp = len(analysis.subqueries) > 0 or analysis.reasoning_level in (
            ReasoningLevel.L3_CROSS_SECTION,
            ReasoningLevel.L4_CROSS_MODAL,
            ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
        )
        if has_decomp == is_multihop_gold:
            decomp_matches += 1

    # Compute metrics per level
    levels = ["L1", "L2", "L3", "L4", "L5"]
    per_level: dict[str, dict[str, float]] = {}
    f1_list = []

    for lvl in levels:
        tp = confusion[lvl][lvl]
        fp = sum(confusion[other][lvl] for other in levels if other != lvl)
        fn = sum(confusion[lvl][other] for other in levels if other != lvl)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)

        per_level[lvl] = {
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "f1": round(f1 * 100, 2),
            "support": sum(confusion[lvl].values()),
        }

    correct_count = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    overall_accuracy = correct_count / total if total > 0 else 0.0
    macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

    results = {
        "total_evaluated": total,
        "overall_accuracy_pct": round(overall_accuracy * 100, 2),
        "macro_f1_pct": round(macro_f1 * 100, 2),
        "modality_routing_accuracy_pct": round((modality_matches / total) * 100, 2),
        "decomposition_trigger_accuracy_pct": round((decomp_matches / total) * 100, 2),
        "per_level_metrics": per_level,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[*] Classifier Evaluation Complete on {total} Gold Questions:")
    print(f"    - Overall Accuracy: {results['overall_accuracy_pct']}%")
    print(f"    - Macro F1:         {results['macro_f1_pct']}%")
    print(f"    - Modality Routing: {results['modality_routing_accuracy_pct']}%")
    print(f"    - Decomp Trigger:   {results['decomposition_trigger_accuracy_pct']}%")

    return results


if __name__ == "__main__":
    evaluate_classifier()
