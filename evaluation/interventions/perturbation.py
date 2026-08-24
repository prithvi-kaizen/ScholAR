from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field


class InterventionResult(BaseModel):
    intervention_id: str
    intervention_type: str  # "TEXT_VALUE_EDIT" | "REGION_MASK"
    original_fact: str
    perturbed_fact: str
    expected_behavior: str
    model_prediction: str
    followed_intervention: bool


class EvidencePerturbationRunner:
    """Runs Text (TESR) and Visual (VESR) evidence sensitivity stress tests."""

    @classmethod
    def test_text_sensitivity(
        cls,
        original_chunk_text: str,
        question: str,
        target_entity: str,
        original_value: str,
        perturbed_value: str,
        generate_fn: Any,
    ) -> InterventionResult:
        """Perturb numeric/factual values in text and measure whether the model follows altered evidence."""
        # Replace value in context
        perturbed_text = re.sub(re.escape(original_value), perturbed_value, original_chunk_text)

        # Generate answer with perturbed context
        prediction = generate_fn(question, perturbed_text)

        # Check if perturbed value appears in prediction
        followed = perturbed_value.lower() in prediction.lower()

        return InterventionResult(
            intervention_id=f"TESR_{target_entity}",
            intervention_type="TEXT_VALUE_EDIT",
            original_fact=f"{target_entity} = {original_value}",
            perturbed_fact=f"{target_entity} = {perturbed_value}",
            expected_behavior="ANSWER_FOLLOWS_PERTURBED_EVIDENCE",
            model_prediction=prediction,
            followed_intervention=followed,
        )

    @classmethod
    def compute_tesr(cls, results: list[InterventionResult]) -> float:
        """Compute Text Evidence Sensitivity Rate (TESR)."""
        if not results:
            return 0.0
        text_results = [r for r in results if r.intervention_type == "TEXT_VALUE_EDIT"]
        if not text_results:
            return 0.0
        followed_count = sum(1 for r in text_results if r.followed_intervention)
        return round(followed_count / len(text_results), 4)

    @classmethod
    def test_visual_sensitivity(
        cls,
        figure_label: str,
        original_caption: str,
        question: str,
        target_entity: str,
        original_value: str,
        perturbed_value: str,
        generate_fn: Any,
    ) -> InterventionResult:
        """Perturb visual figure metadata or caption value and test whether model follows perturbed visual evidence."""
        perturbed_caption = re.sub(re.escape(original_value), perturbed_value, original_caption)
        prediction = generate_fn(question, perturbed_caption)
        followed = perturbed_value.lower() in prediction.lower()

        return InterventionResult(
            intervention_id=f"VESR_{figure_label}_{target_entity}",
            intervention_type="VISUAL_VALUE_EDIT",
            original_fact=f"{figure_label} {target_entity} = {original_value}",
            perturbed_fact=f"{figure_label} {target_entity} = {perturbed_value}",
            expected_behavior="ANSWER_FOLLOWS_PERTURBED_VISUAL_EVIDENCE",
            model_prediction=prediction,
            followed_intervention=followed,
        )

    @classmethod
    def compute_vesr(cls, results: list[InterventionResult]) -> float:
        """Compute Visual Evidence Sensitivity Rate (VESR)."""
        if not results:
            return 0.0
        vis_results = [r for r in results if r.intervention_type == "VISUAL_VALUE_EDIT"]
        if not vis_results:
            return 0.0
        followed_count = sum(1 for r in vis_results if r.followed_intervention)
        return round(followed_count / len(vis_results), 4)
