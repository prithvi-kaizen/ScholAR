"""Multi-Level Question Analyzer & Bounded Query Decomposer for ScholAR.

Classifies incoming queries into the 5-Level Reasoning Taxonomy:
- L1: Direct Lookup
- L2: Same-Section Reasoning
- L3: Cross-Section Reasoning
- L4: Cross-Modal Reasoning
- L5: Multi-Hop Synthesis

Decomposes complex queries (L3-L5) into bounded atomic subqueries (max 3)
with explicit target modalities and section routing.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.schemas.reasoning import (
    QuestionAnalysis,
    ReasoningLevel,
    SubQuery,
    TargetModality,
)

logger = logging.getLogger("scholar.analyzer")

# Lexical trigger patterns
_ARITHMETIC_PATTERNS = [
    r"\b(percentage|percent|%|difference|ratio|margin|increase|decrease|drop|gain|how much (better|faster|higher|lower)|speedup)\b",
    r"\b(compare (accuracy|bleu|f1|score|loss|latency)|outperform by)\b",
]

_VISUAL_PATTERNS = [
    r"\b(figure|fig\.?|plot|chart|curve|loss curve|diagram|attention map|heatmap|visual|schematic|architecture diagram)\b",
    r"\b(shown in figure|illustrated in|visualize|attention distribution|convergence curve|pipeline structure|tradeoff)\b",
]

_TABLE_PATTERNS = [
    r"\b(table|tab\.?|row|column|benchmark results|leaderboard|ablation table|score in table)\b",
    r"\b(bleu|rouge|meteor|f1|f1-score|perplexity|error rate|accuracy|flops|gflops|tflops|latency|throughput|parameters|param count)\b",
    r"\b(learning rate|warmup|warmup steps|weight decay|dropout|attention dropout|d_model|d_ff|d_k|d_v|hidden size|batch size)\b",
    r"\b(wmt|glue|superglue|squad|imagenet|cifar|coco|mmlu|gsm8k|ptb|penn treebank)\b",
]

_COMPARATIVE_PATTERNS = [
    r"\b(why\s+(?:does|do|is|are)|how\s+(?:does|do|is|are)|compare|comparing|comparison|contrast|trade-?off|advantages? over|difference between|versus|vs\.?)\b",
    r"\b(outperform|ablation|support the (?:claim|hypothesis|main claim)|connect .+ with|based on)\b",
]

_DIRECT_LOOKUP_PATTERNS = [
    r"\b(what learning rate|which optimizer|batch size|how many epochs|what dataset|who are the authors|github url|code available)\b",
]


def _extract_clean_concept(query: str) -> str:
    cleaned = re.sub(
        r"^(?:how does|how do|what is|what are|why does|why do|can we|compare|comparing|what happens when|what happens to|in terms of|according to|explain)\s+",
        "",
        query.strip(),
        flags=re.IGNORECASE,
    ).rstrip("?. ")
    return cleaned or query


class QuestionAnalyzer:
    """Analyzes scientific questions and constructs bounded multi-level reasoning plans."""

    @classmethod
    def analyze_query(cls, query: str) -> QuestionAnalysis:
        """Analyze reasoning complexity, modalities, arithmetic need, and subqueries."""
        q_clean = query.strip()
        lowered = q_clean.lower()

        requires_arithmetic = any(bool(re.search(pat, lowered)) for pat in _ARITHMETIC_PATTERNS)
        requires_visual = any(bool(re.search(pat, lowered)) for pat in _VISUAL_PATTERNS)
        requires_table = any(bool(re.search(pat, lowered)) for pat in _TABLE_PATTERNS) or requires_arithmetic
        is_comparative = any(bool(re.search(pat, lowered)) for pat in _COMPARATIVE_PATTERNS)
        is_direct_lookup = any(bool(re.search(pat, lowered)) for pat in _DIRECT_LOOKUP_PATTERNS)

        # 1. Determine Target Modalities
        modalities: list[TargetModality] = []
        if requires_table and requires_visual:
            modalities = [TargetModality.MULTIMODAL, TargetModality.TABLE, TargetModality.FIGURE]
        elif requires_table:
            modalities = [TargetModality.TEXT, TargetModality.TABLE]
        elif requires_visual:
            modalities = [TargetModality.TEXT, TargetModality.FIGURE]
        else:
            modalities = [TargetModality.TEXT]

        # 2. Determine Reasoning Level (L1 to L5)
        level: ReasoningLevel
        rationale: str

        if is_direct_lookup:
            level = ReasoningLevel.L1_DIRECT_LOOKUP
            rationale = "Question asks for a direct factual lookup, hyperparameter, or named entity."
        elif is_comparative and (requires_table or requires_visual or "why" in lowered):
            level = ReasoningLevel.L5_MULTI_HOP_SYNTHESIS
            rationale = "Question requires multi-hop synthesis across architecture, ablation, and comparative results."
        elif requires_table or requires_visual:
            level = ReasoningLevel.L4_CROSS_MODAL
            rationale = "Question requires cross-modal verification between textual narrative and tabular/visual data."
        elif any(sec in lowered for sec in ("method", "experiment", "result", "ablation", "limitation", "conclusion")) and is_comparative:
            level = ReasoningLevel.L3_CROSS_SECTION
            rationale = "Question requires cross-section evidence synthesis across methodology and empirical results."
        elif "why" in lowered or "how" in lowered or "explain" in lowered or "reason" in lowered:
            level = ReasoningLevel.L2_SAME_SECTION
            rationale = "Question asks for explanatory or contextual reasoning within document context."
        else:
            level = ReasoningLevel.L1_DIRECT_LOOKUP
            rationale = "Question asks for a direct factual lookup, hyperparameter, or named entity."

        # 3. Generate Bounded Atomic SubQueries (Max 3)
        subqueries: list[SubQuery] = []
        from backend.services.routing_service import QueryDecomposer
        decomposed_targets = QueryDecomposer.decompose(q_clean)

        if decomposed_targets and len(decomposed_targets) >= 2:
            # Targeted semantic subqueries from decomposition
            for idx, target in enumerate(decomposed_targets[:3], start=1):
                target_lower = target.lower()
                is_table_target = any(term in target_lower for term in (
                    "table", "tab", "metric", "bleu", "score", "ablation", "quality", "ppl",
                    "perplexity", "flops", "latency", "accuracy", "parameter", "drop", "delta",
                    "performance", "result"
                ))
                is_fig_target = any(term in target_lower for term in (
                    "fig", "figure", "mechanism", "structure", "structural", "architecture",
                    "diagram", "projection", "layer", "flowchart", "schematic", "visual",
                    "parallel projection", "attention head", "component"
                ))
                if is_table_target and is_fig_target:
                    target_mod = TargetModality.MULTIMODAL
                    target_secs = ["Architecture", "Methodology", "Results", "Ablation"]
                elif is_table_target:
                    target_mod = TargetModality.TABLE
                    target_secs = ["Ablation", "Experiments", "Results"]
                elif is_fig_target:
                    target_mod = TargetModality.FIGURE
                    target_secs = ["Architecture", "Methodology", "Model"]
                else:
                    target_mod = TargetModality.TEXT
                    target_secs = ["Methodology", "Results", "Experiments"]

                subqueries.append(SubQuery(
                    subquery_id=f"SQ{idx}",
                    query_text=target,
                    target_sections=target_secs,
                    target_modality=target_mod,
                    priority=idx,
                ))
        elif level == ReasoningLevel.L5_MULTI_HOP_SYNTHESIS:
            clean_concept = _extract_clean_concept(q_clean)
            # Subquery 1: Methodological definition / mechanism
            subqueries.append(SubQuery(
                subquery_id="SQ1",
                query_text=f"Architectural mechanism and structural design: {clean_concept}",
                target_sections=["Methodology", "Architecture", "Introduction"],
                target_modality=TargetModality.FIGURE if requires_visual else TargetModality.TEXT,
                priority=1,
            ))
            # Subquery 2: Ablation / isolation evidence
            subqueries.append(SubQuery(
                subquery_id="SQ2",
                query_text=f"Ablation study, component variation, and metrics: {clean_concept}",
                target_sections=["Ablation", "Experiments", "Results"],
                target_modality=TargetModality.TABLE if requires_table else TargetModality.TEXT,
                priority=2,
            ))
            # Subquery 3: Quantitative comparison / final results
            subqueries.append(SubQuery(
                subquery_id="SQ3",
                query_text=f"Quantitative performance comparison and parameter tradeoffs: {clean_concept}",
                target_sections=["Results", "Experiments", "Conclusion"],
                target_modality=TargetModality.TABLE if requires_table else (TargetModality.FIGURE if requires_visual else TargetModality.TEXT),
                priority=3,
            ))

        elif level == ReasoningLevel.L4_CROSS_MODAL:
            subqueries.append(SubQuery(
                subquery_id="SQ1",
                query_text=f"Textual description and discussion: {q_clean}",
                target_sections=["Results", "Methodology", "Experiments"],
                target_modality=TargetModality.TEXT,
                priority=1,
            ))
            subqueries.append(SubQuery(
                subquery_id="SQ2",
                query_text=f"Data and metrics: {q_clean}",
                target_sections=["Results", "Experiments"],
                target_modality=TargetModality.MULTIMODAL if (requires_table and requires_visual) else (TargetModality.TABLE if requires_table else TargetModality.FIGURE),
                priority=2,
            ))

        elif level == ReasoningLevel.L3_CROSS_SECTION:
            subqueries.append(SubQuery(
                subquery_id="SQ1",
                query_text=f"Proposed method and claims: {q_clean}",
                target_sections=["Methodology", "Introduction"],
                target_modality=TargetModality.TEXT,
                priority=1,
            ))
            subqueries.append(SubQuery(
                subquery_id="SQ2",
                query_text=f"Experimental validation and results: {q_clean}",
                target_sections=["Experiments", "Results"],
                target_modality=TargetModality.TEXT,
                priority=2,
            ))

        else:
            # L1 & L2: Single atomic search
            default_mod = (
                TargetModality.MULTIMODAL if (requires_table and requires_visual)
                else (TargetModality.TABLE if requires_table
                else (TargetModality.FIGURE if requires_visual
                else TargetModality.TEXT))
            )
            subqueries.append(SubQuery(
                subquery_id="SQ1",
                query_text=q_clean,
                target_sections=[],
                target_modality=default_mod,
                priority=1,
            ))

        # Target sections summary
        all_sections: list[str] = []
        for sq in subqueries:
            for sec in sq.target_sections:
                if sec not in all_sections:
                    all_sections.append(sec)

        return QuestionAnalysis(
            original_query=q_clean,
            reasoning_level=level,
            target_modalities=modalities,
            requires_arithmetic=requires_arithmetic,
            requires_visual=requires_visual,
            target_sections=all_sections,
            subqueries=subqueries[:3],
            confidence=0.95 if level in (ReasoningLevel.L1_DIRECT_LOOKUP, ReasoningLevel.L4_CROSS_MODAL) else 0.90,
            rationale=rationale,
        )
