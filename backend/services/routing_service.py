from __future__ import annotations

import re
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field

from backend.schemas.capabilities import CapabilityMode, ModelCapabilities


class QuestionRouteType(str, Enum):
    DIRECT_LOOKUP = "DIRECT_LOOKUP"
    EXPLANATION = "EXPLANATION"
    COMPARISON = "COMPARISON"
    MULTI_SECTION = "MULTI_SECTION"
    TABLE_NUMERIC = "TABLE_NUMERIC"
    FIGURE_VISUAL = "FIGURE_VISUAL"
    CHART_NUMERIC = "CHART_NUMERIC"
    MIXED_TEXT_VISUAL = "MIXED_TEXT_VISUAL"
    CODE_ALGORITHM = "CODE_ALGORITHM"
    POTENTIALLY_UNANSWERABLE = "POTENTIALLY_UNANSWERABLE"


class RouteBudget(BaseModel):
    route_type: QuestionRouteType
    text_top_k: int = 4
    visual_items: int = 0
    max_rounds: int = 1
    needs_decomposition: bool = False
    use_numeric_executor: bool = False
    requires_native_vision: bool = False
    capability_fallback: bool = False
    reason: str = ""


# Default budget allocations per question route type
_ROUTE_BUDGETS: dict[QuestionRouteType, dict[str, Any]] = {
    QuestionRouteType.DIRECT_LOOKUP: {
        "text_top_k": 4,
        "visual_items": 0,
        "max_rounds": 1,
        "needs_decomposition": False,
        "requires_native_vision": False,
    },
    QuestionRouteType.EXPLANATION: {
        "text_top_k": 5,
        "visual_items": 1,
        "max_rounds": 1,
        "needs_decomposition": False,
        "requires_native_vision": False,
    },
    QuestionRouteType.COMPARISON: {
        "text_top_k": 6,
        "visual_items": 2,
        "max_rounds": 2,
        "needs_decomposition": True,
        "requires_native_vision": False,
    },
    QuestionRouteType.MULTI_SECTION: {
        "text_top_k": 6,
        "visual_items": 2,
        "max_rounds": 2,
        "needs_decomposition": True,
        "requires_native_vision": False,
    },
    QuestionRouteType.TABLE_NUMERIC: {
        "text_top_k": 4,
        "visual_items": 2,
        "max_rounds": 1,
        "needs_decomposition": False,
        "use_numeric_executor": True,
        "requires_native_vision": True,
    },
    QuestionRouteType.FIGURE_VISUAL: {
        "text_top_k": 3,
        "visual_items": 4,
        "max_rounds": 2,
        "needs_decomposition": False,
        "requires_native_vision": True,
    },
    QuestionRouteType.CHART_NUMERIC: {
        "text_top_k": 3,
        "visual_items": 4,
        "max_rounds": 2,
        "needs_decomposition": False,
        "use_numeric_executor": True,
        "requires_native_vision": True,
    },
    QuestionRouteType.MIXED_TEXT_VISUAL: {
        "text_top_k": 4,
        "visual_items": 3,
        "max_rounds": 2,
        "needs_decomposition": True,
        "requires_native_vision": True,
    },
    QuestionRouteType.CODE_ALGORITHM: {
        "text_top_k": 6,
        "visual_items": 1,
        "max_rounds": 1,
        "needs_decomposition": False,
        "requires_native_vision": False,
    },
    QuestionRouteType.POTENTIALLY_UNANSWERABLE: {
        "text_top_k": 6,
        "visual_items": 1,
        "max_rounds": 1,
        "needs_decomposition": False,
        "requires_native_vision": False,
    },
}


class QuestionRouter:
    """Fast, deterministic question classifier and adaptive budget allocator."""

    _VISUAL_CUES = re.compile(
        r"\b(figure|fig\.?|plot|chart|graph|diagram|heatmap|curve|bar chart|scatterplot|depicted|illustrated|shown in figure)\b",
        re.IGNORECASE,
    )
    _TABLE_CUES = re.compile(
        r"\b(table|tbl\.?|row|column|accuracy on|bleu score|f1 score|percentage|increase by|decrease by|outperform|tabular)\b",
        re.IGNORECASE,
    )
    _CODE_ALGO_CUES = re.compile(
        r"\b(algorithm|pseudo-?code|code snippet|implementation|forward pass|pytorch|tensor|complexity|function|pseudocode|algorithm \d+|routine|subroutine)\b",
        re.IGNORECASE,
    )
    _COMPARISON_CUES = re.compile(
        r"\b(compare|comparison|versus|vs\.?|difference between|better than|tradeoff|relative to)\b",
        re.IGNORECASE,
    )
    _NUMERIC_CALC_CUES = re.compile(
        r"\b(how much|how many|what percentage|ratio|difference|sum|average|mean|gain of)\b",
        re.IGNORECASE,
    )
    _DIRECT_LOOKUP_CUES = re.compile(
        r"\b(what learning rate|which optimizer|batch size|how many epochs|what dataset|who are the authors|github url|code available)\b",
        re.IGNORECASE,
    )

    @classmethod
    def classify_question(cls, query: str) -> QuestionRouteType:
        """Classify question into canonical scientific query route."""
        q_lower = query.strip().lower()

        has_code = bool(cls._CODE_ALGO_CUES.search(q_lower))
        has_visual = bool(cls._VISUAL_CUES.search(q_lower))
        has_table = bool(cls._TABLE_CUES.search(q_lower))
        has_calc = bool(cls._NUMERIC_CALC_CUES.search(q_lower))
        has_comp = bool(cls._COMPARISON_CUES.search(q_lower))
        has_lookup = bool(cls._DIRECT_LOOKUP_CUES.search(q_lower))

        if has_code:
            return QuestionRouteType.CODE_ALGORITHM
        if has_visual and has_calc:
            return QuestionRouteType.CHART_NUMERIC
        if has_visual and (has_comp or "and" in q_lower):
            return QuestionRouteType.MIXED_TEXT_VISUAL
        if has_visual:
            return QuestionRouteType.FIGURE_VISUAL
        if has_table or (has_calc and "table" in q_lower):
            return QuestionRouteType.TABLE_NUMERIC
        if has_comp:
            return QuestionRouteType.COMPARISON
        if has_lookup:
            return QuestionRouteType.DIRECT_LOOKUP
        if any(term in q_lower for term in ("how does", "explain", "why", "architecture", "mechanism", "workflow")):
            return QuestionRouteType.EXPLANATION
        if any(term in q_lower for term in ("overall", "throughout the paper", "from start to finish", "across sections")):
            return QuestionRouteType.MULTI_SECTION

        return QuestionRouteType.EXPLANATION

    @classmethod
    def route(
        cls,
        query: str,
        capabilities: ModelCapabilities | None = None,
    ) -> RouteBudget:
        """Allocate compute and modality budget based on question route and model capabilities."""
        caps = capabilities or ModelCapabilities(
            model_id="qwen3.5:9b",
            display_name="Qwen 3.5",
            supports_vision=True,
        )
        route_type = cls.classify_question(query)
        base_budget = dict(_ROUTE_BUDGETS[route_type])

        requires_vision = base_budget.get("requires_native_vision", False)
        can_vision = caps.can_process_images()
        capability_fallback = False

        if requires_vision and not can_vision:
            # Model cannot receive pixels: downgrade images to 0 and boost text context
            base_budget["visual_items"] = 0
            base_budget["text_top_k"] = base_budget["text_top_k"] + 2
            capability_fallback = True
            reason = f"Question requires vision, but active model ({caps.model_id}) is in TEXT_ONLY mode. Falling back to caption and text metadata."
        else:
            reason = f"Allocated budget for {route_type.value} route."

        return RouteBudget(
            route_type=route_type,
            text_top_k=base_budget["text_top_k"],
            visual_items=base_budget["visual_items"],
            max_rounds=base_budget["max_rounds"],
            needs_decomposition=base_budget["needs_decomposition"],
            use_numeric_executor=base_budget.get("use_numeric_executor", False),
            requires_native_vision=requires_vision,
            capability_fallback=capability_fallback,
            reason=reason,
        )


class QueryDecomposer:
    """Decomposes multi-entity, multi-figure, and comparative queries into atomic subqueries."""

    @classmethod
    def decompose(cls, query: str) -> list[str]:
        """Split a complex or comparative query into distinct target subqueries."""
        q = query.strip()
        q_lower = q.lower()

        # 1. Multi-figure patterns: e.g. "figure 1 and 2", "figures 1, 2, and 3", "fig 1 vs fig 2", "figure 1 and figure 2"
        fig_nums = re.findall(r"\b(?:figures?|figs?\.?)\s*(\d+)\b", q_lower)
        if len(fig_nums) >= 2:
            return [f"Figure {n}" for n in fig_nums[:4]]
        fig_range = re.search(r"\b(?:figures?|figs?\.?)\s+([0-9\s,andvswith\&\.\-]+)", q_lower)
        if fig_range:
            nums = re.findall(r"\b\d+\b", fig_range.group(1))
            if len(nums) >= 2:
                return [f"Figure {n}" for n in nums[:4]]

        # 2. Multi-table patterns: e.g. "table 1 and 2", "tables 1, 2 and 3", "table 1 and table 2"
        tbl_nums = re.findall(r"\b(?:tables?|tbls?\.?)\s*(\d+)\b", q_lower)
        if len(tbl_nums) >= 2:
            return [f"Table {n}" for n in tbl_nums[:4]]
        tbl_range = re.search(r"\b(?:tables?|tbls?\.?)\s+([0-9\s,andvswith\&\.\-]+)", q_lower)
        if tbl_range:
            nums = re.findall(r"\b\d+\b", tbl_range.group(1))
            if len(nums) >= 2:
                return [f"Table {n}" for n in nums[:4]]

        # 3. Explicit "compare X and/with/vs Y" or "difference between X and Y"
        comp_match = re.search(
            r"(?:compare|comparison between|difference between|tradeoff between|versus|vs\.?)\s+(.+?)\s+(?:and|with|versus|vs\.?)\s+(.+?)(?:\?|\.|$)",
            q,
            re.IGNORECASE,
        )
        if comp_match:
            sub_a = comp_match.group(1).strip()
            sub_b = comp_match.group(2).strip()
            # Clean leading/trailing noise
            sub_a = re.sub(r"^(?:the|what is|how does|what are)\s+", "", sub_a, flags=re.IGNORECASE).strip()
            sub_b = re.sub(r"^(?:the|what is|how does|what are)\s+", "", sub_b, flags=re.IGNORECASE).strip()
            if sub_a and sub_b and len(sub_a) > 1 and len(sub_b) > 1:
                return [sub_a, sub_b]

        # 4. Multi-item numbered entities: e.g. "methodology 1 and methodology 2", "approach A and approach B"
        entity_match = re.search(
            r"\b([a-zA-Z_]+)\s+(\d+|[A-Z])\s+(?:and|vs\.?|with)\s+(?:\1\s+)?(\d+|[A-Z])\b",
            q,
            re.IGNORECASE,
        )
        if entity_match:
            ent_name = entity_match.group(1).strip()
            id1 = entity_match.group(2).strip()
            id2 = entity_match.group(3).strip()
            return [f"{ent_name} {id1}", f"{ent_name} {id2}"]

        # Default: single atomic query
        return [q]

