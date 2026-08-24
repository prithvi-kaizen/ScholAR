"""Capability-Adaptive Evidence Budgeting Service for ScholAR.

Dynamically manages evidence and token budgets across consumer hardware tiers:
- 8GB Tier (2K token context, max 4 blocks, 1 table, text fallback for figures)
- 16GB Tier (4K token context, max 6 blocks, 2 tables, 1 visual crop)
- 32GB+ Tier (8K token context, max 10 blocks, full multimodal graph)

Ensures critical graph bridge nodes are preserved during budget pruning.
"""

from __future__ import annotations

import logging
import os
import psutil
from typing import Any

from backend.schemas.capabilities import EvidenceBudget, HardwareTier, ModelCapabilities
from backend.schemas.evidence_graph import EvidenceGraph, ReasoningPath

logger = logging.getLogger("scholar.budgeting")


class BudgetingService:
    """Calculates and enforces dynamic evidence budgets."""

    @classmethod
    def get_hardware_tier(cls, capabilities: ModelCapabilities | None = None) -> HardwareTier:
        """Infer active hardware tier based on system RAM and model settings."""
        # Check override env var if present
        env_tier = os.getenv("SCHOLAR_HARDWARE_TIER")
        if env_tier == "8GB":
            return HardwareTier.TIER_8GB
        elif env_tier == "16GB":
            return HardwareTier.TIER_16GB
        elif env_tier == "32GB+":
            return HardwareTier.TIER_32GB_PLUS

        # Infer from psutil system memory
        try:
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if total_ram_gb < 12.0:
                return HardwareTier.TIER_8GB
            elif total_ram_gb < 24.0:
                return HardwareTier.TIER_16GB
            else:
                return HardwareTier.TIER_32GB_PLUS
        except Exception:
            return HardwareTier.TIER_16GB

    @classmethod
    def get_evidence_budget(cls, capabilities: ModelCapabilities | None = None) -> EvidenceBudget:
        """Construct the EvidenceBudget for current hardware environment."""
        tier = cls.get_hardware_tier(capabilities)
        can_vision = capabilities.can_process_images() if capabilities else True

        if tier == HardwareTier.TIER_8GB:
            return EvidenceBudget(
                hardware_tier=HardwareTier.TIER_8GB,
                max_context_tokens=2048,
                max_evidence_blocks=4,
                max_table_blocks=1,
                max_visual_crops=0,
                allow_vision_pixels=False,
            )
        elif tier == HardwareTier.TIER_16GB:
            return EvidenceBudget(
                hardware_tier=HardwareTier.TIER_16GB,
                max_context_tokens=4096,
                max_evidence_blocks=6,
                max_table_blocks=2,
                max_visual_crops=1 if can_vision else 0,
                allow_vision_pixels=can_vision,
            )
        else:
            return EvidenceBudget(
                hardware_tier=HardwareTier.TIER_32GB_PLUS,
                max_context_tokens=8192,
                max_evidence_blocks=10,
                max_table_blocks=4,
                max_visual_crops=3 if can_vision else 0,
                allow_vision_pixels=can_vision,
            )

    @classmethod
    def prune_to_budget(
        cls,
        graph: EvidenceGraph,
        path: ReasoningPath,
        budget: EvidenceBudget,
    ) -> tuple[EvidenceGraph, ReasoningPath]:
        """Prune EvidenceGraph and ReasoningPath to fit strictly within the hardware budget."""
        if len(graph.nodes) <= budget.max_evidence_blocks:
            return graph, path

        # Priority score: Bridge nodes in reasoning roles are highest priority
        role_weights = {
            "method_definition": 10.0,
            "ablation_support": 9.0,
            "final_result": 8.0,
            "primary_evidence": 5.0,
        }

        # Count tables and visuals to enforce sub-limits
        table_count = 0
        visual_count = 0
        selected_nodes = []

        # Sort candidates by role importance + relevance score
        sorted_nodes = sorted(
            graph.nodes,
            key=lambda n: role_weights.get(n.reasoning_role, 1.0) + n.score,
            reverse=True,
        )

        for node in sorted_nodes:
            if len(selected_nodes) >= budget.max_evidence_blocks:
                break
            if node.modality == "table":
                if table_count >= budget.max_table_blocks:
                    continue
                table_count += 1
            elif node.modality == "visual":
                if visual_count >= budget.max_visual_crops:
                    continue
                visual_count += 1

            selected_nodes.append(node)

        selected_ids = {n.node_id for n in selected_nodes}

        # Filter graph edges to include only edges between retained nodes
        pruned_edges = [
            e for e in graph.edges
            if e.source_id in selected_ids and e.target_id in selected_ids
        ]

        # Filter ReasoningPath steps
        pruned_steps = [
            step for step in path.steps
            if step.evidence_id in selected_ids
        ]
        # Re-index step numbers
        for idx, step in enumerate(pruned_steps, start=1):
            step.step_index = idx

        pruned_graph = EvidenceGraph(
            nodes=selected_nodes,
            edges=pruned_edges,
            graph_id=graph.graph_id,
            query=graph.query,
            reasoning_level=graph.reasoning_level,
        )

        pruned_path = ReasoningPath(
            query=path.query,
            reasoning_level=path.reasoning_level,
            steps=pruned_steps,
            graph=pruned_graph,
            synthesized_rationale=path.synthesized_rationale,
        )

        logger.info(
            "Pruned graph for [%s] to fit %s budget: %d -> %d nodes",
            graph.query[:30], budget.hardware_tier.value, len(graph.nodes), len(selected_nodes)
        )
        return pruned_graph, pruned_path
