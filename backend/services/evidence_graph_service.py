"""Evidence Graph & Reasoning Path Construction Service for ScholAR.

Connects isolated evidence chunks into a structured directed graph:
- Methodological definitions (E1) -> Ablation evidence (E2) -> Benchmark results (E3)
- Text claims -> 2D Table cells -> Figure visual regions
- Builds transparent ReasoningPath for model synthesis and interactive UI inspection
"""

from __future__ import annotations

import logging
from typing import Any

from backend.schemas.evidence_graph import (
    EvidenceEdge,
    EvidenceGraph,
    EvidenceNode,
    EvidenceRelation,
    ReasoningPath,
    ReasoningPathStep,
)
from backend.schemas.reasoning import QuestionAnalysis, ReasoningLevel

logger = logging.getLogger("scholar.evidence_graph")


class EvidenceGraphService:
    """Constructs and traverses multi-level scientific evidence graphs."""

    @classmethod
    def build_evidence_graph(
        cls,
        query: str,
        retrieved_chunks: list[dict[str, Any]],
        analysis: QuestionAnalysis,
    ) -> tuple[EvidenceGraph, ReasoningPath]:
        """Construct a directed EvidenceGraph and ordered ReasoningPath from retrieved evidence."""
        nodes: list[EvidenceNode] = []
        edges: list[EvidenceEdge] = []
        steps: list[ReasoningPathStep] = []

        if not retrieved_chunks:
            empty_graph = EvidenceGraph(query=query, reasoning_level=analysis.reasoning_level.value)
            empty_path = ReasoningPath(query=query, reasoning_level=analysis.reasoning_level.value, graph=empty_graph)
            return empty_graph, empty_path

        # 1. Create Evidence Nodes
        for idx, chunk in enumerate(retrieved_chunks):
            eid = str(chunk.get("evidence_id") or chunk.get("chunk_id") or f"E_{idx+1:03d}")
            doc_id = str(chunk.get("document_id") or "doc")
            page_no = int(chunk.get("page", 1) or 1)
            sec = str(chunk.get("section") or "")
            mod = "table" if chunk.get("is_table_chunk") else ("visual" if chunk.get("is_figure_chunk") else "text")
            role = str(chunk.get("reasoning_role") or "primary_evidence")
            text = str(chunk.get("text") or "")
            preview = text[:140].replace("\n", " ").strip() + ("..." if len(text) > 140 else "")

            node = EvidenceNode(
                node_id=eid,
                document_id=doc_id,
                page=page_no,
                section=sec,
                modality=mod,
                text_preview=preview,
                reasoning_role=role,
                score=float(chunk.get("rerank_score") or chunk.get("rrf_score") or 1.0),
                metadata={
                    "figure_id": chunk.get("figure_id"),
                    "label": chunk.get("label"),
                    "subquery_id": chunk.get("subquery_id"),
                },
            )
            nodes.append(node)

        # 2. Sort nodes by logical reasoning order: method -> ablation -> result -> other
        role_priority = {
            "method_definition": 1,
            "ablation_support": 2,
            "final_result": 3,
            "primary_evidence": 4,
        }
        nodes.sort(key=lambda n: (role_priority.get(n.reasoning_role, 5), n.page))

        # 3. Infer Directed Edges
        for i in range(len(nodes)):
            src = nodes[i]
            for j in range(i + 1, len(nodes)):
                tgt = nodes[j]

                # Rule A: Method -> Ablation
                if src.reasoning_role == "method_definition" and tgt.reasoning_role == "ablation_support":
                    edges.append(EvidenceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        relation=EvidenceRelation.ABLATION_EVIDENCE,
                        description=f"Ablation in {tgt.section or 'experiments'} tests mechanism defined in {src.section or 'method'}",
                    ))
                # Rule B: Ablation -> Results
                elif src.reasoning_role == "ablation_support" and tgt.reasoning_role == "final_result":
                    edges.append(EvidenceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        relation=EvidenceRelation.EXPLAINS_RESULT,
                        description=f"Component ablation explains performance gain in {tgt.section or 'results'}",
                    ))
                # Rule C: Text Claim -> Table / Figure
                elif src.modality == "text" and tgt.modality in ("table", "visual"):
                    edges.append(EvidenceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        relation=EvidenceRelation.CROSS_MODAL_GROUNDING,
                        description=f"Prose narrative grounded by {tgt.modality} data on page {tgt.page}",
                    ))
                # Rule D: Adjacent Section Bridge
                elif abs(src.page - tgt.page) <= 1:
                    edges.append(EvidenceEdge(
                        source_id=src.node_id,
                        target_id=tgt.node_id,
                        relation=EvidenceRelation.SECTION_SEQUENCE,
                        description="Linear section sequence",
                    ))

        # 4. Construct Ordered ReasoningPath
        for step_idx, node in enumerate(nodes, start=1):
            contrib = ""
            if node.reasoning_role == "method_definition":
                contrib = "Establishes core architectural mechanism and proposed formula."
            elif node.reasoning_role == "ablation_support":
                contrib = "Provides controlled ablation isolating contribution of component."
            elif node.reasoning_role == "final_result":
                contrib = "Reports benchmark results and comparative empirical metrics."
            else:
                contrib = f"Provides contextual evidence from page {node.page}."

            steps.append(ReasoningPathStep(
                step_index=step_idx,
                evidence_id=node.node_id,
                section=node.section,
                page=node.page,
                modality=node.modality,
                role=node.reasoning_role,
                claim_contribution=contrib,
            ))

        graph = EvidenceGraph(
            nodes=nodes,
            edges=edges,
            graph_id=f"graph_{len(nodes)}_nodes",
            query=query,
            reasoning_level=analysis.reasoning_level.value,
        )

        path = ReasoningPath(
            query=query,
            reasoning_level=analysis.reasoning_level.value,
            steps=steps,
            graph=graph,
            synthesized_rationale=f"Constructed multi-level reasoning path across {len(steps)} evidence steps.",
        )

        logger.info(
            "Built EvidenceGraph for [%s]: %d nodes, %d edges, %d steps",
            query[:40], len(nodes), len(edges), len(steps)
        )
        return graph, path
