"""Multi-Document Cross-Paper Reasoning Engine for ScholAR.

Orchestrates multi-hop reasoning across 2 or more research papers:
- Multi-document evidence pooling and canonical chunk attribution
- Cross-paper conceptual bridging (Architectural evolution, comparative benchmarks, methodology adaptations)
- Synthesizes unified multi-document Evidence Graphs and Reasoning Paths
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
from backend.services.evidence_graph_service import EvidenceGraphService
from backend.services.multi_hop_service import MultiHopRetrievalService
from backend.services.pdf_service import paper_dir, read_json

logger = logging.getLogger("scholar.cross_doc")


ID_ALIASES = {
    "attention_vaswani_2017": "1706.03762",
    "latent_diffusion_rombach_2022": "2112.10752",
    "adam_kingma_2014": "1412.6980",
    "gan_goodfellow_2014": "1406.2661",
    "vision_llm_v2_2024": "2406.08394",
    "beir_zeroshot_2021": "2104.08663",
    "interdoc_multihop_2026": "2603.14257",
    "crossdoc_multientity_2025": "2025.emnlp-main.77",
    "multimodal_multidoc_2024": "yale_thesis_1003",
    "paperqa2_2024": "2410.00526",
}


class CrossDocumentReasoningService:
    """Executes multi-hop reasoning across multiple scientific papers simultaneously."""

    @classmethod
    def synthesize_cross_document_reasoning(
        cls,
        query: str,
        primary_paper_id: str,
        secondary_paper_ids: list[str],
        analysis: QuestionAnalysis | None = None,
    ) -> tuple[EvidenceGraph, ReasoningPath, list[dict[str, Any]]]:
        """Load chunks across all specified papers and build a cross-paper EvidenceGraph."""
        resolved_primary = ID_ALIASES.get(primary_paper_id, primary_paper_id)
        resolved_secondary = [ID_ALIASES.get(pid, pid) for pid in secondary_paper_ids]
        all_paper_ids = [resolved_primary] + [pid for pid in resolved_secondary if pid != resolved_primary]
        pooled_chunks: list[dict[str, Any]] = []

        for pid in all_paper_ids:
            chunks_file = paper_dir(pid) / "chunks.json"
            if chunks_file.exists():
                chunks = read_json(chunks_file)
                for c in chunks:
                    c_tagged = dict(c)
                    c_tagged["document_id"] = pid
                    c_tagged["source_paper_id"] = pid
                    pooled_chunks.append(c_tagged)

        if not pooled_chunks:
            empty_graph = EvidenceGraph(query=query)
            empty_path = ReasoningPath(query=query, reasoning_level="L5_MULTI_HOP_SYNTHESIS", graph=empty_graph)
            return empty_graph, empty_path, []

        from backend.services.question_analyzer import QuestionAnalyzer
        analysis = analysis or QuestionAnalyzer.analyze_query(query)

        # Retrieve cross-document candidates
        retrieved_chunks, analysis = MultiHopRetrievalService.execute_multi_hop_retrieval(
            query=query,
            chunks=pooled_chunks,
            limit=8,
            analysis=analysis,
        )

        # Build unified graph
        graph, path = EvidenceGraphService.build_evidence_graph(query, retrieved_chunks, analysis)

        # Infer Cross-Document Comparative Edges between different papers
        cross_doc_edges: list[EvidenceEdge] = []
        for i in range(len(graph.nodes)):
            n1 = graph.nodes[i]
            for j in range(i + 1, len(graph.nodes)):
                n2 = graph.nodes[j]
                if n1.document_id != n2.document_id:
                    cross_doc_edges.append(EvidenceEdge(
                        source_id=n1.node_id,
                        target_id=n2.node_id,
                        relation=EvidenceRelation.SUPPORTS_MECHANISM,
                        description=f"Cross-paper bridge: {n1.document_id} ({n1.section}) <-> {n2.document_id} ({n2.section})",
                    ))

        graph.edges.extend(cross_doc_edges)

        logger.info(
            "Synthesized cross-document reasoning across %d papers: %d nodes, %d edges",
            len(all_paper_ids), len(graph.nodes), len(graph.edges)
        )
        return graph, path, retrieved_chunks
