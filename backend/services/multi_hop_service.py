"""Multi-Hop Retrieval Pipeline for ScholAR.

Executes bounded query decomposition across multiple reasoning levels (L1 to L5):
- L1/L2: Fast single-pass retrieval
- L3/L4/L5: Multi-channel subquery execution with modality routing and evidence pooling
"""

from __future__ import annotations

import logging
from typing import Any

from backend.schemas.reasoning import (
    QuestionAnalysis,
    ReasoningLevel,
    SubQuery,
    TargetModality,
)
from backend.services.question_analyzer import QuestionAnalyzer
from backend.services.retrieval_service import retrieve_chunks

logger = logging.getLogger("scholar.multihop")


class MultiHopRetrievalService:
    """Executes multi-hop retrieval over decomposed subqueries."""

    @classmethod
    def execute_multi_hop_retrieval(
        cls,
        query: str,
        chunks: list[dict[str, Any]],
        limit: int = 6,
        paper_id: str = "",
        analysis: QuestionAnalysis | None = None,
    ) -> tuple[list[dict[str, Any]], QuestionAnalysis]:
        """Analyze question and retrieve balanced multi-level evidence."""
        analysis = analysis or QuestionAnalyzer.analyze_query(query)

        # Fast path for L1 / L2
        if analysis.reasoning_level in (ReasoningLevel.L1_DIRECT_LOOKUP, ReasoningLevel.L2_SAME_SECTION) or len(analysis.subqueries) <= 1:
            results = retrieve_chunks(
                message=query,
                chunks=chunks,
                limit=limit,
                paper_id=paper_id,
            )
            for r in results:
                r["subquery_id"] = "SQ1"
                r["reasoning_role"] = "primary_evidence"
            return results, analysis

        # Multi-Hop path for L3 / L4 / L5
        collected_chunks: list[dict[str, Any]] = []
        seen_cids: set[str] = set()
        per_subquery_limit = max(2, limit // len(analysis.subqueries) + 1)

        for sq in analysis.subqueries:
            # Filter or boost chunks by target modality if specified
            filtered_chunks = chunks
            if sq.target_modality == TargetModality.TABLE:
                table_chunks = [c for c in chunks if c.get("is_table_chunk")]
                if table_chunks:
                    filtered_chunks = table_chunks + [c for c in chunks if not c.get("is_table_chunk")]
            elif sq.target_modality == TargetModality.FIGURE:
                fig_chunks = [c for c in chunks if c.get("is_figure_chunk")]
                if fig_chunks:
                    filtered_chunks = fig_chunks + [c for c in chunks if not c.get("is_figure_chunk")]

            sq_results = retrieve_chunks(
                message=sq.query_text,
                chunks=filtered_chunks,
                limit=per_subquery_limit,
                paper_id=paper_id,
            )

            # Assign semantic role based on subquery
            role = "method_definition" if sq.subquery_id == "SQ1" else ("ablation_support" if sq.subquery_id == "SQ2" else "final_result")
            for chunk in sq_results:
                cid = str(chunk.get("chunk_id") or chunk.get("evidence_id") or id(chunk))
                if cid not in seen_cids:
                    seen_cids.add(cid)
                    c_copy = dict(chunk)
                    c_copy["subquery_id"] = sq.subquery_id
                    c_copy["reasoning_role"] = role
                    collected_chunks.append(c_copy)

        logger.info(
            "MultiHop retrieval [%s] level=%s: collected %d unique evidence blocks",
            query[:40], analysis.reasoning_level.value, len(collected_chunks)
        )
        return collected_chunks[:limit], analysis
