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
from backend.services.retrieval_service import evidence_key, retrieve_chunks

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
        from backend.services.retrieval_service import tokenize

        collected_chunks: list[dict[str, Any]] = []
        seen_evidence: set[tuple[str, str, str]] = set()
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

            # PAR-RAG Intermediate Error-Arresting Verification
            sq_tokens = set(tokenize(sq.query_text))
            combined_text = " ".join(c.get("text", "") for c in sq_results)
            evid_tokens = set(tokenize(combined_text))

            if not sq_results:
                sq.is_grounded = False
                sq.sufficiency_score = 0.0
                sq.retrieved_evidence_ids = []
                logger.warning("PAR-RAG: Subquery [%s] returned 0 chunks; branch pruned.", sq.subquery_id)
                continue

            overlap = len(sq_tokens.intersection(evid_tokens)) / max(len(sq_tokens), 1) if sq_tokens else 1.0
            is_grounded = overlap >= 0.10 or len(sq_tokens) < 3

            sq.is_grounded = is_grounded
            sq.sufficiency_score = round(float(overlap), 3)
            sq.retrieved_evidence_ids = [
                str(c.get("evidence_id") or c.get("chunk_id")) for c in sq_results
            ]

            if not is_grounded:
                logger.info(
                    "PAR-RAG: Subquery [%s] failed intermediate sufficiency (overlap=%.2f); ungrounded evidence filtered.",
                    sq.subquery_id, overlap
                )
                continue

            # Assign semantic role based on subquery
            role = "method_definition" if sq.subquery_id == "SQ1" else ("ablation_support" if sq.subquery_id == "SQ2" else "final_result")
            for chunk in sq_results:
                key = evidence_key(chunk, paper_id=paper_id)
                if key not in seen_evidence:
                    seen_evidence.add(key)
                    c_copy = dict(chunk)
                    c_copy["subquery_id"] = sq.subquery_id
                    c_copy["reasoning_role"] = role
                    c_copy["intermediate_grounded"] = is_grounded
                    collected_chunks.append(c_copy)

        logger.info(
            "MultiHop retrieval [%s] level=%s: collected %d unique evidence blocks",
            query[:40], analysis.reasoning_level.value, len(collected_chunks)
        )
        return collected_chunks[:limit], analysis
