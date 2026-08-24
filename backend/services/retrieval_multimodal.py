from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from backend.services.retrieval_service import _bm25_scores, tokenize


class RetrievalCandidate(BaseModel):
    chunk_id: str
    text: str
    page: int
    section_title: str
    section_path: list[str] = Field(default_factory=list)
    chunk_type: str
    is_figure_chunk: bool = False
    figure_id: str | None = None
    label: str | None = None
    image_file: str | None = None
    bbox: dict[str, Any] | None = None
    bm25_score: float = 0.0
    bm25_rank: int = 999
    dense_score: float = 0.0
    dense_rank: int = 999
    rrf_score: float = 0.0
    final_rank: int = 999


class MultimodalHybridRetriever:
    """Unified hybrid retrieval combining lexical BM25, dense semantic representations, and RRF fusion."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def _compute_dense_similarity(self, query_tokens: list[str], chunk_tokens: list[str]) -> float:
        """Lightweight normalized pseudo-dense vector cosine similarity for exact on-device retrieval."""
        if not query_tokens or not chunk_tokens:
            return 0.0
        q_set = set(query_tokens)
        c_set = set(chunk_tokens)
        shared = q_set.intersection(c_set)
        if not shared:
            return 0.0
        # Cosine-like overlap with length damping
        return len(shared) / math.sqrt(len(q_set) * len(c_set))

    def retrieve_hybrid(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        text_limit: int = 6,
        visual_limit: int = 2,
    ) -> list[dict[str, Any]]:
        """Perform hybrid BM25 + Dense RRF fusion retrieval."""
        if not chunks:
            return []

        q_tokens = tokenize(query)
        bm25_scores_list = _bm25_scores(query, chunks)

        # 1. Lexical BM25 ranking
        lexical_candidates: list[tuple[int, float, dict[str, Any]]] = []
        for idx, (score, chunk) in enumerate(zip(bm25_scores_list, chunks)):
            lexical_candidates.append((idx, score, chunk))
        lexical_candidates.sort(key=lambda x: x[1], reverse=True)

        # 2. Dense semantic ranking
        dense_candidates: list[tuple[int, float, dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            # Use retrieval_text if present, fallback to text
            target_text = chunk.get("retrieval_text") or chunk.get("text", "")
            c_tokens = tokenize(target_text)
            d_score = self._compute_dense_similarity(q_tokens, c_tokens)
            dense_candidates.append((idx, d_score, chunk))
        dense_candidates.sort(key=lambda x: x[1], reverse=True)

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF(d) = sum( 1 / (k + rank) )
        rrf_map: dict[int, float] = {}
        bm25_ranks: dict[int, int] = {}
        dense_ranks: dict[int, int] = {}

        for rank, (idx, _, _) in enumerate(lexical_candidates, start=1):
            bm25_ranks[idx] = rank
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (idx, _, _) in enumerate(dense_candidates, start=1):
            dense_ranks[idx] = rank
            rrf_map[idx] = rrf_map.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort all chunks by fused RRF score
        sorted_indices = sorted(rrf_map.keys(), key=lambda i: rrf_map[i], reverse=True)

        results: list[dict[str, Any]] = []
        text_count = 0
        visual_count = 0

        for final_rank, idx in enumerate(sorted_indices, start=1):
            chunk = chunks[idx]
            is_fig = chunk.get("is_figure_chunk", False)

            if is_fig and visual_count >= visual_limit:
                continue
            if not is_fig and text_count >= text_limit:
                continue

            annotated_chunk = {
                **chunk,
                "bm25_score": round(bm25_scores_list[idx], 4),
                "bm25_rank": bm25_ranks.get(idx, 999),
                "dense_score": round(self._compute_dense_similarity(q_tokens, tokenize(chunk.get("text", ""))), 4),
                "dense_rank": dense_ranks.get(idx, 999),
                "rrf_score": round(rrf_map[idx], 5),
                "final_rank": final_rank,
            }
            results.append(annotated_chunk)

            if is_fig:
                visual_count += 1
            else:
                text_count += 1

            if text_count >= text_limit and visual_count >= visual_limit:
                break

        return results
